
import numpy as np
from tqdm import tqdm
import random
import math

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler
import itertools

from models import KMeans

from utils import recall_at_k, ndcg_k, get_metric, get_user_seqs, nCr
from modules import NCELoss, NTXent, SupConLoss, PCLoss
from mudule_wes import wasserstein_distance, kl_distance, wasserstein_distance_matmul, d2s_gaussiannormal, d2s_1overx, kl_distance_matmul, WassersteinNCELoss



class Trainer:
    def __init__(self, generator_A,
                 generator_B,

                 discriminator_A,
                 discriminator_B,
                 train_dataloader,
                 cluster_dataloader,
                 eval_dataloader,
                 test_dataloader,
                 args):

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.generator_A = generator_A
        self.discriminator_A = discriminator_A
        self.generator_B = generator_B
        self.discriminator_B = discriminator_B

        self.num_intent_clusters = [int(i) for i in self.args.num_intent_clusters.split(",")]
        self.clusters = []
        for num_intent_cluster in self.num_intent_clusters:
            # initialize Kmeans
            if self.args.seq_representation_type == "mean":
                cluster = KMeans(
                    num_cluster=num_intent_cluster,
                    seed=self.args.seed,
                    hidden_size=self.args.hidden_size,
                    gpu_id=self.args.gpu_id,
                    device=self.device,
                )
                self.clusters.append(cluster)
            else:
                cluster = KMeans(
                    num_cluster=num_intent_cluster,
                    seed=self.args.seed,
                    hidden_size=self.args.hidden_size * self.args.max_seq_length,
                    gpu_id=self.args.gpu_id,
                    device=self.device,
                )
                self.clusters.append(cluster)


        self.total_augmentaion_pairs = nCr(self.args.n_views, 2)


        if self.cuda_condition:
            self.generator_A.cuda()
            self.discriminator_A.cuda()
            self.generator_B.cuda()
            self.discriminator_B.cuda()

        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.cluster_dataloader = cluster_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        betas = (self.args.adam_beta1, self.args.adam_beta2)

        self.optim_A = Adam(list(self.generator_A.parameters()) + list(self.discriminator_A.parameters()), lr=self.args.lr,
                           betas=betas, weight_decay=self.args.weight_decay)
        self.optim_B = Adam(list(self.generator_B.parameters()) + list(self.discriminator_B.parameters()), lr=self.args.lr,
                           betas=betas, weight_decay=self.args.weight_decay)


        print("A Parameters:", sum([p.nelement() for p in self.generator_A.parameters()] + [p.nelement() for p in
                                                                                              self.discriminator_A.parameters()]))


        self.m = nn.Softmax(dim=1)
        self.loss_fct = nn.CrossEntropyLoss()
        self.cf_criterion = NCELoss(self.args.temperature, self.device)
        self.pcl_criterion = PCLoss(self.args.temperature, self.device)

    def train(self, epoch):
        self.epoch = epoch
        self.iteration(epoch, self.train_dataloader, self.cluster_dataloader)

    def valid(self, epoch, full_sort=False):
        return self.iteration(epoch, self.eval_dataloader, full_sort=full_sort, train=False)

    def test(self, epoch, full_sort=False):
        return self.iteration(epoch, self.test_dataloader, full_sort=full_sort, train=False)

    def iteration(self, epoch, dataloader, full_sort=False, train=True):
        raise NotImplementedError

    def get_sample_scores(self, epoch, pred_list):
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
        HIT_5, NDCG_5, MRR = get_metric(pred_list, 5)
        HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)
        post_fix = {
            "Epoch": epoch,
            "HIT@1": '{:.4f}'.format(HIT_1), "NDCG@1": '{:.4f}'.format(NDCG_1),
            "HIT@5": '{:.4f}'.format(HIT_5), "NDCG@5": '{:.4f}'.format(NDCG_5),
            "HIT@10": '{:.4f}'.format(HIT_10), "NDCG@10": '{:.4f}'.format(NDCG_10),
            "MRR": '{:.4f}'.format(MRR),
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR], str(post_fix)

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10, 15, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HIT@5": '{:.4f}'.format(recall[0]), "NDCG@5": '{:.4f}'.format(ndcg[0]),
            "HIT@10": '{:.4f}'.format(recall[1]), "NDCG@10": '{:.4f}'.format(ndcg[1]),
            "HIT@20": '{:.4f}'.format(recall[3]), "NDCG@20": '{:.4f}'.format(ndcg[3])
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3]], str(post_fix)

    def save(self, file_name):
        # torch.save(self.discriminator_A.cpu().state_dict(), file_name)
        # self.discriminator_A.to(self.device)
        torch.save(self.discriminator_B.cpu().state_dict(), file_name)
        self.discriminator_B.to(self.device)

    def load(self, file_name):
        # self.discriminator_A.load_state_dict(torch.load(file_name))
        self.discriminator_B.load_state_dict(torch.load(file_name))

    def _generate_sample(self, probability, pos_ids, neg_ids, neg_nums):
        neg_ids = neg_ids.expand(probability.shape[0], -1)
        # try:
        neg_idxs = torch.multinomial(probability, neg_nums).to(self.device)

        neg_ids = torch.gather(neg_ids, 1, neg_idxs)
        neg_ids = neg_ids.view(-1, self.args.max_seq_length)
        # replace the sampled positive ids with uniform sampled items
        return neg_ids


    def sample_from_generator(self, seq_out_A, seq_out_B, pos_ids, generator_A, generator_B):
        # 同时对两个generator进行采样，从而确保对应的item对比
        seq_emb_A = seq_out_A.view(-1, self.args.hidden_size)
        seq_emb_B = seq_out_B.view(-1, self.args.hidden_size)
        istarget_A = (pos_ids > 0).view(pos_ids.size(0) * generator_A.args.max_seq_length).float()  # [batch*seq_len]
        istarget_B = (pos_ids > 0).view(pos_ids.size(0) * generator_B.args.max_seq_length).float()  # [batch*seq_len]

        K = int(self.args.item_size * self.args.item_sample_ratio) - 1
        neg_ids = random.sample([i for i in range(1, self.args.item_size)], K)
        neg_ids = torch.tensor(neg_ids, dtype=torch.long).to(self.device)
        neg_emb_A = generator_A.item_embeddings(neg_ids)
        neg_emb_B = generator_B.item_embeddings(neg_ids)
        full_probability_A = torch.matmul(seq_emb_A, neg_emb_A.transpose(0, 1))
        full_probability_A = self.m(full_probability_A) ** self.args.prob_power
        full_probability_B = torch.matmul(seq_emb_B, neg_emb_B.transpose(0, 1))
        full_probability_B = self.m(full_probability_B) ** self.args.prob_power
        sampled_neg_ids_A = self._generate_sample(full_probability_A, pos_ids, neg_ids, 1)
        sampled_neg_ids_B = self._generate_sample(full_probability_B, pos_ids, neg_ids, 1)

        # replace certain percentage of items as absolute positive items
        replace_idx = (torch.rand(size=(pos_ids.size(0), pos_ids.size(1))) < (1 - self.args.sample_ratio))
        sampled_neg_ids_A[replace_idx] = pos_ids[replace_idx]
        sampled_neg_ids_B[replace_idx] = pos_ids[replace_idx]
        mask_idx = torch.logical_not(replace_idx).float()
        pos_idx_A = (pos_ids == sampled_neg_ids_A).view(pos_ids.size(0) * generator_A.args.max_seq_length).float()
        neg_idx_A = (pos_ids != sampled_neg_ids_A).view(pos_ids.size(0) * generator_A.args.max_seq_length).float()
        pos_idx_B = (pos_ids == sampled_neg_ids_B).view(pos_ids.size(0) * generator_B.args.max_seq_length).float()
        neg_idx_B = (pos_ids != sampled_neg_ids_B).view(pos_ids.size(0) * generator_B.args.max_seq_length).float()

        return sampled_neg_ids_A, sampled_neg_ids_B, pos_idx_A, pos_idx_B, neg_idx_A, \
               neg_idx_B, mask_idx, istarget_A, istarget_B

    def discriminator_cross_entropy(self, seq_out, pos_idx, neg_idx, mask_idx, istarget, discriminator):
        seq_emb = seq_out.view(-1, self.args.hidden_size)
        # sum over feature dim
        if self.args.project_type == 'sum':
            neg_logits = torch.sum((seq_emb) / self.args.temperature, -1)
        elif self.args.project_type == 'affine':
            neg_logits = torch.squeeze(discriminator.dis_projection(seq_emb))

        prob_score = torch.sigmoid(neg_logits) + 1e-24
        if self.args.dis_opt_versioin == 'mask_only':
            total_pos_loss = torch.log(prob_score) * istarget * pos_idx * mask_idx
            total_neg_loss = torch.log(1 - prob_score) * istarget * neg_idx * mask_idx
        else:
            total_pos_loss = torch.log(prob_score) * istarget * pos_idx
            total_neg_loss = torch.log(1 - prob_score) * istarget * neg_idx
        if self.args.dis_loss_type in ['bce']:
            loss = torch.sum(
                - total_pos_loss -
                total_neg_loss
            ) / (torch.sum(istarget))
        return loss

    def generator_cross_entropy(self, seq_out, pos_ids, multi_neg_ids, generator):
        # [batch seq_len hidden_size]
        pos_emb = generator.item_embeddings(pos_ids)
        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        seq_emb = seq_out.view(-1, self.args.hidden_size)  # [batch*seq_len hidden_size]
        pos_logits = torch.sum((pos * seq_emb) / self.args.temperature, -1)  # [batch*seq_len]
        istarget = (pos_ids > 0).view(pos_ids.size(0) * generator.args.max_seq_length).float()  # [batch*seq_len]

        # handle multiple negatives
        total_neg_loss = 0.0

        if self.args.gen_loss_type in ['full-softmax']:
            test_item_emb = generator.item_embeddings.weight
            logits = torch.matmul(seq_emb, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, torch.squeeze(pos_ids.view(-1)))
        return loss

    def CL_generator_cross_entropy(self, seq_out_A, seq_out_B, pos_ids, multi_neg_ids, generator_A, generator_B):
        # [batch seq_len hidden_size]
        pos_emb_A = generator_A.item_embeddings(pos_ids)
        pos_emb_B = generator_B.item_embeddings(pos_ids)
        # [batch*seq_len hidden_size]
        pos_A = pos_emb_A.view(-1, pos_emb_A.size(2))
        pos_B = pos_emb_B.view(-1, pos_emb_B.size(2))
        seq_emb_A = seq_out_A.view(-1, self.args.hidden_size)  # [batch*seq_len hidden_size]
        seq_emb_B = seq_out_B.view(-1, self.args.hidden_size)
        pos_logits_A = torch.sum((pos_A * seq_emb_A) / self.args.temperature, -1)  # [batch*seq_len]
        pos_logits_B = torch.sum((pos_B * seq_emb_B) / self.args.temperature, -1)


        # handle multiple negatives
        total_neg_loss = 0.0

        if self.args.gen_loss_type in ['full-softmax']:
            test_item_emb_A = generator_A.item_embeddings.weight
            logits_A = torch.matmul(seq_emb_A, test_item_emb_A.transpose(0, 1))
            test_item_emb_B = generator_B.item_embeddings.weight
            logits_B = torch.matmul(seq_emb_B, test_item_emb_B.transpose(0, 1))
            logits = torch.sigmoid(logits_B - logits_A) + 1e-24
            loss = self.loss_fct(logits, torch.squeeze(pos_ids.view(-1)))
        return loss




    def predict_sample(self, seq_out, test_neg_sample):
        # [batch 100 hidden_size]
        test_item_emb = self.discriminator_B.item_embeddings(test_neg_sample)
        # [batch hidden_size]
        test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)  # [B 100]
        return test_logits

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.discriminator_B.item_embeddings.weight
        # [batch hidden_size ]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred


class ELECITY(Trainer):

    def __init__(self, generator_A,
                 generator_B,
                 discriminator_A,
                 discriminator_B,
                 train_dataloader,
                 cluster_dataloader,
                 eval_dataloader,
                 test_dataloader,
                 args):
        super(ELECITY, self).__init__(
            generator_A,
            generator_B,
            discriminator_A,
            discriminator_B,
            train_dataloader,
            cluster_dataloader,
            eval_dataloader,
            test_dataloader,
            args
        )

    def _instance_cl_one_pair_contrastive_learning(self, inputs, intent_ids=None):
        """
        contrastive learning given one pair sequences (batch)
        inputs: [batch1_augmented_data, batch2_augmentated_data]
        """
        cl_batch = torch.cat(inputs, dim=0)
        cl_batch = cl_batch.to(self.device)
        cl_sequence_output = self.generator_A(cl_batch)
        cl_sequence_output_B = self.generator_B(cl_batch)
        # cf_sequence_output = cf_sequence_output[:, -1, :]
        if self.args.seq_representation_instancecl_type == "mean":
            cl_sequence_output = torch.mean(cl_sequence_output, dim=1, keepdim=False)
            cl_sequence_output_B = torch.mean(cl_sequence_output, dim=1, keepdim=False)
        cl_sequence_flatten = cl_sequence_output.view(cl_batch.shape[0], -1)
        cl_sequence_flatten_B = cl_sequence_output_B.view(cl_batch.shape[0], -1)
        batch_size = cl_batch.shape[0] // 2
        cl_output_slice = torch.split(cl_sequence_flatten, batch_size)
        cl_output_slice_B = torch.split(cl_sequence_flatten_B, batch_size)
        if self.args.de_noise:
            cl_loss = self.cf_criterion(cl_output_slice[0], cl_output_slice[1], intent_ids=intent_ids)
            cl_loss_B = self.cf_criterion(cl_output_slice_B[0], cl_output_slice[1], intent_ids=intent_ids)
            cl_loss_in = self.cf_criterion(cl_output_slice[0], cl_output_slice_B[0], intent_ids=intent_ids)
            cl_loss_fanl = cl_loss + cl_loss_B + cl_loss_in
        else:
            cl_loss = self.cf_criterion(cl_output_slice[0], cl_output_slice[1], intent_ids=None)
            cl_loss_B = self.cf_criterion(cl_output_slice_B[0], cl_output_slice[1], intent_ids=None)
            cl_loss_in = self.cf_criterion(cl_output_slice[0], cl_output_slice_B[0], intent_ids=None)
            cl_loss_fanl = cl_loss + cl_loss_B + cl_loss_in
        return cl_loss_fanl

    def _pcl_one_pair_contrastive_learning(self, inputs, intents, intent_ids):
        """
        contrastive learning given one pair sequences (batch)
        inputs: [batch1_augmented_data, batch2_augmentated_data]
        intents: [num_clusters batch_size hidden_dims]
        """
        n_views, (bsz, seq_len) = len(inputs), inputs[0].shape
        cl_batch = torch.cat(inputs, dim=0)
        cl_batch = cl_batch.to(self.device)
        cl_sequence_output = self.generator_A(cl_batch)
        cl_sequence_output_B = self.generator_B(cl_batch)
        if self.args.seq_representation_type == "mean":
            cl_sequence_output = torch.mean(cl_sequence_output, dim=1, keepdim=False)
            cl_sequence_output_B = torch.mean(cl_sequence_output_B, dim=1, keepdim=False)
        cl_sequence_flatten = cl_sequence_output.view(cl_batch.shape[0], -1)
        cl_sequence_flatten_B = cl_sequence_output_B.view(cl_batch.shape[0], -1)
        cl_output_slice = torch.split(cl_sequence_flatten, bsz)
        cl_output_slice_B = torch.split(cl_sequence_flatten_B, bsz)
        if self.args.de_noise:
            cl_loss = self.pcl_criterion(cl_output_slice[0], cl_output_slice[1], intents=intents, intent_ids=intent_ids)
            cl_loss_B = self.pcl_criterion(cl_output_slice_B[0], cl_output_slice[1], intents=intents, intent_ids=intent_ids)
            cl_loss_in = self.pcl_criterion(cl_output_slice[0], cl_output_slice_B[0], intents=intents, intent_ids=intent_ids)
            cl_loss_fanl = cl_loss + cl_loss_B + cl_loss_in

        else:
            cl_loss = self.pcl_criterion(cl_output_slice[0], cl_output_slice[1], intents=intents, intent_ids=None)
            cl_loss_B = self.pcl_criterion(cl_output_slice_B[0], cl_output_slice[1], intents=intents,
                                           intent_ids=None)
            cl_loss_in = self.pcl_criterion(cl_output_slice[0], cl_output_slice_B[0], intents=intents,
                                            intent_ids=None)
            cl_loss_fanl = cl_loss + cl_loss_B + cl_loss_in

        return cl_loss_fanl

    def iteration(self, epoch, dataloader, cluster_dataloader=None, full_sort=True, train=True):

        str_code = "train" if train else "test"

        # Setting the tqdm progress bar

        if train:
            # ------ intentions clustering ----- #
            if self.args.contrast_type in ["IntentCL", "Hybrid"] and epoch >= self.args.warm_up_epoches:
                print("Preparing Clustering:")
                self.generator_A.eval()
                kmeans_training_data = []
                rec_cf_data_iter = tqdm(enumerate(cluster_dataloader), total=len(cluster_dataloader))
                for i, (rec_batch, _, _) in rec_cf_data_iter:
                    rec_batch = tuple(t.to(self.device) for t in rec_batch)
                    _, input_ids, target_pos, target_neg, _ = rec_batch
                    sequence_output_A = self.generator_A(input_ids)
                    sequence_output_B = self.generator_B(input_ids)
                    # average sum
                    if self.args.seq_representation_type == "mean":
                        sequence_output_A = torch.mean(sequence_output_A, dim=1, keepdim=False)
                        sequence_output_B = torch.mean(sequence_output_B, dim=1, keepdim=False)
                    sequence_output_A = sequence_output_A.view(sequence_output_A.shape[0], -1)
                    sequence_output_A = sequence_output_A.detach().cpu().numpy()
                    sequence_output_B = sequence_output_B.view(sequence_output_B.shape[0], -1)
                    sequence_output_B = sequence_output_B.detach().cpu().numpy()
                    kmeans_training_data.append(sequence_output_A)
                    kmeans_training_data.append(sequence_output_B)
                kmeans_training_data = np.concatenate(kmeans_training_data, axis=0)


                # train multiple clusters
                print("Training Clusters:")
                for i, cluster in tqdm(enumerate(self.clusters), total=len(self.clusters)):
                    cluster.train(kmeans_training_data)
                    self.clusters[i] = cluster

                import gc

                gc.collect()

            # ------ model training -----#
            print("Performing ELECITY model Training:")
            self.generator_A.train()
            self.generator_B.train()

            self.discriminator_A.train()
            self.discriminator_B.train()


            joint_avg_loss = 0.0

            gen_avg_loss = 0.0
            dis_avg_loss = 0.0


            print(f"rec dataset length: {len(dataloader)}")
            rec_cf_data_iter = tqdm(enumerate(dataloader), total=len(dataloader))

            # training
            for i, (rec_batch, cl_batches, seq_class_label_batches) in rec_cf_data_iter:
                '''
                rec_batch shape: key_name x batch_size x feature_dim
                cl_batches shape:
                    list of n_views x batch_size x feature_dim tensors
                '''
                # 0. batch_data will be sent into the device(GPU or CPU)
                new_rec_batch = []
                for t in rec_batch:
                    if isinstance(t, list):
                        new_neg_list = []
                        for neg_i in t:
                            new_neg_list.append(neg_i.to(self.device))
                        new_rec_batch.append(new_neg_list)
                    else:
                        new_rec_batch.append(t.to(self.device))

                _, input_ids, target_pos, target_neg, _ = new_rec_batch

                # ---------- generator task ---------------#

                sequence_output_A = self.generator_A(input_ids)

                sequence_output_B = self.generator_B(input_ids)

                sampled_neg_ids_A, sampled_neg_ids_B, pos_idx_A, pos_idx_B, neg_idx_A, neg_idx_B, \
                mask_idx, istarget_A, istarget_B = self.sample_from_generator(sequence_output_A,
                                                                              sequence_output_B,
                                                                              target_pos,
                                                                              self.generator_A,
                                                                              self.generator_B)

                gen_loss = self.generator_cross_entropy(sequence_output_A, target_pos, target_neg, self.generator_A) \
                           + self.generator_cross_entropy(sequence_output_B, target_pos, target_neg, self.generator_B)\

                cl_loss = self.CL_generator_cross_entropy(sequence_output_A, sequence_output_B, target_pos, target_neg, self.generator_A, self.generator_B)

                # ---------- discriminator task -----------#

                disc_logits_A = self.discriminator_A(sampled_neg_ids_A)
                disc_logits_B = self.discriminator_B(sampled_neg_ids_B)


                dis_loss = self.discriminator_cross_entropy(disc_logits_A, pos_idx_A, neg_idx_A, mask_idx, istarget_A,
                                                            self.discriminator_A) \
                           + self.discriminator_cross_entropy(disc_logits_B, pos_idx_B, neg_idx_B, mask_idx, istarget_B,
                                                              self.discriminator_B)
                    # ---------- contrastive learning task -------------#
                cl_losses = []
                for cl_batch in cl_batches:
                    if self.args.contrast_type == "InstanceCL":
                        cl_loss = self._instance_cl_one_pair_contrastive_learning(
                            cl_batch, intent_ids=seq_class_label_batches
                        )
                        cl_losses.append(self.args.cf_weight * cl_loss)
                    elif self.args.contrast_type == "IntentCL":
                        # ------ performing clustering for getting users' intentions ----#
                        # average sum
                        if epoch >= self.args.warm_up_epoches:
                            if self.args.seq_representation_type == "mean":
                                sequence_output = torch.mean(sequence_output, dim=1, keepdim=False)
                            sequence_output = sequence_output.view(sequence_output.shape[0], -1)
                            sequence_output = sequence_output.detach().cpu().numpy()

                            # query on multiple clusters
                            for cluster in self.clusters:
                                seq2intents = []
                                intent_ids = []
                                intent_id, seq2intent = cluster.query(sequence_output)
                                seq2intents.append(seq2intent)
                                intent_ids.append(intent_id)
                            cl_loss = self._pcl_one_pair_contrastive_learning(
                                cl_batch, intents=seq2intents, intent_ids=intent_ids
                            )
                            cl_losses.append(self.args.intent_cf_weight * cl_loss)
                        else:
                            continue
                    elif self.args.contrast_type == "Hybrid":
                        if epoch < self.args.warm_up_epoches:
                            cl_loss1 = self._instance_cl_one_pair_contrastive_learning(
                                cl_batch, intent_ids=seq_class_label_batches
                            )
                            cl_losses.append(self.args.cf_weight * cl_loss1)
                        else:
                            cl_loss1 = self._instance_cl_one_pair_contrastive_learning(
                                cl_batch, intent_ids=seq_class_label_batches
                            )
                            cl_losses.append(self.args.cf_weight * cl_loss1)
                            if self.args.seq_representation_type == "mean":
                                sequence_output = torch.mean(sequence_output_A, dim=1, keepdim=False)
                            sequence_output = sequence_output.view(sequence_output.shape[0], -1)
                            sequence_output = sequence_output.view(sequence_output.shape[0], -1)
                            sequence_output = sequence_output.detach().cpu().numpy()
                            # query on multiple clusters
                            for cluster in self.clusters:
                                seq2intents = []
                                intent_ids = []
                                intent_id, seq2intent = cluster.query(sequence_output)
                                seq2intents.append(seq2intent)
                                intent_ids.append(intent_id)
                            cl_loss3 = self._pcl_one_pair_contrastive_learning(
                                cl_batch, intents=seq2intents, intent_ids=intent_ids
                            )
                            cl_losses.append(self.args.intent_cf_weight * cl_loss3)
                            # ---------- contrastive learning task -------------#
                            cl_losses = []
                            for cl_batch in cl_batches:
                                if self.args.contrast_type == "InstanceCL":
                                    cl_loss = self._instance_cl_one_pair_contrastive_learning(
                                        cl_batch, intent_ids=seq_class_label_batches
                                    )
                                    cl_losses.append(self.args.cf_weight * cl_loss)
                                elif self.args.contrast_type == "IntentCL":
                                    # ------ performing clustering for getting users' intentions ----#
                                    # average sum
                                    if epoch >= self.args.warm_up_epoches:
                                        if self.args.seq_representation_type == "mean":
                                            sequence_output = torch.mean(sequence_output, dim=1, keepdim=False)
                                        sequence_output = sequence_output.view(sequence_output.shape[0], -1)
                                        sequence_output = sequence_output.detach().cpu().numpy()

                                        # query on multiple clusters
                                        for cluster in self.clusters:
                                            seq2intents = []
                                            intent_ids = []
                                            intent_id, seq2intent = cluster.query(sequence_output)
                                            seq2intents.append(seq2intent)
                                            intent_ids.append(intent_id)
                                        cl_loss = self._pcl_one_pair_contrastive_learning(
                                            cl_batch, intents=seq2intents, intent_ids=intent_ids
                                        )
                                        cl_losses.append(self.args.intent_cf_weight * cl_loss)
                                    else:
                                        continue
                                elif self.args.contrast_type == "Hybrid":
                                    if epoch < self.args.warm_up_epoches:
                                        cl_loss1 = self._instance_cl_one_pair_contrastive_learning(
                                            cl_batch, intent_ids=seq_class_label_batches
                                        )
                                        cl_losses.append(self.args.cf_weight * cl_loss1)
                                    else:
                                        cl_loss1 = self._instance_cl_one_pair_contrastive_learning(
                                            cl_batch, intent_ids=seq_class_label_batches
                                        )
                                        cl_losses.append(self.args.cf_weight * cl_loss1)
                                        if self.args.seq_representation_type == "mean":
                                            sequence_output = torch.mean(sequence_output_A, dim=1, keepdim=False)
                                        sequence_output = sequence_output.view(sequence_output.shape[0], -1)
                                        sequence_output = sequence_output.detach().cpu().numpy()
                                        # query on multiple clusters
                                        for cluster in self.clusters:
                                            seq2intents = []
                                            intent_ids = []
                                            intent_id, seq2intent = cluster.query(sequence_output)
                                            seq2intents.append(seq2intent)
                                            intent_ids.append(intent_id)
                                        cl_loss3 = self._pcl_one_pair_contrastive_learning(
                                            cl_batch, intents=seq2intents, intent_ids=intent_ids
                                        )
                                        cl_losses.append(self.args.intent_cf_weight * cl_loss3)

                joint_loss = self.args.gen_loss_weight * gen_loss + self.args.dis_loss_weight * dis_loss + cl_loss * 0.5



                self.optim_A.zero_grad()
                self.optim_B.zero_grad()

                joint_loss.backward()
                self.optim_A.step()
                self.optim_B.step()


                gen_avg_loss = gen_loss.item()
                cl_avg_loss = cl_loss.item()
                dis_avg_loss = dis_loss.item()



                joint_avg_loss += joint_loss.item()
            # except:
            #     print("minor compute issue")

            post_fix = {
                "epoch": epoch,
                "generator loss": '{:.4f}'.format(gen_avg_loss / len(rec_cf_data_iter)),
                "discriminator loss": '{:.4f}'.format(dis_avg_loss / len(rec_cf_data_iter)),

                "joint_avg_loss": '{:.4f}'.format(joint_avg_loss / len(rec_cf_data_iter)),
            }
            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')

        else:
            rec_data_iter = tqdm(enumerate(dataloader),
                                  total=len(dataloader))
            self.discriminator_A.eval()

            pred_list = None

            if full_sort:
                answer_list = None
                for i, org_batch in rec_data_iter:
                    batch = []
                    for t in org_batch:
                        if isinstance(t, list):
                            new_neg_list = []
                            for neg_i in t:
                                new_neg_list.append(neg_i.to(self.device))
                            batch.append(new_neg_list)
                        else:
                            batch.append(t.to(self.device))
                    user_ids, input_ids, target_pos, _, answers = batch
                    recommend_output = self.discriminator_A(input_ids)

                    recommend_output = recommend_output[:, -1, :]
                    # recommendation results

                    rating_pred = self.predict_full(recommend_output)

                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                    # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
                    # argpartition T: O(n)  argsort O(nlogn)
                    ind = np.argpartition(rating_pred, -20)[:, -20:]
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                    if i == 0:
                        pred_list = batch_pred_list
                        answer_list = answers.cpu().data.numpy()
                    else:
                        pred_list = np.append(pred_list, batch_pred_list, axis=0)
                        answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                return self.get_full_sort_score(epoch, answer_list, pred_list)


class CoDistSAModelTrainer(Trainer):

    def __init__(self, generator_A,
                 generator_B,
                 discriminator_A,
                 discriminator_B,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):
        super(CoDistSAModelTrainer, self).__init__(
            generator_A,
            generator_B,
            discriminator_A,
            discriminator_B,
            train_dataloader,
            eval_dataloader,
            test_dataloader, args
        )
        self.cf_criterion = WassersteinNCELoss(self.args.temperature, self.device)
        print("self.cf_criterion:", self.cf_criterion.__class__.__name__)

    def qualitative_analysis(self, dataloader, user_seq, args):
        rec_data_iter = tqdm(enumerate(dataloader),
                             desc=f"Recommendation Test Analysis",
                             total=len(dataloader),
                             bar_format="{l_bar}{r_bar}")
        item_freq = defaultdict(int)
        user_freq = defaultdict(int)
        for user_id, seq in enumerate(user_seq):
            train_seq = seq[:-2]
            user_freq[user_id] = len(train_seq)
            for itemid in train_seq:
                item_freq[itemid] += 1
        self.model.eval()
        pred_list = None
        all_att_scores = None
        input_seqs = None

        item_mean_emb = self.model.item_mean_embeddings.weight.cpu().data.numpy().copy()
        item_cov_emb = self.model.item_cov_embeddings.weight.cpu().data.numpy().copy()
        with torch.no_grad():
            for i, batch in rec_data_iter:
                # i = 0
                # for batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or cpu)
                batch = tuple(t.to(self.device) for t in batch)
                user_ids, input_ids, target_pos, target_neg, answers = batch
                recommend_mean_output, recommend_cov_output, att_scores = self.model.finetune(input_ids)

                recommend_mean_output = recommend_mean_output[:, -1, :]
                recommend_cov_output = recommend_cov_output[:, -1, :]

                rating_pred = self.dist_predict_full(recommend_mean_output, recommend_cov_output)

                rating_pred = rating_pred.cpu().data.numpy().copy()
                att_scores = att_scores.cpu().data.numpy().copy()

                batch_user_index = user_ids.cpu().numpy()
                rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 1e+24

                batch_pred_list = np.argsort(rating_pred, axis=1)
                pos_items = answers.cpu().data.numpy()

                partial_batch_pred_list = batch_pred_list[:, :40]
                if i == 0:
                    pred_list = partial_batch_pred_list
                    answer_list = answers.cpu().data.numpy()
                    all_att_scores = att_scores[:, 0, -30:, -30:]
                    input_seqs = input_ids.cpu().data.numpy()
                else:
                    pred_list = np.append(pred_list, partial_batch_pred_list, axis=0)
                    answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                    all_att_scores = np.append(all_att_scores, att_scores[:, 0, -30:, -30:], axis=0)
                    input_seqs = np.append(input_seqs, input_ids.cpu().data.numpy(), axis=0)
            scores, result_info, [recall_dict_list, ndcg_dict_list, mrr_dict] = self.get_full_sort_score('best',
                                                                                                         answer_list,
                                                                                                         pred_list)

        pred_details = {'ndcg': ndcg_dict_list, 'pred_list': pred_list, 'answer_list': answer_list,
                        'attentions': all_att_scores, 'user_freq': user_freq, 'item_freq': item_freq,
                        'train': input_seqs, 'embeddings': [item_mean_emb, item_cov_emb]}
        return scores, result_info, pred_details

    def eval_analysis(self, dataloader, user_seq, args):
        rec_data_iter = tqdm(enumerate(dataloader),
                             desc=f"Recommendation Analysis",
                             total=len(dataloader),
                             bar_format="{l_bar}{r_bar}")
        # rec_data_iter = dataloader
        Ks = [1, 5, 10, 15, 20, 40]
        item_freq = defaultdict(int)
        train = {}
        for user_id, seq in enumerate(user_seq):
            train_seq = seq[:-2]
            train[user_id] = train_seq
            for itemid in train_seq:
                item_freq[itemid] += 1
        freq_quantiles = np.array([3, 7, 20, 50])
        items_in_freqintervals = [[] for _ in range(len(freq_quantiles) + 1)]
        for item, freq_i in item_freq.items():
            interval_ind = -1
            for quant_ind, quant_freq in enumerate(freq_quantiles):
                if freq_i <= quant_freq:
                    interval_ind = quant_ind
                    break
            if interval_ind == -1:
                interval_ind = len(items_in_freqintervals) - 1
            items_in_freqintervals[interval_ind].append(item)

        self.model.eval()
        all_pos_items_ranks = defaultdict(list)
        pred_list = None
        with torch.no_grad():
            for i, batch in rec_data_iter:
                # i = 0
                # for batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or cpu)
                batch = tuple(t.to(self.device) for t in batch)
                user_ids, input_ids, target_pos, target_neg, answers = batch
                recommend_mean_output, recommend_cov_output, _ = self.model.finetune(input_ids)

                recommend_mean_output = recommend_mean_output[:, -1, :]
                recommend_cov_output = recommend_cov_output[:, -1, :]

                rating_pred = self.dist_predict_full(recommend_mean_output, recommend_cov_output)

                rating_pred = rating_pred.cpu().data.numpy().copy()
                batch_user_index = user_ids.cpu().numpy()
                rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 1e+24

                batch_pred_list = np.argsort(rating_pred, axis=1)
                pos_items = answers.cpu().data.numpy()

                pos_ranks = np.where(batch_pred_list == pos_items)[1] + 1
                for each_pos_item, each_rank in zip(pos_items, pos_ranks):
                    all_pos_items_ranks[each_pos_item[0]].append(each_rank)

                partial_batch_pred_list = batch_pred_list[:, :40]

                if i == 0:
                    pred_list = partial_batch_pred_list
                    answer_list = answers.cpu().data.numpy()
                else:
                    pred_list = np.append(pred_list, partial_batch_pred_list, axis=0)
                    answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                # i += 1
            scores, result_info, [recall_dict_list, ndcg_dict_list, mrr_dict] = self.get_full_sort_score('best',
                                                                                                         answer_list,
                                                                                                         pred_list)

            get_user_performance_perpopularity(train, [recall_dict_list, ndcg_dict_list, mrr_dict], Ks)
            get_item_performance_perpopularity(items_in_freqintervals, all_pos_items_ranks, Ks, freq_quantiles,
                                               args.item_size)
            return scores, result_info, None

    def _one_pair_contrastive_learning(self, inputs):
        '''
        contrastive learning given one pair sequences (batch)
        inputs: [batch1_augmented_data, batch2_augmentated_data]
        '''
        cl_batch = torch.cat(inputs, dim=0)
        cl_batch = cl_batch.to(self.device)
        cl_mean_sequence_output, cl_cov_sequence_output, _ = self.generator_A.finetune(cl_batch)
        # cf_sequence_output = cf_sequence_output[:, -1, :]
        cl_mean_sequence_flatten = cl_mean_sequence_output.view(cl_batch.shape[0], -1)
        cl_cov_sequence_flatten = cl_cov_sequence_output.view(cl_batch.shape[0], -1)
        # cf_output = self.projection(cf_sequence_flatten)
        batch_size = cl_batch.shape[0] // 2
        cl_mean_output_slice = torch.split(cl_mean_sequence_flatten, batch_size)
        cl_cov_output_slice = torch.split(cl_cov_sequence_flatten, batch_size)
        cl_loss = self.cf_criterion(cl_mean_output_slice[0], cl_cov_output_slice[0],
                                    cl_mean_output_slice[1], cl_cov_output_slice[1])
        return cl_loss

    def bpr_optimization(self, seq_mean_out, seq_cov_out, pos_ids, neg_ids):
        # [batch seq_len hidden_size]
        activation = nn.ELU()
        pos_mean_emb = self.model.item_mean_embeddings(pos_ids)

        pos_cov_emb = activation(self.model.item_cov_embeddings(pos_ids)) + 1
        neg_mean_emb = self.model.item_mean_embeddings(neg_ids)

        neg_cov_emb = activation(self.model.item_cov_embeddings(neg_ids)) + 1

        # [batch*seq_len hidden_size]
        pos_mean = pos_mean_emb.view(-1, pos_mean_emb.size(2))
        pos_cov = pos_cov_emb.view(-1, pos_cov_emb.size(2))
        neg_mean = neg_mean_emb.view(-1, neg_mean_emb.size(2))
        neg_cov = neg_cov_emb.view(-1, neg_cov_emb.size(2))
        seq_mean_emb = seq_mean_out.view(-1, self.args.hidden_size)  # [batch*seq_len hidden_size]
        seq_cov_emb = seq_cov_out.view(-1, self.args.hidden_size)  # [batch*seq_len hidden_size]

        if self.args.distance_metric == 'wasserstein':
            pos_logits = wasserstein_distance(seq_mean_emb, seq_cov_emb, pos_mean, pos_cov)
            neg_logits = wasserstein_distance(seq_mean_emb, seq_cov_emb, neg_mean, neg_cov)
            pos_vs_neg = wasserstein_distance(pos_mean, pos_cov, neg_mean, neg_cov)
        else:
            pos_logits = kl_distance(seq_mean_emb, seq_cov_emb, pos_mean, pos_cov)
            neg_logits = kl_distance(seq_mean_emb, seq_cov_emb, neg_mean, neg_cov)
            pos_vs_neg = kl_distance(pos_mean, pos_cov, neg_mean, neg_cov)

        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float()  # [batch*seq_len]
        loss = torch.sum(-torch.log(torch.sigmoid(neg_logits - pos_logits + 1e-24)) * istarget) / torch.sum(istarget)

        pvn_loss = torch.sum(torch.clamp(pos_logits - pos_vs_neg, 0) * istarget) / torch.sum(istarget)
        auc = torch.sum(
            ((torch.sign(neg_logits - pos_logits) + 1) / 2) * istarget
        ) / torch.sum(istarget)

        return loss, auc, pvn_loss

    def dist_predict_full(self, seq_mean_out, seq_cov_out):
        elu_activation = torch.nn.ELU()
        test_item_mean_emb = self.discriminator_B.item_mean_embeddings.weight
        test_item_cov_emb = elu_activation(self.discriminator_B.item_cov_embeddings.weight) + 1

        return wasserstein_distance_matmul(seq_mean_out, seq_cov_out, test_item_mean_emb, test_item_cov_emb)

    def iteration(self, epoch, dataloader, full_sort=False, train=True):

        str_code = "train" if train else "test"


        rec_data_iter = dataloader

        if train:
            # self.model.train()
            self.generator_A.train()
            self.generator_B.train()

            self.discriminator_A.train()
            self.discriminator_B.train()

            cl_individual_avg_losses = [0.0 for i in range(self.total_augmentaion_pairs)]
            cl_sum_avg_loss = 0.0
            joint_avg_loss = 0.0

            # for i, batch in rec_data_iter:
            i = 1
            for rec_batch, cl_batches in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or CPU)
                rec_batch = tuple(t.to(self.device) for t in rec_batch)
                user_ids, input_ids, target_pos, target_neg, _ = rec_batch

                # ---------- generator task ---------------#

                sequence_mean_output_A, sequence_cov_output_A, att_scores = self.generator_A.finetune(input_ids)
                sequence_mean_output_B, sequence_cov_output_B, att_scores = self.generator_B.finetune(input_ids)

                sampled_neg_ids_A, pos_idx_A, neg_idx_A, mask_idx_A, istarget_A = self.sample_from_generator(
                    sequence_mean_output_A, target_pos, self.generator_A)

                sampled_neg_ids_B, pos_idx, neg_idx, mask_idx, istarget = self.sample_from_generator(
                    sequence_mean_output_B, target_pos, self.generator_B)

                gen_loss = self.generator_cross_entropy(sequence_mean_output_A, target_pos, target_neg,
                                                        self.generator_A, type='mean') \
                           + self.generator_cross_entropy(sequence_mean_output_B, target_pos, target_neg,
                                                          self.generator_B, type='mean')
                gen_pvn_loss = self.generator_cross_entropy(sequence_cov_output_A, target_pos, target_neg,
                                                            self.generator_A, type='cov') \
                               + self.generator_cross_entropy(sequence_cov_output_B, target_pos, target_neg,
                                                              self.generator_B, type='cov')

                cl_loss_gen = self.CL_generator_cross_entropy(sequence_mean_output_A, sequence_mean_output_B,
                                                              target_pos, target_neg, self.generator_A,
                                                              self.generator_B, type='mean') \

                # ---------- discriminator task ---------------#
                disc_logits_A, disc_logits_cov_A, att_scores = self.discriminator_A.finetune(sampled_neg_ids_A)
                disc_logits_B, disc_logits_cov_B, att_scores = self.discriminator_B.finetune(sampled_neg_ids_B)

                dis_loss = self.discriminator_cross_entropy(disc_logits_A, pos_idx_A, neg_idx_A, mask_idx_A, istarget_A,
                                                            self.discriminator_A) \
                           + self.discriminator_cross_entropy(disc_logits_B, pos_idx, neg_idx, mask_idx, istarget,
                                                              self.discriminator_B)
                dis_pvn_loss = self.discriminator_cross_entropy(disc_logits_cov_A, pos_idx_A, neg_idx_A, mask_idx_A,
                                                                istarget_A,
                                                                self.discriminator_A) \
                               + self.discriminator_cross_entropy(disc_logits_cov_B, pos_idx, neg_idx, mask_idx,
                                                                  istarget,
                                                                  self.discriminator_B)

                # ---------- contrastive learning task -------------#

                joint_loss = self.args.gen_loss_weight * gen_loss + \
                             self.args.dis_loss_weight * dis_loss + \
                             cl_loss_gen * 0.5 + \
                             self.args.pvn_weight * (gen_pvn_loss + dis_pvn_loss)
                cl_losses = []
                for cl_batch in cl_batches:
                    cl_loss = self._one_pair_contrastive_learning(cl_batch)
                    cl_losses.append(cl_loss)

                for cl_loss in cl_losses:
                    # loss += self.args.cf_weight * cl_loss
                    joint_loss += self.args.cf_weight * cl_loss

                self.optim_A.zero_grad()
                self.optim_B.zero_grad()

                joint_loss.backward()
                self.optim_A.step()
                self.optim_B.step()



                gen_avg_loss = gen_loss.item()
                dis_avg_loss = dis_loss.item()
                cl_avg_loss = cl_loss_gen.item()

                joint_avg_loss += joint_loss.item()

                for i, cl_loss in enumerate(cl_losses):
                    cl_individual_avg_losses[i] += cl_loss.item()
                    cl_sum_avg_loss += self.args.cf_weight * cl_loss.item()


            post_fix = {
                "epoch": epoch,
                "total_avg_loss": '{:.4f}'.format(joint_avg_loss / len(rec_data_iter)),

            }
            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix), flush=True)

            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')

        else:
            # self.model.eval()
            self.discriminator_B.eval()

            pred_list = None

            if full_sort:
                answer_list = None
                with torch.no_grad():
                    # for i, batch in rec_data_iter:
                    i = 0
                    for batch in rec_data_iter:
                        # 0. batch_data will be sent into the device(GPU or cpu)
                        batch = tuple(t.to(self.device) for t in batch)
                        user_ids, input_ids, target_pos, target_neg, answers = batch
                        # recommend_mean_output, recommend_cov_output, _ = self.model.finetune(input_ids)
                        recommend_mean_output, recommend_cov_output, _ = self.discriminator_B.finetune(input_ids)

                        recommend_mean_output = recommend_mean_output[:, -1, :]
                        recommend_cov_output = recommend_cov_output[:, -1, :]

                        rating_pred = self.dist_predict_full(recommend_mean_output, recommend_cov_output)

                        rating_pred = rating_pred.cpu().data.numpy().copy()
                        batch_user_index = user_ids.cpu().numpy()
                        rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 1e+24
                        # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
                        ind = np.argpartition(rating_pred, 40)[:, :40]
                        # ind = np.argpartition(rating_pred, -40)[:, -40:]
                        arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                        # ascending order
                        arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::]
                        # arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                        batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                        if i == 0:
                            pred_list = batch_pred_list
                            answer_list = answers.cpu().data.numpy()
                        else:
                            pred_list = np.append(pred_list, batch_pred_list, axis=0)
                            answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                        i += 1
                    return self.get_full_sort_score(epoch, answer_list, pred_list)


