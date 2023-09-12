# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#

import numpy as np
from tqdm import tqdm
import random
import math

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler
import itertools

from utils import recall_at_k, ndcg_k, get_metric, get_user_seqs

# class Trainer:
#     def __init__(self, generator,
#                  discriminator,
#                  train_dataloader,
#                  eval_dataloader,
#                  test_dataloader,
#                  args):
#
#         self.args = args
#         self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
#         self.device = torch.device("cuda" if self.cuda_condition else "cpu")
#
#         self.generator = generator
#         self.discriminator = discriminator
#
#         if self.cuda_condition:
#             self.generator.cuda()
#             self.discriminator.cuda()
#         # Setting the train and test data loader
#         self.train_dataloader = train_dataloader
#         self.eval_dataloader = eval_dataloader
#         self.test_dataloader = test_dataloader
#
#         # projection on discriminator output
#         # self.data_name = self.args.data_name
#         betas = (self.args.adam_beta1, self.args.adam_beta2)
#         self.optim = Adam(list(self.generator.parameters()) + list(self.discriminator.parameters()), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)
#
#         print("Total Parameters:", sum([p.nelement() for p in self.generator.parameters()] + [p.nelement() for p in self.discriminator.parameters()]))
#
#
#         self.m = nn.Softmax(dim=1)
#         self.loss_fct = nn.CrossEntropyLoss()
#
#     def train(self, epoch):
#         self.epoch = epoch
#         self.iteration(epoch, self.train_dataloader)
#
#     def valid(self, epoch, full_sort=False):
#         return self.iteration(epoch, self.eval_dataloader, full_sort=full_sort, train=False)
#
#     def test(self, epoch, full_sort=False):
#         return self.iteration(epoch, self.test_dataloader, full_sort=full_sort, train=False)
#
#     def iteration(self, epoch, dataloader, full_sort=False, train=True):
#         raise NotImplementedError
#
#     def get_sample_scores(self, epoch, pred_list):
#         pred_list = (-pred_list).argsort().argsort()[:, 0]
#         HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
#         HIT_5, NDCG_5, MRR = get_metric(pred_list, 5)
#         HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)
#         post_fix = {
#             "Epoch": epoch,
#             "HIT@1": '{:.4f}'.format(HIT_1), "NDCG@1": '{:.4f}'.format(NDCG_1),
#             "HIT@5": '{:.4f}'.format(HIT_5), "NDCG@5": '{:.4f}'.format(NDCG_5),
#             "HIT@10": '{:.4f}'.format(HIT_10), "NDCG@10": '{:.4f}'.format(NDCG_10),
#             "MRR": '{:.4f}'.format(MRR),
#         }
#         print(post_fix)
#         with open(self.args.log_file, 'a') as f:
#             f.write(str(post_fix) + '\n')
#         return [HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR], str(post_fix)
#
#     def get_full_sort_score(self, epoch, answers, pred_list):
#         recall, ndcg = [], []
#         for k in [5, 10, 15, 20]:
#             recall.append(recall_at_k(answers, pred_list, k))
#             ndcg.append(ndcg_k(answers, pred_list, k))
#         post_fix = {
#             "Epoch": epoch,
#             "HIT@5": '{:.4f}'.format(recall[0]), "NDCG@5": '{:.4f}'.format(ndcg[0]),
#             "HIT@10": '{:.4f}'.format(recall[1]), "NDCG@10": '{:.4f}'.format(ndcg[1]),
#             "HIT@20": '{:.4f}'.format(recall[3]), "NDCG@20": '{:.4f}'.format(ndcg[3])
#         }
#         print(post_fix)
#         with open(self.args.log_file, 'a') as f:
#             f.write(str(post_fix) + '\n')
#         return [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3]], str(post_fix)
#
#     def save(self, file_name):
#         torch.save(self.discriminator.cpu().state_dict(), file_name)
#         self.discriminator.to(self.device)
#
#     def load(self, file_name):
#         self.discriminator.load_state_dict(torch.load(file_name))
#
#     def _generate_sample(self, probability, pos_ids, neg_ids, neg_nums):
#         neg_ids = neg_ids.expand(probability.shape[0],-1)
#         # try:
#         neg_idxs = torch.multinomial(probability, neg_nums).to(self.device)
#
#         neg_ids = torch.gather(neg_ids, 1, neg_idxs)
#         neg_ids = neg_ids.view(-1, self.args.max_seq_length)
#         # replace the sampled positive ids with uniform sampled items
#         return neg_ids
#
#     def sample_from_generator(self, seq_out, pos_ids):
#         seq_emb = seq_out.view(-1, self.args.hidden_size)
#         istarget = (pos_ids > 0).view(pos_ids.size(0) * self.generator.args.max_seq_length).float() # [batch*seq_len]
#         K = int(self.args.item_size*self.args.item_sample_ratio) - 1
#         neg_ids = random.sample([i for i in range(1, self.args.item_size)], K)
#         neg_ids = torch.tensor(neg_ids, dtype=torch.long).to(self.device)
#         neg_emb = self.generator.item_embeddings(neg_ids)
#         full_probability = torch.matmul(seq_emb, neg_emb.transpose(0, 1))
#         full_probability = self.m(full_probability)**self.args.prob_power
#         sampled_neg_ids = self._generate_sample(full_probability, pos_ids, neg_ids, 1)
#
#         #replace certain percentage of items as absolute positive items
#         replace_idx =  (torch.rand(size=(pos_ids.size(0), pos_ids.size(1))) < (1 - self.args.sample_ratio))
#         sampled_neg_ids[replace_idx] = pos_ids[replace_idx]
#         mask_idx = torch.logical_not(replace_idx).float()
#         pos_idx = (pos_ids == sampled_neg_ids).view(pos_ids.size(0) * self.generator.args.max_seq_length).float()
#         neg_idx = (pos_ids != sampled_neg_ids).view(pos_ids.size(0) * self.generator.args.max_seq_length).float()
#         return sampled_neg_ids, pos_idx, neg_idx, mask_idx, istarget
#
#     def discriminator_cross_entropy(self, seq_out, pos_idx, neg_idx, mask_idx, istarget):
#         seq_emb = seq_out.view(-1, self.args.hidden_size)
#         # sum over feature dim
#         if self.args.project_type == 'sum':
#             neg_logits = torch.sum((seq_emb)/self.args.temperature, -1)
#         elif self.args.project_type == 'affine':
#             neg_logits = torch.squeeze(self.discriminator.dis_projection(seq_emb))
#
#         prob_score = torch.sigmoid(neg_logits) + 1e-24
#         if self.args.dis_opt_versioin == 'mask_only':
#             total_pos_loss = torch.log(prob_score) * istarget * pos_idx  * mask_idx
#             total_neg_loss = torch.log(1 - prob_score) * istarget * neg_idx * mask_idx
#         else:
#             total_pos_loss = torch.log(prob_score) * istarget * pos_idx
#             total_neg_loss = torch.log(1 - prob_score) * istarget * neg_idx
#         if self.args.dis_loss_type in ['bce']:
#             loss = torch.sum(
#                 - total_pos_loss -
#                 total_neg_loss
#             ) / (torch.sum(istarget))
#         return loss
#
#     def generator_cross_entropy(self, seq_out, pos_ids, multi_neg_ids):
#         # [batch seq_len hidden_size]
#         pos_emb = self.generator.item_embeddings(pos_ids)
#         # [batch*seq_len hidden_size]
#         pos = pos_emb.view(-1, pos_emb.size(2))
#         seq_emb = seq_out.view(-1, self.args.hidden_size) # [batch*seq_len hidden_size]
#         pos_logits = torch.sum((pos * seq_emb)/self.args.temperature, -1) # [batch*seq_len]
#         istarget = (pos_ids > 0).view(pos_ids.size(0) * self.generator.args.max_seq_length).float() # [batch*seq_len]
#
#         # handle multiple negatives
#         total_neg_loss = 0.0
#
#         if self.args.gen_loss_type in ['full-softmax']:
#             test_item_emb = self.generator.item_embeddings.weight
#             logits = torch.matmul(seq_emb, test_item_emb.transpose(0, 1))
#             loss = self.loss_fct(logits, torch.squeeze(pos_ids.view(-1)))
#         return loss
#
#     def predict_sample(self, seq_out, test_neg_sample):
#         # [batch 100 hidden_size]
#         test_item_emb = self.discriminator.item_embeddings(test_neg_sample)
#         # [batch hidden_size]
#         test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)  # [B 100]
#         return test_logits
#
#     def predict_full(self, seq_out):
#         # [item_num hidden_size]
#         test_item_emb = self.discriminator.item_embeddings.weight
#         # [batch hidden_size ]
#         rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
#         return rating_pred

# class ELECITY(Trainer):
#
#     def __init__(self, generator,
#                  discriminator,
#                  train_dataloader,
#                  eval_dataloader,
#                  test_dataloader,
#                  args):
#         super(ELECITY, self).__init__(
#             generator,
#             discriminator,
#             train_dataloader,
#             eval_dataloader,
#             test_dataloader,
#             args
#         )
#
#
#     def iteration(self, epoch, dataloader, full_sort=True, train=True):
#
#         str_code = "train" if train else "test"
#
#         # Setting the tqdm progress bar
#
#         if train:
#             # ------ model training -----#
#             print("Performing ELECITY model Training:")
#             self.generator.train()
#             self.discriminator.train()
#             joint_avg_loss = 0.0
#             gen_avg_loss = 0.0
#             dis_avg_loss = 0.0
#             print(f"rec dataset length: {len(dataloader)}")
#             rec_cf_data_iter = tqdm(enumerate(dataloader), total=len(dataloader))
#
#             # training
#             for i, (rec_batch, _) in rec_cf_data_iter:
#                 '''
#                 rec_batch shape: key_name x batch_size x feature_dim
#                 cl_batches shape:
#                     list of n_views x batch_size x feature_dim tensors
#                 '''
#                 # 0. batch_data will be sent into the device(GPU or CPU)
#                 new_rec_batch = []
#                 for t in rec_batch:
#                     if isinstance(t, list):
#                         new_neg_list = []
#                         for neg_i in t:
#                             new_neg_list.append(neg_i.to(self.device))
#                         new_rec_batch.append(new_neg_list)
#                     else:
#                         new_rec_batch.append(t.to(self.device))
#                 # try:
#                 # rec_batch = tuple(t.to(self.device) for t in rec_batch)
#                 _, input_ids, target_pos, target_neg, _ = new_rec_batch
#
#                 # ---------- generator task ---------------#
#                 sequence_output = self.generator(input_ids)
#                 gen_loss = self.generator_cross_entropy(sequence_output, target_pos, target_neg)
#
#                 # ---------- discriminator task -----------#
#                 sampled_neg_ids, pos_idx, neg_idx, mask_idx, istarget = self.sample_from_generator(sequence_output, target_pos)
#                 disc_logits = self.discriminator(sampled_neg_ids)
#                 dis_loss = self.discriminator_cross_entropy(disc_logits, pos_idx, neg_idx,  mask_idx, istarget)
#
#                 joint_loss = self.args.gen_loss_weight * gen_loss + self.args.dis_loss_weight * dis_loss
#
#                 self.optim.zero_grad()
#                 joint_loss.backward()
#                 self.optim.step()
#
#                 gen_avg_loss = gen_loss.item()
#                 dis_avg_loss = dis_loss.item()
#                 joint_avg_loss += joint_loss.item()
#             # except:
#             #     print("minor compute issue")
#
#             post_fix = {
#                 "epoch": epoch,
#                 "generator loss": '{:.4f}'.format(gen_avg_loss / len(rec_cf_data_iter)),
#                 "discriminator loss": '{:.4f}'.format(dis_avg_loss / len(rec_cf_data_iter)),
#                 "joint_avg_loss": '{:.4f}'.format(joint_avg_loss / len(rec_cf_data_iter)),
#             }
#             if (epoch + 1) % self.args.log_freq == 0:
#                 print(str(post_fix))
#
#             with open(self.args.log_file, 'a') as f:
#                 f.write(str(post_fix) + '\n')
#
#         else:
#             rec_data_iter = tqdm(enumerate(dataloader),
#                                   total=len(dataloader))
#             self.discriminator.eval()
#
#             pred_list = None
#
#             if full_sort:
#                 answer_list = None
#                 for i, org_batch in rec_data_iter:
#                     batch = []
#                     for t in org_batch:
#                         if isinstance(t, list):
#                             new_neg_list = []
#                             for neg_i in t:
#                                 new_neg_list.append(neg_i.to(self.device))
#                             batch.append(new_neg_list)
#                         else:
#                             batch.append(t.to(self.device))
#                     user_ids, input_ids, target_pos, _, answers = batch
#                     recommend_output = self.discriminator(input_ids)
#
#                     recommend_output = recommend_output[:, -1, :]
#                     # recommendation results
#
#                     rating_pred = self.predict_full(recommend_output)
#
#                     rating_pred = rating_pred.cpu().data.numpy().copy()
#                     batch_user_index = user_ids.cpu().numpy()
#                     rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
#                     # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
#                     # argpartition T: O(n)  argsort O(nlogn)
#                     ind = np.argpartition(rating_pred, -20)[:, -20:]
#                     arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
#                     arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
#                     batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]
#
#                     if i == 0:
#                         pred_list = batch_pred_list
#                         answer_list = answers.cpu().data.numpy()
#                     else:
#                         pred_list = np.append(pred_list, batch_pred_list, axis=0)
#                         answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
#                 return self.get_full_sort_score(epoch, answer_list, pred_list)

class Trainer:
    def __init__(self, generator_A,
                 generator_B,
                 generator_C,
                 discriminator_A,
                 discriminator_B,
                 discriminator_C,
                 train_dataloader,
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
        self.generator_C = generator_C
        self.discriminator_C = discriminator_C

        if self.cuda_condition:
            self.generator_A.cuda()
            self.discriminator_A.cuda()
            self.generator_B.cuda()
            self.discriminator_B.cuda()
            self.generator_C.cuda()
            self.discriminator_C.cuda()
        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        # projection on discriminator output
        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        # self.optim = Adam(list(self.generator.parameters()) + list(self.discriminator.parameters()), lr=self.args.lr,
        #                   betas=betas, weight_decay=self.args.weight_decay)
        self.optim_A = Adam(list(self.generator_A.parameters()) + list(self.discriminator_A.parameters()), lr=self.args.lr,
                           betas=betas, weight_decay=self.args.weight_decay)
        self.optim_B = Adam(list(self.generator_B.parameters()) + list(self.discriminator_B.parameters()), lr=self.args.lr,
                           betas=betas, weight_decay=self.args.weight_decay)
        self.optim_C = Adam(list(self.generator_C.parameters()) + list(self.discriminator_C.parameters()),
                            lr=self.args.lr,
                            betas=betas, weight_decay=self.args.weight_decay)




        print("A Parameters:", sum([p.nelement() for p in self.generator_A.parameters()] + [p.nelement() for p in
                                                                                              self.discriminator_A.parameters()]))
        print("B Parameters:", sum([p.nelement() for p in self.generator_B.parameters()] + [p.nelement() for p in
                                                                                            self.discriminator_B.parameters()]))


        self.m = nn.Softmax(dim=1)
        self.loss_fct = nn.CrossEntropyLoss()

    def train(self, epoch):
        self.epoch = epoch
        self.iteration(epoch, self.train_dataloader)

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
        torch.save(self.discriminator_A.cpu().state_dict(), file_name)
        self.discriminator_A.to(self.device)
        torch.save(self.discriminator_B.cpu().state_dict(), file_name)
        self.discriminator_B.to(self.device)

    def load(self, file_name):
        self.discriminator_A.load_state_dict(torch.load(file_name))
        self.discriminator_B.load_state_dict(torch.load(file_name))

    def _generate_sample(self, probability, pos_ids, neg_ids, neg_nums):
        neg_ids = neg_ids.expand(probability.shape[0], -1)
        # try:
        neg_idxs = torch.multinomial(probability, neg_nums).to(self.device)

        neg_ids = torch.gather(neg_ids, 1, neg_idxs)
        neg_ids = neg_ids.view(-1, self.args.max_seq_length)
        # replace the sampled positive ids with uniform sampled items
        return neg_ids

    def sample_from_generator(self, seq_out, pos_ids, generator):
        seq_emb = seq_out.view(-1, self.args.hidden_size)
        istarget = (pos_ids > 0).view(pos_ids.size(0) * generator.args.max_seq_length).float()  # [batch*seq_len]

        K = int(self.args.item_size * self.args.item_sample_ratio) - 1
        neg_ids = random.sample([i for i in range(1, self.args.item_size)], K)
        neg_ids = torch.tensor(neg_ids, dtype=torch.long).to(self.device)
        neg_emb = generator.item_embeddings(neg_ids)
        full_probability = torch.matmul(seq_emb, neg_emb.transpose(0, 1))
        full_probability = self.m(full_probability) ** self.args.prob_power
        sampled_neg_ids = self._generate_sample(full_probability, pos_ids, neg_ids, 1)

        # replace certain percentage of items as absolute positive items
        replace_idx = (torch.rand(size=(pos_ids.size(0), pos_ids.size(1))) < (1 - self.args.sample_ratio))
        sampled_neg_ids[replace_idx] = pos_ids[replace_idx]
        mask_idx = torch.logical_not(replace_idx).float()
        pos_idx = (pos_ids == sampled_neg_ids).view(pos_ids.size(0) * generator.args.max_seq_length).float()
        neg_idx = (pos_ids != sampled_neg_ids).view(pos_ids.size(0) * generator.args.max_seq_length).float()
        return sampled_neg_ids, pos_idx, neg_idx, mask_idx, istarget

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
        # istarget = (pos_ids > 0).view(pos_ids.size(0) * generator.args.max_seq_length).float()  # [batch*seq_len]

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

    def cycle_cross_entropy(self, seq_out_a, seq_out_b, generator_C,istarget):
        seq_emb_a = generator_C.item_embeddings(seq_out_a)
        seq_emb_b = generator_C.item_embeddings(seq_out_b)
        neg_logits_a = torch.sum((seq_emb_a) / self.args.temperature, -1)
        neg_logits_b = torch.sum((seq_emb_b) / self.args.temperature, -1)


        prob_score = torch.sigmoid(neg_logits_b - neg_logits_a) + 1e-24
        if self.args.dis_opt_versioin == 'mask_only':
            total_pos_loss = torch.log(prob_score)
            total_neg_loss = torch.log(1 - prob_score)
        else:
            total_pos_loss = torch.log(prob_score)
            total_neg_loss = torch.log(1 - prob_score)
        if self.args.dis_loss_type in ['bce']:
            loss = torch.sum(
                - total_pos_loss -
                total_neg_loss
            )/torch.sum(istarget)
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
                 generator_C,
                 discriminator_A,
                 discriminator_B,
                 discriminator_C,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader,
                 args):
        super(ELECITY, self).__init__(
            generator_A,
            generator_B,
            generator_C,
            discriminator_A,
            discriminator_B,
            discriminator_C,
            train_dataloader,
            eval_dataloader,
            test_dataloader,
            args
        )


    def iteration(self, epoch, dataloader, full_sort=True, train=True):

        str_code = "train" if train else "test"

        # Setting the tqdm progress bar

        if train:
            # ------ model training -----#
            print("Performing ELECITY model Training:")
            self.generator_A.train()
            self.generator_B.train()
            self.generator_C.train()

            self.discriminator_A.train()
            self.discriminator_B.train()
            self.discriminator_C.train()

            joint_avg_loss = 0.0

            gen_avg_loss = 0.0
            dis_avg_loss = 0.0
            cyc_avg_loss = 0.0
            cl_avg_loss = 0.0

            print(f"rec dataset length: {len(dataloader)}")
            rec_cf_data_iter = tqdm(enumerate(dataloader), total=len(dataloader))

            # training
            for i, (rec_batch, _) in rec_cf_data_iter:
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
                # try:
                # rec_batch = tuple(t.to(self.device) for t in rec_batch)
                _, input_ids, target_pos, target_neg, _ = new_rec_batch

                # ---------- generator task ---------------#

                sequence_output_A = self.generator_A(input_ids)

                sequence_output_B = self.generator_B(input_ids)
                # sequence_output_B = self.generator_B(input_ids)
                sampled_neg_ids_A, pos_idx_A, neg_idx_A, mask_idx_A, istarget_A = self.sample_from_generator(
                    sequence_output_A, target_pos, self.generator_A)

                sequence_output_C = self.generator_C(sampled_neg_ids_A)


                sampled_neg_ids_B, pos_idx, neg_idx, mask_idx, istarget = self.sample_from_generator(sequence_output_B,
                                                                                                     target_pos,
                                                                                                     self.generator_B)
                sampled_neg_ids_C, pos_idx_C, neg_idx_C, mask_idx_C, istarget_C = self.sample_from_generator(sequence_output_C,
                                                                                                     target_pos,
                                                                                                     self.generator_C)
                gen_loss = self.generator_cross_entropy(sequence_output_A, target_pos, target_neg, self.generator_A) \
                           + self.generator_cross_entropy(sequence_output_B, target_pos, target_neg, self.generator_B)\
                           + self.generator_cross_entropy(sequence_output_C, target_pos, target_neg, self.generator_C)
                cl_loss = self.CL_generator_cross_entropy(sequence_output_A, sequence_output_B, target_pos, target_neg, self.generator_A, self.generator_B)

                # ---------- discriminator task -----------#

                disc_logits_A = self.discriminator_A(sampled_neg_ids_A)
                disc_logits_B = self.discriminator_B(sampled_neg_ids_B)
                disc_logits_C = self.discriminator_C(sampled_neg_ids_C)

                dis_loss = self.discriminator_cross_entropy(disc_logits_A, pos_idx_A, neg_idx_A, mask_idx_A, istarget_A, self.discriminator_A)\
                           + self.discriminator_cross_entropy(disc_logits_B,pos_idx,neg_idx,mask_idx,istarget, self.discriminator_B) \
                           + self.discriminator_cross_entropy(disc_logits_C,pos_idx_C, neg_idx_C, mask_idx_C, istarget_C ,
                                                              self.discriminator_C)



                cyc_loss = self.cycle_cross_entropy(
                    sampled_neg_ids_C, input_ids, self.generator_C, istarget_C)

#0.9
                joint_loss = self.args.gen_loss_weight * gen_loss + self.args.dis_loss_weight * dis_loss + cyc_loss * 0.1 + cl_loss * 0.55
                self.optim_A.zero_grad()
                self.optim_B.zero_grad()
                self.optim_C.zero_grad()
                joint_loss.backward()
                self.optim_A.step()
                self.optim_B.step()
                self.optim_C.step()

                gen_avg_loss = gen_loss.item()
                cl_avg_loss = cl_loss.item()
                dis_avg_loss = dis_loss.item()
                cyc_avg_loss = cyc_loss.item()


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

