# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#

import os
import numpy as np
import random
import torch
import argparse

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from datasets import IDMARecDataset

from train_intent import ELECITY
from models import SASRecModel, SASRecUserPropmtModel, SASRecAttributeModel, SASRecUserItemPromptModel
from utils import EarlyStopping, get_user_seqs, get_item2attribute_json, check_path, set_seed


def show_args_info(args):
    print(f"--------------------Configure Info:------------")
    for arg in vars(args):
        print(f"{arg:<30} : {getattr(args, arg):>35}")


def update_to_module_parameters(args, m_type='gen'):
    if m_type == 'gen':
        args.hidden_size = args.gen_hidden_size
        args.num_hidden_layers = args.gen_num_hidden_layers
        args.num_attention_heads = args.gen_num_attention_heads
        args.hidden_act = args.gen_hidden_act
        args.attention_probs_dropout_prob = args.gen_attention_probs_dropout_prob
        args.hidden_dropout_prob = args.gen_hidden_dropout_prob
        args.initializer_range = args.gen_initializer_range
    else:
        args.hidden_size = args.dis_hidden_size
        args.num_hidden_layers = args.dis_num_hidden_layers
        args.num_attention_heads = args.dis_num_attention_heads
        args.hidden_act = args.dis_hidden_act
        args.attention_probs_dropout_prob = args.dis_attention_probs_dropout_prob
        args.hidden_dropout_prob = args.dis_hidden_dropout_prob
        args.initializer_range = args.dis_initializer_range
    return args


def main():
    parser = argparse.ArgumentParser()
    # system args
    parser.add_argument('--data_dir', default='../data/', type=str)
    parser.add_argument('--output_dir', default='output/', type=str)
    parser.add_argument('--data_name', default='Yelp', type=str)

    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--model_idx', default=1, type=int, help="model idenfier 10, 20, 30...")
    parser.add_argument("--gpu_id", type=str, default="1", help="gpu_id")
    parser.add_argument('--training_data_ratio', default=1.0, type=float, \
                        help="percentage of training samples used for training - robustness analysis")
    parser.add_argument('--noise_ratio', default=0.0, type=float, \
                        help="percentage of negative interactions in a sequence - robustness analysis")
    parser.add_argument("--model_name", default='IDMARec', type=str)

    parser.add_argument(
        "--augment_type",
        default="random",
        type=str,
        help="default data augmentation types. Chosen from: \
                           mask, crop, reorder, substitute, insert, random, \
                           combinatorial_enumerate (for multi-view).",
    )
    parser.add_argument("--tao", type=float, default=0.2, help="crop ratio for crop operator")
    parser.add_argument("--gamma", type=float, default=0.7, help="mask ratio for mask operator")
    parser.add_argument("--beta", type=float, default=0.2, help="reorder ratio for reorder operator")

    ## contrastive learning task args

    parser.add_argument(
        "--n_views", default=2, type=int, metavar="N", help="Number of augmented data for each sequence - not studied."
    )
    parser.add_argument(
        "--contrast_type",
        default="Hybrid",
        type=str,
        help="Ways to contrastive of. \
                           Support InstanceCL and ShortInterestCL, IntentCL, and Hybrid types.",
    )
    parser.add_argument(
        "--num_intent_clusters",
        default="256",
        type=str,
        help="Number of cluster of intents. Activated only when using \
                           IntentCL or Hybrid types.",
    )
    parser.add_argument(
        "--seq_representation_type",
        default="mean",
        type=str,
        help="operate of item representation overtime. Support types: \
                           mean, concatenate",
    )
    parser.add_argument(
        "--seq_representation_instancecl_type",
        default="concatenate",
        type=str,
        help="operate of item representation overtime. Support types: \
                           mean, concatenate",
    )
    parser.add_argument("--warm_up_epoches", type=float, default=0, help="number of epochs to start IntentCL.")
    parser.add_argument("--de_noise", action="store_true", help="whether to de-false negative pairs during learning.")

    # Generator args
    parser.add_argument("--gen_hidden_size", type=int, default=64, help="hidden size of transformer model")
    parser.add_argument("--gen_num_hidden_layers", type=int, default=2, help="number of layers")
    parser.add_argument('--gen_num_attention_heads', default=2, type=int)
    parser.add_argument('--gen_hidden_act', default="gelu", type=str)  # gelu relu
    parser.add_argument("--gen_attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout p")
    parser.add_argument("--gen_hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p")
    parser.add_argument("--gen_initializer_range", type=float, default=0.02)
    parser.add_argument("--gen_loss_weight", type=float, default=1.0)
    parser.add_argument('--gen_loss_type', default="full-softmax", type=str)
    parser.add_argument("--sample_ratio", type=float, default=0.2, \
                        help="weight of contrastive learning task")

    parser.add_argument('--alpha', default=0.5, type=float, help='calculate position importance')

    parser.add_argument('--temperature', default=1.0, type=float, \
                        help="temperature of similarity scores")
    parser.add_argument('--prob_power', default=1.0, type=float, \
                        help="negative sample probability power of idx")
    parser.add_argument('--item_sample_ratio', default=1.0, type=float, help='sampled item ratio')

    # Discriminator args
    parser.add_argument("--dis_hidden_size", type=int, default=64, help="hidden size of transformer model")
    parser.add_argument("--dis_num_hidden_layers", type=int, default=2, help="number of layers")
    parser.add_argument('--dis_num_attention_heads', default=2, type=int)
    parser.add_argument('--dis_hidden_act', default="gelu", type=str)  # gelu relu
    parser.add_argument("--dis_attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout p")
    parser.add_argument("--dis_hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p")
    parser.add_argument("--dis_initializer_range", type=float, default=0.02)
    parser.add_argument("--dis_loss_weight", type=float, default=0.2)
    parser.add_argument('--dis_loss_type', default="bce", type=str)
    parser.add_argument('--dis_opt_versioin', default="full", type=str)
    parser.add_argument('--project_type', default="affine", type=str, \
                        help='sum, affine')

    # train args
    parser.add_argument('--max_seq_length', default=50, type=int)
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=256, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=500, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--model_shared_type", default='full', type=str, \
                        help='embed_only, encoder_only, full')
    parser.add_argument("--cf_weight", type=float, default=0.1, help="weight of contrastive learning task")
    parser.add_argument("--rec_weight", type=float, default=1.0, help="weight of contrastive learning task")
    parser.add_argument("--intent_cf_weight", type=float, default=0.6, help="weight of contrastive learning task")

    # learning related
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")

    # negative sampling strategies
    # parser.add_argument("--neg_numbers", type=int, default=1, help="number of negative items\
    #                      for next item prediction")

    args = parser.parse_args()

    set_seed(args.seed)
    check_path(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    print("Using Cuda:", torch.cuda.is_available())
    args.data_file = args.data_dir + args.data_name + '.txt'

    user_seq, max_item, valid_rating_matrix, test_rating_matrix, num_user, = get_user_seqs(args.data_file)

    item2attribute_file = args.data_dir + args.data_name + '_item2attributes.json'

    item2attribute, attribute_size = get_item2attribute_json(item2attribute_file)


    args.item_size = max_item + 2
    args.mask_id = max_item + 1
    args.attribute_size = attribute_size + 1
    # 增加用户的个数
    args.user_size = num_user

    # save model args
    args_str = f'{args.model_name}-{args.data_name}-{args.model_idx}'
    args.log_file = os.path.join(args.output_dir, args_str + 'yelp.txt')

    show_args_info(args)

    with open(args.log_file, 'a') as f:
        f.write(str(args) + '\n')

    # set item score in train set to `0` in validation
    args.train_matrix = valid_rating_matrix

    # save model
    checkpoint = args_str + '.pt'
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    # training data
    cluster_dataset = IDMARecDataset(args,
                                     user_seq[:int(len(user_seq) * args.training_data_ratio)], \
                                     data_type='train')
    cluster_sampler = SequentialSampler(cluster_dataset)
    cluster_dataloader = DataLoader(cluster_dataset, sampler=cluster_sampler, batch_size=args.batch_size)

    train_dataset = IDMARecDataset(args,
                                   user_seq[:int(len(user_seq) * args.training_data_ratio)], \
                                   data_type='train')
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    eval_dataset = IDMARecDataset(args, user_seq, data_type='valid')
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)

    test_dataset = IDMARecDataset(args, user_seq, data_type='test')
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)


    args = update_to_module_parameters(args, m_type='gen')
    generator_A = SASRecAttributeModel(args=args)
    generator_B = SASRecAttributeModel(args=args)

    args = update_to_module_parameters(args, m_type='dis')
    discriminator_A = SASRecModel(args=args)
    discriminator_B = SASRecModel(args=args)

    if args.model_shared_type == 'embed_only':
        discriminator_A.item_embeddings = generator_A.item_embeddings
        discriminator_A.position_embeddings = generator_A.position_embeddings
        generator_B.item_embeddings = generator_A.item_embeddings
        generator_B.position_embeddings = generator_A.position_embeddings
        discriminator_B.item_embeddings = generator_B.item_embeddings
        discriminator_B.position_embeddings = generator_B.position_embeddings
        discriminator_A.LayerNorm = generator_A.LayerNorm
        discriminator_A.dropout = generator_A.dropout
        discriminator_B.LayerNorm = generator_B.LayerNorm
        discriminator_B.dropout = generator_B.dropout
    elif args.model_shared_type == 'encoder_only':
        discriminator_A.item_encoder = generator_A.item_encoder
        generator_B.item_encoder = generator_A.item_encoder
        discriminator_B.item_encoder = generator_B.item_encoder
    elif args.model_shared_type == 'full':
        discriminator_A.item_embeddings = generator_A.item_embeddings
        discriminator_A.position_embeddings = generator_A.position_embeddings
        generator_B.item_embeddings = discriminator_A.item_embeddings
        generator_B.position_embeddings = discriminator_A.position_embeddings
        discriminator_B.item_embeddings = generator_B.item_embeddings
        discriminator_B.position_embeddings = generator_B.position_embeddings
        discriminator_A.item_encoder = generator_A.item_encoder
        generator_B.item_encoder = discriminator_A.item_encoder
        discriminator_B.item_encoder = generator_B.item_encoder
        discriminator_A.LayerNorm = generator_A.LayerNorm
        discriminator_A.dropout = generator_A.dropout
        discriminator_B.LayerNorm = generator_B.LayerNorm
        discriminator_B.dropout = generator_B.dropout


    else:
        print("isolated models")


    trainer = ELECITY(generator_A, generator_B, discriminator_A, discriminator_B,
                      train_dataloader, cluster_dataloader,
                      eval_dataloader, test_dataloader, args)

    if args.do_eval:
        trainer.args.train_matrix = test_rating_matrix
        trainer.load(args.checkpoint_path)
        print(f'Load model from {args.checkpoint_path} for test!')
        scores, result_info = trainer.test(0, full_sort=True)

    else:
        print(f'Train IDMARec')
        early_stopping = EarlyStopping(args.checkpoint_path, patience=40, verbose=True)
        for epoch in range(args.epochs):
            args.train_matrix = valid_rating_matrix
            trainer.train(epoch)
            # evaluate on NDCG@20

            scores, _ = trainer.valid(epoch, full_sort=True)
            trainer.args.train_matrix = test_rating_matrix

            print("Discriminator_C:")
            scores_test, _ = trainer.test(epoch, full_sort=True)

            early_stopping(np.array(scores[-1:]), trainer.discriminator_A)
            print("test_score:")

            trainer.discriminator_A.load_state_dict(torch.load(args.checkpoint_path))
            scores_test, _ = trainer.test(epoch, full_sort=True)

            if early_stopping.early_stop:
                print("Early stopping")
                break
        trainer.args.train_matrix = test_rating_matrix
        print('---------------Change to test_rating_matrix!-------------------')
        # load the best model
        trainer.discriminator_A.load_state_dict(torch.load(args.checkpoint_path))
        scores, result_info = trainer.test(0, full_sort=True)
main()