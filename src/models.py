# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import  math
import os
import pickle
from tqdm import tqdm
import random
import copy
from collections import defaultdict
import numpy as np
from operator import itemgetter


import torch
import torch.nn as nn
import gensim
import faiss
import time

from modules import  LayerNorm, Intermediate, Encoder
from mudule_wes import DistSAEncoder



class KMeans(object):
    def __init__(self, num_cluster, seed, hidden_size, gpu_id=0, device="cpu"):
        """
        Args:
            k: number of clusters
        """
        self.seed = seed
        self.num_cluster = num_cluster
        self.max_points_per_centroid = 4096
        self.min_points_per_centroid = 0
        self.gpu_id = 0
        self.device = device
        self.first_batch = True
        self.hidden_size = hidden_size
        self.clus, self.index = self.__init_cluster(self.hidden_size)
        self.centroids = []

    def __init_cluster(
        self, hidden_size, verbose=False, niter=20, nredo=5, max_points_per_centroid=4096, min_points_per_centroid=0
    ):
        print(" cluster train iterations:", niter)
        clus = faiss.Clustering(hidden_size, self.num_cluster)
        clus.verbose = verbose
        # 迭代次数
        clus.niter = niter
        # 聚类的次数
        clus.nredo = nredo
        clus.seed = self.seed
        # 每一簇的最大样本和最小样本数
        clus.max_points_per_centroid = max_points_per_centroid
        clus.min_points_per_centroid = min_points_per_centroid

        res = faiss.StandardGpuResources()
        res.noTempMemory()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = self.gpu_id
        index = faiss.GpuIndexFlatL2(res, hidden_size, cfg)
        return clus, index

    def train(self, x):
        # train to get centroids
        if x.shape[0] > self.num_cluster:
            self.clus.train(x, self.index)
        # get cluster centroids
        centroids = faiss.vector_to_array(self.clus.centroids).reshape(self.num_cluster, self.hidden_size)
        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).to(self.device)
        self.centroids = nn.functional.normalize(centroids, p=2, dim=1)

    def query(self, x):
        # self.index.add(x)
        D, I = self.index.search(x, 1)  # for each sample, find cluster distance and assignments
        seq2cluster = [int(n[0]) for n in I]
        # print("cluster number:", self.num_cluster,"cluster in batch:", len(set(seq2cluster)))
        seq2cluster = torch.LongTensor(seq2cluster).to(self.device)
        return seq2cluster, self.centroids[seq2cluster]


class KMeans_Pytorch(object):
    def __init__(self, num_cluster, seed, hidden_size, gpu_id=0, device="cpu"):
        """
        Args:
            k: number of clusters
        """
        self.seed = seed
        self.num_cluster = num_cluster
        self.max_points_per_centroid = 4096
        self.min_points_per_centroid = 10
        self.first_batch = True
        self.hidden_size = hidden_size
        self.gpu_id = gpu_id
        self.device = device
        print(self.device, "-----")

    def run_kmeans(self, x, Niter=20, tqdm_flag=False):
        if x.shape[0] >= self.num_cluster:
            seq2cluster, centroids = kmeans(
                X=x, num_clusters=self.num_cluster, distance="euclidean", device=self.device, tqdm_flag=False
            )
            seq2cluster = seq2cluster.to(self.device)
            centroids = centroids.to(self.device)
        # last batch where
        else:
            seq2cluster, centroids = kmeans(
                X=x, num_clusters=x.shape[0] - 1, distance="euclidean", device=self.device, tqdm_flag=False
            )
            seq2cluster = seq2cluster.to(self.device)
            centroids = centroids.to(self.device)
        return seq2cluster, centroids

# class Light_GCN(nn.Module):
#     def __init__(self, args):
#         super(Light_GCN, self).__init__()
#         self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
#         # 引入user的原始嵌入作为一个提示
#         # self.user_embeddings = nn.Embedding(args.user_size, args.hidden_size, padding_idx=0)
#         # 引入项目属性
#         self.attribute_embeddings = nn.Embedding(args.attribute_size, args.hidden_size, padding_idx=0)
#
#         self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
#         self.item_encoder = Encoder(args)
#         self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
#         self.dropout = nn.Dropout(args.hidden_dropout_prob)
#
#         self.args = args
#         self.dis_projection = nn.Linear(self.args.hidden_size, 1)
#
#         att_encoder = nn.TransformerEncoderLayer(args.hidden_size, args.num_attention_heads, args.hidden_size,
#                                                  args.attention_probs_dropout_prob)
#         self.g_enc = nn.TransformerEncoder(att_encoder, args.num_hidden_layers)
#
#         self.criterion = nn.BCELoss(reduction="none")
#         self.apply(self.init_weights)
#     def add_position_embedding(self, sequence):
#
#         seq_length = sequence.size(1)
#         position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
#         position_ids = position_ids.unsqueeze(0).expand_as(sequence)
#         item_embeddings = self.item_embeddings(sequence)
#         # user_embeddings = self.user_embeddings(sequence)
#         att_embeddings = self.attribute_embeddings(position_ids)
#         position_embeddings = self.position_embeddings(position_ids)
#         sequence_emb = item_embeddings + position_embeddings + att_embeddings
#         sequence_emb = self.LayerNorm(sequence_emb)
#         sequence_emb = self.dropout(sequence_emb)
#
#         return sequence_emb
#
#     def gcn_embedding(self, sequence):
#         item_embeddings = self.item_embeddings(sequence)
#         for layer in range(self.args.num_hidden_layers):
#
#
#
#
#     # model same as SASRec
#     def forward(self, input_ids):
#         # 单向
#         attention_mask = (input_ids > 0).long()
#         extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
#         max_len = attention_mask.size(-1)
#         attn_shape = (1, max_len, max_len)
#         subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
#         subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
#         subsequent_mask = subsequent_mask.long()
#
#         if self.args.cuda_condition:
#             subsequent_mask = subsequent_mask.cuda()
#
#         extended_attention_mask = extended_attention_mask * subsequent_mask
#         extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
#         extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
#
#
#
#         sequence_emb = self.add_position_embedding(input_ids)
#
#         # sequence_emb = torch.concat(sequence_emb,intent)
#
#         item_encoded_layers = self.item_encoder(sequence_emb, extended_attention_mask, output_all_encoded_layers=True)
#
#         sequence_output = item_encoded_layers[-1]
#         return sequence_output
#
#     def init_weights(self, module):
#         """ Initialize the weights.
#         """
#         if isinstance(module, (nn.Linear, nn.Embedding)):
#             # Slightly different from the TF version which uses truncated_normal for initialization
#             # cf https://github.com/pytorch/pytorch/pull/5617
#             module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
#         elif isinstance(module, LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)
#         if isinstance(module, nn.Linear) and module.bias is not None:
#             module.bias.data.zero_()


#
class SASRecUserItemPromptModel(nn.Module):
    def __init__(self, args):
        super(SASRecUserItemPromptModel, self).__init__()
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        # 引入user的原始嵌入作为一个提示
        self.user_embeddings = nn.Embedding(args.user_size, args.hidden_size, padding_idx=0)
        # 引入项目属性
        self.attribute_embeddings = nn.Embedding(args.attribute_size, args.hidden_size, padding_idx=0)

        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.item_encoder = Encoder(args)
        DistSAEncoder
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args
        self.dis_projection = nn.Linear(self.args.hidden_size, 1)

        self.criterion = nn.BCELoss(reduction="none")
        self.apply(self.init_weights)

    # Positional Embedding
    def add_position_embedding(self, sequence):

        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence)
        user_embeddings = self.user_embeddings(sequence)
        att_embeddings = self.attribute_embeddings(position_ids)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings + user_embeddings + att_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb
    #
    # # 融合item和属性的嵌入
    # def fusing_embedding(self, item_embeddings, att_embeddings, att_att_mask):
    #
    #     # ID self-attention
    #     # append the item embedding to the start of the attribute embeddings
    #     fused_embedding = self.g_enc(
    #         torch.cat([torch.unsqueeze(item_embeddings, dim=1), att_embeddings], dim=1).transpose(0, 1),
    #         src_key_padding_mask=att_att_mask.bool())[0]  # N_(all items in the batch) * hidden_sz
    #
    #     return fused_embedding


    # model same as SASRec
    def forward(self, input_ids):
        # 单向
        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # # 双向
        # attention_mask = (input_ids > 0).long()
        # extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # # bidirectional mask
        # extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        # extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0


        sequence_emb = self.add_position_embedding(input_ids)

        # sequence_emb = torch.concat(sequence_emb,intent)

        item_encoded_layers = self.item_encoder(sequence_emb, extended_attention_mask, output_all_encoded_layers=True)

        sequence_output = item_encoded_layers[-1]
        return sequence_output

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


# class SASRecUserItemPromptModel(nn.Module):
#     def __init__(self, args):
#         super(SASRecUserItemPromptModel, self).__init__()
#         self.item_mean_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
#         self.item_cov_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
#         self.position_mean_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
#         self.position_cov_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
#         self.user_mean_embeddings = nn.Embedding(args.num_users, args.hidden_size)
#         self.user_cov_embeddings = nn.Embedding(args.num_users, args.hidden_size)
#         self.att_mean_embeddings = nn.Embedding(args.attribute_size, args.hidden_size)
#         self.att_cov_embeddings = nn.Embedding(args.attribute_size, args.hidden_size)
#
#         self.item_encoder = DistSAEncoder(args)
#         self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
#         self.dropout = nn.Dropout(args.hidden_dropout_prob)
#         self.args = args
#         self.dis_projection = nn.Linear(self.args.hidden_size, 1)
#
#
#         self.apply(self.init_weights)
#         # self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
#         # # 引入user的原始嵌入作为一个提示
#         # self.user_embeddings = nn.Embedding(args.user_size, args.hidden_size, padding_idx=0)
#         # # 引入项目属性
#         # self.attribute_embeddings = nn.Embedding(args.attribute_size, args.hidden_size, padding_idx=0)
#         #
#         # self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
#         # self.item_encoder = DistSAEncoder(args)
#         # self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
#         # self.dropout = nn.Dropout(args.hidden_dropout_prob)
#         # self.args = args
#         # self.dis_projection = nn.Linear(self.args.hidden_size, 1)
#
#         self.criterion = nn.BCELoss(reduction="none")
#         # self.apply(self.init_weights)
#
#     # Positional Embedding
#     # def add_position_embedding(self, sequence):
#     #
#     #     seq_length = sequence.size(1)
#     #     position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
#     #     position_ids = position_ids.unsqueeze(0).expand_as(sequence)
#     #     item_embeddings = self.item_embeddings(sequence)
#     #     user_embeddings = self.user_embeddings(sequence)
#     #     att_embeddings = self.attribute_embeddings(position_ids)
#     #     position_embeddings = self.position_embeddings(position_ids)
#     #     sequence_emb = item_embeddings + position_embeddings + user_embeddings + att_embeddings
#     #     sequence_emb = self.LayerNorm(sequence_emb)
#     #     sequence_emb = self.dropout(sequence_emb)
#     #
#     #     return sequence_emb
#
#     # def add_position_mean_embedding(self, sequence):
#     #     seq_length = sequence.size(1)
#     #     position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
#     #     position_ids = position_ids.unsqueeze(0).expand_as(sequence)
#     #     item_embeddings = self.item_mean_embeddings(sequence)
#     #     position_embeddings = self.position_mean_embeddings(position_ids)
#     #     user_embeddings = self.user_mean_embeddings(sequence)
#     #     att_embeddings = self.att_mean_embeddings(position_ids)
#     #     sequence_emb = item_embeddings + position_embeddings + user_embeddings + att_embeddings
#     #     sequence_emb = self.LayerNorm(sequence_emb)
#     #     sequence_emb = self.dropout(sequence_emb)
#     #     elu_act = torch.nn.ELU()
#     #     sequence_emb = elu_act(sequence_emb)
#     #
#     #     return sequence_emb
#
#     # def add_position_cov_embedding(self, sequence):
#     #
#     #     seq_length = sequence.size(1)
#     #     position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
#     #     position_ids = position_ids.unsqueeze(0).expand_as(sequence)
#     #     item_embeddings = self.item_cov_embeddings(sequence)
#     #     position_embeddings = self.position_cov_embeddings(position_ids)
#     #     user_embeddings = self.user_cov_embeddings(sequence)
#     #     att_embeddings = self.att_cov_embeddings(position_ids)
#     #     sequence_emb = item_embeddings + position_embeddings + user_embeddings + att_embeddings
#     #     sequence_emb = self.LayerNorm(sequence_emb)
#     #     elu_act = torch.nn.ELU()
#     #     sequence_emb = elu_act(self.dropout(sequence_emb)) + 1
#     #
#     #     return sequence_emb
#     #
#     # # 融合item和属性的嵌入
#     # def fusing_embedding(self, item_embeddings, att_embeddings, att_att_mask):
#     #
#     #     # ID self-attention
#     #     # append the item embedding to the start of the attribute embeddings
#     #     fused_embedding = self.g_enc(
#     #         torch.cat([torch.unsqueeze(item_embeddings, dim=1), att_embeddings], dim=1).transpose(0, 1),
#     #         src_key_padding_mask=att_att_mask.bool())[0]  # N_(all items in the batch) * hidden_sz
#     #
#     #     return fused_embedding
#
#
#     # model same as SASRec
#     def forward(self, input_ids):
#         # 单向
#         attention_mask = (input_ids > 0).long()
#         extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
#         max_len = attention_mask.size(-1)
#         attn_shape = (1, max_len, max_len)
#         subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
#         subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
#         subsequent_mask = subsequent_mask.long()
#
#         if self.args.cuda_condition:
#             subsequent_mask = subsequent_mask.cuda()
#
#         extended_attention_mask = extended_attention_mask * subsequent_mask
#         extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
#         extended_attention_mask = (1.0 - extended_attention_mask) * (-2 ** 32 + 1)
#
#         mean_sequence_emb = self.add_position_mean_embedding(input_ids)
#         cov_sequence_emb = self.add_position_cov_embedding(input_ids)
#
#         item_encoded_layers = self.item_encoder(mean_sequence_emb,
#                                                 cov_sequence_emb,
#                                                 extended_attention_mask,
#                                                 output_all_encoded_layers=True)
#
#         mean_sequence_output, cov_sequence_output, att_scores = item_encoded_layers[-1]
#         return mean_sequence_output, cov_sequence_output, att_scores


def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class SASRecAttributeModel(nn.Module):
    def __init__(self, args):
        super(SASRecAttributeModel, self).__init__()
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        # 引入user的原始嵌入作为一个提示
        # self.user_embeddings = nn.Embedding(args.user_size, args.hidden_size, padding_idx=0)
        # 引入项目属性
        self.attribute_embeddings = nn.Embedding(args.attribute_size, args.hidden_size, padding_idx=0)

        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.item_encoder = Encoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        self.args = args
        self.dis_projection = nn.Linear(self.args.hidden_size, 1)


        att_encoder = nn.TransformerEncoderLayer(args.hidden_size, args.num_attention_heads, args.hidden_size,
                                                 args.attention_probs_dropout_prob)
        self.g_enc = nn.TransformerEncoder(att_encoder, args.num_hidden_layers)

        self.criterion = nn.BCELoss(reduction="none")
        self.apply(self.init_weights)

    # Positional Embedding
    def add_position_embedding(self, sequence):

        seq_length = sequence.size(1)

        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence)
        # user_embeddings = self.user_embeddings(sequence)
        att_embeddings = self.attribute_embeddings(position_ids)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings + att_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    # model same as SASRec
    def forward(self, input_ids):
        # 单向
        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0



        sequence_emb = self.add_position_embedding(input_ids)

        # sequence_emb = torch.concat(sequence_emb,intent)

        item_encoded_layers = self.item_encoder(sequence_emb, extended_attention_mask, output_all_encoded_layers=True)

        sequence_output = item_encoded_layers[-1]
        return sequence_output

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


# class SASRecAttributeModel(nn.Module):
#     def __init__(self, args):
#         super(SASRecAttributeModel, self).__init__()
#         self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
#         # 引入项目属性 作为prompt
#         self.attribute_embeddings = nn.Embedding(args.attribute_size, args.hidden_size, padding_idx=0)
#         self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size * 2)
#
#         self.item_encoder = Encoder(args)
#         self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
#         self.dropout = nn.Dropout(args.hidden_dropout_prob)
#
#         self.args = args
#         self.dis_projection = nn.Linear(self.args.hidden_size, 1)
#
#         # # attention for fusing item and attribute
#         # att_encoder = nn.TransformerEncoderLayer(args.hidden_size, args.num_attention_heads, args.hidden_size,
#         #                                          args.attention_probs_dropout_prob)
#         #
#         # self.g_enc = nn.TransformerEncoder(att_encoder, args.num_hidden_layers)
#
#         self.criterion = nn.BCELoss(reduction="none")
#         self.apply(self.init_weights)
#
#     # Positional Embedding
#     def add_position_embedding(self, sequence):
#
#         seq_length = sequence.size(1)
#
#         position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
#         position_ids = position_ids.unsqueeze(0).expand_as(sequence)
#         item_embeddings = self.item_embeddings(sequence)
#         # user_embeddings = self.user_embeddings(sequence)
#         att_embeddings = self.attribute_embeddings(position_ids)
#         position_embeddings = self.position_embeddings(position_ids)
#         sequence_emb = item_embeddings + position_embeddings + att_embeddings
#         sequence_emb = self.LayerNorm(sequence_emb)
#         sequence_emb = self.dropout(sequence_emb)
#
#         return sequence_emb
#
#     # model same as SASRec
#     def forward(self, inp):
#         user_id, answer, input_ids, seq_len, attributes, att_att_mask = batch
#         # 单向
#         attention_mask = (input_ids > 0).long()
#         extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
#         max_len = attention_mask.size(-1)
#         attn_shape = (1, max_len, max_len)
#         subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
#         subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
#         subsequent_mask = subsequent_mask.long()
#
#         if self.args.cuda_condition:
#             subsequent_mask = subsequent_mask.cuda()
#
#         extended_attention_mask = extended_attention_mask * subsequent_mask
#         extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
#         extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
#
#
#
#         sequence_emb = self.add_position_embedding(input_ids)
#
#         # sequence_emb = torch.concat(sequence_emb,intent)
#
#         item_encoded_layers = self.item_encoder(sequence_emb, extended_attention_mask, output_all_encoded_layers=True)
#
#         sequence_output = item_encoded_layers[-1]
#         return sequence_output
#
#     def init_weights(self, module):
#         """ Initialize the weights.
#         """
#         if isinstance(module, (nn.Linear, nn.Embedding)):
#             # Slightly different from the TF version which uses truncated_normal for initialization
#             # cf https://github.com/pytorch/pytorch/pull/5617
#             module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
#         elif isinstance(module, LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)
#         if isinstance(module, nn.Linear) and module.bias is not None:
#             module.bias.data.zero_()

class SASRecUserPropmtModel(nn.Module):
    def __init__(self, args):
        super(SASRecUserPropmtModel, self).__init__()
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        # 引入user的原始嵌入作为一个提示
        self.user_embeddings = nn.Embedding(args.user_size, args.hidden_size, padding_idx=0)
        # 引入项目属性
        # self.attribute_embeddings = nn.Embedding(args.attribute_size, args.hidden_size, padding_idx=0)

        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.item_encoder = Encoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args
        self.dis_projection = nn.Linear(self.args.hidden_size, 1)

        self.criterion = nn.BCELoss(reduction="none")
        self.apply(self.init_weights)

    # Positional Embedding
    def add_position_embedding(self, sequence):

        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence)
        user_embeddings = self.user_embeddings(sequence)
        # att_embeddings = self.attribute_embeddings(position_ids)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings + user_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    # model same as SASRec
    def forward(self, input_ids):
        # 单向
        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0



        sequence_emb = self.add_position_embedding(input_ids)

        # sequence_emb = torch.concat(sequence_emb,intent)

        item_encoded_layers = self.item_encoder(sequence_emb, extended_attention_mask, output_all_encoded_layers=True)

        sequence_output = item_encoded_layers[-1]
        return sequence_output

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()




class SASRecModel(nn.Module):
    def __init__(self, args):
        super(SASRecModel, self).__init__()
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)

        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.item_encoder = Encoder(args)


        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        self.intermediate = Intermediate(args)
        self.args = args
        
        # projection on discriminator output
        self.dis_projection = nn.Linear(self.args.hidden_size, 1)

        self.criterion = nn.BCELoss(reduction='none')
        self.apply(self.init_weights)


    # Positional Embedding
    def add_position_embedding(self, sequence):

        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    # model same as SASRec
    def forward(self, input_ids):

        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1) # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        sequence_emb = self.add_position_embedding(input_ids)



        item_encoded_layers = self.item_encoder(sequence_emb,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True)


        if len(item_encoded_layers) == 0:
            sequence_output = self.intermediate(sequence_emb)
        else:
            sequence_output = item_encoded_layers[-1]
        return sequence_output

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()



if __name__ == '__main__':
    onlineitemsim = OnlineItemSimilarity(item_size=10)
    item_embeddings = nn.Embedding(10, 6, padding_idx=0)
    onlineitemsim.update_embedding_matrix(item_embeddings)
    item_idx = torch.tensor(2, dtype=torch.long)
    similiar_items = onlineitemsim.most_similar(item_idx=item_idx, top_k=1)
    print(similiar_items)