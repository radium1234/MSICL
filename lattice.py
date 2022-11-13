# -*- coding: utf-8 -*-
r"""
NCL
################################################
Reference:
    Zihan Lin*, Changxin Tian*, Yupeng Hou*, Wayne Xin Zhao. "Improving Graph Collaborative Filtering with Neighborhood-enriched Contrastive Learning." in WWW 2022.
"""
import argparse
import os.path

import numpy as np
import json
import array
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType

from utility.parser import parse_args
from utility.norm import build_sim, build_knn_normalized_graph


class Lattice(GeneralRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(Lattice, self).__init__(config, dataset)
        self.dataset = config["dataset"]
        self.topk = config["k"]
        self.sparse = config["sparse"]
        self.norm_type = config["norm_type"]
        self.reg_weight = config["reg_weight"]
        self.layers = config["layers"]
        self.lambda_coeff = config["lambda_coeff"]
        self.weight_size = [64, 64]
        self.embedding_dim = 64
        self.embed_size = config["embed_size"]

        if not os.path.exists('dataset/' + self.dataset + '/image_feat.npy'):
            self.image_feat()
        image_feats = np.load('dataset/' + self.dataset + '/image_feat.npy').astype('float32')
        l = image_feats.shape[0]
        if self.n_items != l + 1:
            self.image_feat()
            image_feats = np.load('dataset/' + self.dataset + '/image_feat.npy').astype('float32')
            l = image_feats.shape[0]
        assert self.n_items == l + 1
        avg = image_feats.mean(0)
        image_feats = np.vstack([avg, image_feats])
        if not os.path.exists('dataset/' + self.dataset + '/text_feat.npy'):
            self.text_feat()
        text_feats = np.load('dataset/' + self.dataset + '/text_feat.npy').astype('float32')
        l = text_feats.shape[0]
        if self.n_items != l + 1:
            self.text_feat()
            text_feats = np.load('dataset/' + self.dataset + '/text_feat.npy').astype('float32')
            l = text_feats.shape[0]
        assert self.n_items == l + 1

        avg = text_feats.mean(0)
        text_feats = np.vstack([avg, text_feats])

        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        self.image_embedding = nn.Embedding.from_pretrained(torch.Tensor(image_feats), freeze=False)
        self.text_embedding = nn.Embedding.from_pretrained(torch.Tensor(text_feats), freeze=False)

        self.n_ui_layers = len(self.weight_size)
        self.weight_size = [self.embedding_dim] + self.weight_size
        image_adj = build_sim(self.image_embedding.weight.detach())
        image_adj = build_knn_normalized_graph(image_adj, topk=self.topk, is_sparse=self.sparse,
                                               norm_type=self.norm_type)

        text_adj = build_sim(self.text_embedding.weight.detach())
        text_adj = build_knn_normalized_graph(text_adj, topk=self.topk, is_sparse=self.sparse, norm_type=self.norm_type)

        self.text_original_adj = text_adj.cuda()
        self.image_original_adj = image_adj.cuda()

        self.image_trs = nn.Linear(image_feats.shape[1], self.embed_size)
        self.text_trs = nn.Linear(text_feats.shape[1], self.embed_size)

        self.adj = self.get_norm_adj_mat().to(self.device)

        self.softmax = nn.Softmax(dim=-1)

        self.query = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, 1, bias=False)
        )
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        self.tau = 0.5

        # load dataset info

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.embedding_dim)
        self.item_id_embedding = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.embedding_dim)

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']
        self.loss_ratio = config["loss_ratio"]
        # parameters initialization
        self.apply(xavier_uniform_initialization)

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        self.diag = torch.from_numpy(diag).to(self.device)
        D = sp.diags(diag)
        L = D @ A @ D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def mm(self, x, y):
        if self.sparse:
            return torch.sparse.mm(x, y)
        else:
            return torch.mm(x, y)

    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    # def batched_contrastive_loss(self, z1, z2, batch_size=4096):
    #     device = z1.device
    #     num_nodes = z1.size(0)
    #     num_batches = (num_nodes - 1) // batch_size + 1
    #     f = lambda x: torch.exp(x / self.tau)
    #     indices = torch.arange(0, num_nodes).to(device)
    #     losses = []
    #
    #     for i in range(num_batches):
    #         mask = indices[i * batch_size:(i + 1) * batch_size]
    #         refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
    #         between_sim = f(self.sim(z1[mask], z2))  # [B, N]
    #
    #         losses.append(-torch.log(
    #             between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
    #             / (refl_sim.sum(1) + between_sim.sum(1)
    #                - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))
    #
    #     loss_vec = torch.cat(losses)
    #     return loss_vec.mean()

    def forward(self):
        image_feats = self.image_trs(self.image_embedding.weight)
        text_feats = self.text_trs(self.text_embedding.weight)
        self.image_adj = build_sim(image_feats)
        self.image_adj = build_knn_normalized_graph(self.image_adj, topk=self.topk, is_sparse=self.sparse,
                                                    norm_type=self.norm_type)
        self.image_adj = (1 - self.lambda_coeff) * self.image_adj + self.lambda_coeff * self.image_original_adj

        self.text_adj = build_sim(text_feats)
        self.text_adj = build_knn_normalized_graph(self.text_adj, topk=self.topk, is_sparse=self.sparse,
                                                   norm_type=self.norm_type)
        self.text_adj = (1 - self.lambda_coeff) * self.text_adj + self.lambda_coeff * self.text_original_adj

        image_item_embeds = self.item_id_embedding.weight
        text_item_embeds = self.item_id_embedding.weight

        for i in range(self.layers):
            image_item_embeds = self.mm(self.image_adj, image_item_embeds)
        for i in range(self.layers):
            text_item_embeds = self.mm(self.text_adj, text_item_embeds)
        att = torch.cat([self.query(image_item_embeds), self.query(text_item_embeds)], dim=-1)
        weight = self.softmax(att)
        fusion_item_embeds = weight[:, 0].unsqueeze(dim=1) * image_item_embeds + weight[:, 1].unsqueeze(
            dim=1) * text_item_embeds
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(self.adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        i_g_embeddings = i_g_embeddings + F.normalize(fusion_item_embeds, p=2, dim=1)
        return u_g_embeddings, i_g_embeddings, image_item_embeds, text_item_embeds, fusion_item_embeds

    def text_feat(self):
        model = SentenceTransformer('paraphrase-albert-small-v2')

        with open('dataset' + '/' + self.dataset + '/' + 'id-newid.json', 'r') as file:
            content = file.read()
        dic1 = json.loads(content)  # 将json格式文件转化为python的字典文件
        dic2 = {}
        with open('dataset' + '/' + self.dataset + '/' + self.dataset + '.item', 'r') as f:
            k = 0
            while True:
                k = k + 1
                if (k % 10000 == 0): print(k)
                a = f.readline()
                if a == '': break
                a = a.split('\t')
                id = a[0]
                if (id in dic1.keys()):
                    dic2[dic1[id]] = a[1]
        lst = []
        for i in range(1, len(dic2) + 1):
            k = model.encode(dic2[i])
            lst.append(k)
            if (i % 100 == 0): print(i)
        lst = np.array(lst)
        np.save('dataset' + '/' + self.dataset + '/' + 'text_feat.npy', lst)

    def image_feat(self):
        with open('dataset' + '/' + self.dataset + '/' + 'id-newid.json', 'r') as file:
            content = file.read()
        b1 = json.loads(content)  # 将json格式文件转化为python的字典文件
        with open('dataset' + '/' + self.dataset + '/' + 'newid-id.json', 'r') as file:
            content = file.read()
        b2 = json.loads(content)  # 将json格式文件转化为python的字典文件
        path = 'dataset' +'/' + self.dataset + '.b'
        f = open(path, 'rb')
        k = 0
        d = dict()
        while True:
            if k % 10000 == 0:
                print(k)
            k = k + 1
            asin = f.read(10).decode('UTF-8')
            if asin == '': break
            a = array.array('f')
            a.fromfile(f, 4096)
            if asin in b1.keys():
                d[asin] = a.tolist()
        lst = []
        for i in range(1, len(b1) + 1):
            lst.append(d[b2[str(i)]])
        k = np.array(lst)
        np.save('dataset' + '/' + self.dataset + '/' + 'image_feat.npy', k)

    # def bpr_loss(self, users, pos_items, neg_items):
    #     self.regs = eval(self.regs)
    #     self.decay = self.regs[0]
    #     self.batch_size = self.batch_size
    #     pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
    #     neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)
    #
    #     regularizer = 1. / 2 * (users ** 2).sum() + 1. / 2 * (pos_items ** 2).sum() + 1. / 2 * (neg_items ** 2).sum()
    #     regularizer = regularizer / self.batch_size
    #
    #     maxi = F.logsigmoid(pos_scores - neg_scores)
    #     mf_loss = -torch.mean(maxi)
    #
    #     emb_loss = self.decay * regularizer
    #     reg_loss = 0.0
    #     return mf_loss, emb_loss, reg_loss

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None
        # batch里4096个交互数据,id 从1开始
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]
        # 全员的embedding
        ua_embeddings, ia_embeddings, image_item_embeds, text_item_embeds, fusion_item_embed = self.forward()

        u_embeddings = ua_embeddings[user]
        pos_embeddings = ia_embeddings[pos_item]
        neg_embeddings = ia_embeddings[neg_item]

        # image_loss = self.loss_ratio * self.batched_contrastive_loss(image_item_embeds, fusion_item_embed)
        # text_loss = self.loss_ratio * self.batched_contrastive_loss(image_item_embeds, fusion_item_embed)
        # # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)

        mf_loss = self.mf_loss(pos_scores, neg_scores)

        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_id_embedding(pos_item)
        neg_ego_embeddings = self.item_id_embedding(neg_item)

        reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings)

        return mf_loss + self.reg_weight * reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e, _, _, _ = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)
