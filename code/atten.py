#!/usr/bin/env python

"""
   Factor Graph Attention

   See: https://arxiv.org/abs/1904.05880

   Code by: Idan Schwartz (idanschwartz@gmail.com)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import  product,permutations, combinations_with_replacement, chain


class Unary(nn.Module):
    def __init__(self, embed_size):
        super(Unary, self).__init__()
        self.embed = nn.Conv1d(embed_size, embed_size, 1)
        self.feature_reduce = nn.Conv1d(embed_size, 1, 1)

    def forward(self,  X):
        X = X.transpose(1, 2)

        X_embed = self.embed(X)

        X_nl_embed = F.dropout(F.relu(X_embed))
        X_poten = self.feature_reduce(X_nl_embed)
        return X_poten.squeeze(1)

class Pairwise(nn.Module):
    def __init__(self, embed_x_size, x_spatial_dim=None, embed_y_size=None, y_spatial_dim=None):
        super(Pairwise, self).__init__()
        print(x_spatial_dim, y_spatial_dim)
        embed_y_size = embed_y_size if y_spatial_dim is not None else embed_x_size
        self.y_spatial_dim = y_spatial_dim if y_spatial_dim is not None else x_spatial_dim

        self.embed_size = max(embed_x_size, embed_y_size)
        self.x_spatial_dim = x_spatial_dim

        self.embed_X = nn.Conv1d(embed_x_size, self.embed_size, 1)
        self.embed_Y = nn.Conv1d(embed_y_size, self.embed_size, 1)
        if x_spatial_dim is not None:
            self.normalize_S = nn.BatchNorm1d(self.x_spatial_dim * self.y_spatial_dim)

            self.margin_X = nn.Conv1d(self.y_spatial_dim, 1, 1)
            self.margin_Y = nn.Conv1d(self.x_spatial_dim, 1, 1)

    def forward(self, X, Y=None):

        X_t = X.transpose(1, 2)
        Y_t = Y.transpose(1, 2) if Y is not None else X_t

        X_embed = self.embed_X(X_t)
        Y_embed = self.embed_Y(Y_t)

        X_norm = F.normalize(X_embed)
        Y_norm = F.normalize(Y_embed)

        S = X_norm.transpose(1, 2).bmm(Y_norm)
        if self.x_spatial_dim is not None:
            S = self.normalize_S(S.view(-1, self.x_spatial_dim * self.y_spatial_dim)) \
                .view(-1, self.x_spatial_dim, self.y_spatial_dim)

            X_poten = self.margin_X(S.transpose(1, 2)).transpose(1, 2).squeeze(2)
            Y_poten = self.margin_Y(S).transpose(1, 2).squeeze(2)
        else:
            X_poten = S.mean(dim=2, keepdim=False)
            Y_poten = S.mean(dim=1, keepdim=False)

        if Y is None:
            return X_poten
        else:
            return X_poten, Y_poten


class Atten(nn.Module):
    def __init__(self, util_e, high_order_utils=[], prior_flag=False,
                 sizes=[], size_flag=False, size_force=False, pairwise_flag=True, unary_flag=True, self_flag=True):
        super(Atten, self).__init__()

        self.util_e = util_e

        self.prior_flag = prior_flag

        self.n_utils = len(util_e)

        self.spatial_pool = nn.ModuleDict()

        self.un_models = nn.ModuleList()

        self.self_flag = self_flag
        self.pairwise_flag = pairwise_flag
        self.unary_flag = unary_flag
        self.size_flag = size_flag
        self.size_force = size_force
        if not self.size_flag:
           sizes = [None for _ in util_e]
        self.high_order_utils = high_order_utils
        self.high_order_set = set([h[0] for h in self.high_order_utils])

        for idx, e_dim in enumerate(util_e):
            self.un_models.append(Unary(e_dim))
            if self.size_force:
                self.spatial_pool[str(idx)] = nn.AdaptiveAvgPool1d(sizes[idx])

        self.pp_models = nn.ModuleDict()
        for ((idx1, e_dim_1), (idx2, e_dim_2)) \
                in combinations_with_replacement(enumerate(util_e), 2):
            if idx1 == idx2:
                self.pp_models[str(idx1)] = Pairwise(e_dim_1, sizes[idx1])
            else:
                if pairwise_flag:
                    for i, num_utils, connected_list in self.high_order_utils:
                        if i == idx1 and idx2 not in set(connected_list) \
                                or idx2 == i and idx1 not in set(connected_list):
                            continue
                    self.pp_models[str((idx1, idx2))] = Pairwise(e_dim_1, sizes[idx1], e_dim_2, sizes[idx2])



        self.reduce_potentials = nn.ModuleList()
        self.num_of_potentials = dict()

        self.default_num_of_potentials = 0

        if self.self_flag:
            self.default_num_of_potentials += 1
        if self.unary_flag:
            self.default_num_of_potentials += 1
        if self.prior_flag:
            self.default_num_of_potentials += 1
        for idx in range(self.n_utils):
            self.num_of_potentials[idx] = self.default_num_of_potentials


        '''
        ' All other utils
        '''
        if pairwise_flag:
            for idx, num_utils, connected_utils in high_order_utils:
                for c_u in connected_utils:
                    self.num_of_potentials[c_u] += num_utils
                    self.num_of_potentials[idx] += 1
            for k in self.num_of_potentials.keys():
                if k not in self.high_order_set:
                    self.num_of_potentials[k] += (self.n_utils-1) - len(high_order_utils)




        for idx in range(self.n_utils):
            self.reduce_potentials.append(nn.Conv1d(self.num_of_potentials[idx], 1, 1, bias=False))

    def forward(self, utils, priors=None):
            assert self.n_utils == len(utils)
            assert (priors is None and not self.prior_flag)\
                or (priors is not None
                    and self.prior_flag
                    and len(priors) == self.n_utils)
            b_size = utils[0].size(0)
            util_poten = dict()
            attention = list()
            if self.size_force:
                for i, num_utils, _ in self.high_order_utils:
                    if str(i) not in self.spatial_pool.keys():
                        continue
                    else:
                        high_util = utils[i]
                        high_util = high_util.view(num_utils * b_size, high_util.size(2), high_util.size(3))
                        high_util = high_util.transpose(1, 2)
                        utils[i] = self.spatial_pool[str(i)](high_util).transpose(1, 2)

                for i in range(self.n_utils):
                    if i in self.high_order_set \
                            or str(i) not in self.spatial_pool.keys():
                        continue
                    utils[i] = utils[i].transpose(1, 2)
                    utils[i] = self.spatial_pool[str(i)](utils[i]).transpose(1, 2)
                    if self.prior_flag and priors[i] is not None:
                        priors[i] = self.spatial_pool[str(i)](priors[i].unsqueeze(1)).squeeze(1)



            for i, num_utils, connected_list in self.high_order_utils:
                # i.e. Sharing-Wieghts
                if self.unary_flag:
                    util_poten.setdefault(i, []).append(self.un_models[i](utils[i]))

                if self.self_flag:
                    util_poten.setdefault(i, []).append(self.pp_models[str(i)](utils[i]))

                if self.pairwise_flag:
                    for j in connected_list:
                        other_util = utils[j]
                        expanded_util = other_util.unsqueeze(1).expand(b_size,
                                                             num_utils,
                                                             other_util.size(1),
                                                             other_util.size(2)).contiguous().view(b_size * num_utils,
                                                                        other_util.size(1),
                                                                        other_util.size(2))

                        if i < j:
                            poten_ij, poten_ji = self.pp_models[str((i, j))](utils[i], expanded_util)
                        else:
                            poten_ji, poten_ij = self.pp_models[str((j, i))](expanded_util, utils[i])
                        util_poten[i].append(poten_ij)
                        util_poten.setdefault(j, []).append(poten_ji.view(b_size, num_utils, poten_ji.size(1)))


            #local
            for i in range(self.n_utils):
                if i in self.high_order_set:
                    continue
                if self.unary_flag:
                    util_poten.setdefault(i, []).append(self.un_models[i](utils[i]))
                if self.self_flag:
                    util_poten.setdefault(i, []).append(self.pp_models[str(i)](utils[i]))

            #joint
            if self.pairwise_flag:
                for (i, j) in combinations_with_replacement(range(self.n_utils), 2):
                    if i in self.high_order_set \
                            or j in self.high_order_set:
                        continue
                    if i == j: continue
                    else:
                        poten_ij, poten_ji = self.pp_models[str((i, j))](utils[i], utils[j])
                        util_poten.setdefault(i, []).append(poten_ij)
                        util_poten.setdefault(j, []).append(poten_ji)


            #utils
            for i in range(self.n_utils):
                if self.prior_flag:
                    prior = priors[i] \
                        if priors[i] is not None \
                        else torch.zeros_like(util_poten[i][0], requires_grad=False).cuda()

                    util_poten[i].append(prior)

                util_poten[i] = torch.cat([p if len(p.size()) == 3 else p.unsqueeze(1)
                                           for p in util_poten[i]], dim=1)
                util_poten[i] = self.reduce_potentials[i](util_poten[i]).squeeze(1)
                util_poten[i] = F.softmax(util_poten[i], dim=1).unsqueeze(2)
                attention.append(torch.bmm(utils[i].transpose(1, 2), util_poten[i]).squeeze(2))

            return attention



class NaiveAttention(nn.Module):
    def __init__(self):
        super(NaiveAttention, self).__init__()

    def forward(self, utils, priors):
        atten = []
        spatial_atten = []
        #print([u[1].shape if type(u) is tuple else u.shape for u in utils])
        #print([0 if p is None else p.shape for p in priors])
        for u, p in zip(utils, priors):
            if type(u) is tuple:
                u = u[1]
                num_elements = u.shape[0]
                if p is not None:
                    u = u.view(-1, u.shape[-2], u.shape[-1])
                    p = p.view(-1, p.shape[-2], p.shape[-1])
                    spatial_atten.append(torch.bmm(p.transpose(1, 2), u).squeeze(2).view(num_elements,-1,u.shape[-2],u.shape[-1]))
                else:
                    spatial_atten.append(u.mean(2))
                continue
            if p is not None:
                atten.append(torch.bmm(u.transpose(1, 2), p.unsqueeze(2)).squeeze(2))
            else:
                atten.append(u.mean(1))
        return atten, spatial_atten

