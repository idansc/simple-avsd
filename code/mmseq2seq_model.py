# -*- coding: utf-8 -*-
"""Multimodal sequence-to-sequence model module
   Adapted from 2018 Mitsubishi Electric Research Labs
   Used in: A Simple Baseline for Audio-Visual Scene-Aware Dialog
   https://arxiv.org/abs/1904.05876v1
"""

import sys
import math
import numpy as np
import six
import torch
import torch.nn as nn
import torch.nn.functional as F
from atten import Atten, NaiveAttention
torch.manual_seed(1)


class MMSeq2SeqModel(nn.Module):

    def __init__(self, mm_encoder, history_encoder, input_encoder, response_decoder):
        """ Define model structure
            Args:
                history_encoder (~chainer.Chain): history encoder network
                input_encoder (~chainer.Chain): input encoder network
                response_decoder (~chainer.Chain): response decoder network
        """
        super(MMSeq2SeqModel, self).__init__()
        self.history_encoder = history_encoder
        self.mm_encoder = mm_encoder
        self.input_encoder = input_encoder
        self.response_decoder = response_decoder
        q_embed = 256
        self.s_embed = 256
        self.a_embed = 256

        #used for sharing-weights
        high_order_utils = []

        self.emb_s = nn.Conv1d(512, self.s_embed, 1)
        self.emb_a = nn.Conv1d(128, self.a_embed, 1)
        self.emb_temporal_sp = nn.LSTM(self.s_embed, self.s_embed, dropout=0.5, batch_first=True)

        #self.atten = NaiveAttention()
        self.atten = Atten(util_e=[q_embed, self.s_embed, self.s_embed, self.s_embed, self.s_embed, self.a_embed], high_order_utils=high_order_utils,
                           prior_flag=True, sizes=[10, 49, 49, 49, 49, 10], size_flag=False, pairwise_flag=True, unary_flag=True, self_flag=True)

    def loss(self, mx, hx, x, y, t, s):
        """ Forward propagation and loss calculation
            Args:
                es (pair of ~chainer.Variable): encoder state
                x (list of ~chainer.Variable): list of input sequences
                y (list of ~chainer.Variable): list of output sequences
                t (list of ~chainer.Variable): list of target sequences
                                   if t is None, it returns only states
            Return:
                es (pair of ~chainer.Variable(s)): encoder state
                ds (pair of ~chainer.Variable(s)): decoder state
                loss (~chainer.Variable) : cross-entropy loss
        """
        # encode

        ei, ei_len = self.input_encoder(None, x)

        q_prior = torch.zeros(ei.size(0), ei.size(1)).cuda()
        idx = torch.from_numpy(ei_len - 1).long().cuda()
        batch_index = torch.arange(0, ei_len.shape[0]).long().cuda()
        q_prior[batch_index, idx] = 1
        num_samples = s.shape[0]
        s = s.view(-1, s.size(2), s.size(3)).transpose(1, 2)
        s = self.emb_s(s)


        s = s.view(num_samples, -1, s.size(1), s.size(2)).transpose(2, 3)


        a = mx[0].cuda().permute(1, 2, 0)
        a = self.emb_a(a)
        a = a.transpose(1, 2)

        ei = self.atten(utils=[ei, s[0], s[1], s[2], s[3], a], priors=[q_prior, None, None, None, None, None])

        a_q = ei[0]
        a_s = [ei[1], ei[2], ei[3], ei[4]]
        a_a = ei[5]

        a_a_s = torch.cat([a_a.unsqueeze(1)] + [u.unsqueeze(1) for u in a_s], dim=1)
        _, hidden_temporal_state = self.emb_temporal_sp(a_a_s)
        eh = self.history_encoder(None, hx)

        # concatenate encodings
        es = torch.cat((a_q, eh[-1]), dim=1)
        if hasattr(self.response_decoder, 'context_to_state') \
            and self.response_decoder.context_to_state==True:
            ds, dy = self.response_decoder(es, None, y)
        else:
            # decode
            ds, dy = self.response_decoder(hidden_temporal_state, es, y)

        # compute loss
        if t is not None:
            tt = torch.cat(t, dim=0)
            loss = F.cross_entropy(dy, torch.tensor(tt, dtype=torch.long).cuda())
            #max_index = dy.max(dim=1)[1]
            #hit = (max_index == torch.tensor(tt, dtype=torch.long).cuda()).sum()
            return None, ds, loss
        else:  # if target is None, it only returns states
            return None, ds

    def generate(self, mx, hx, x, s, sos=2, eos=2, unk=0, minlen=1, maxlen=100, beam=5, penalty=1.0, nbest=1):
        """ Generate sequence using beam search
            Args:
                es (pair of ~chainer.Variable(s)): encoder state
                x (list of ~chainer.Variable): list of input sequences
                sos (int): id number of start-of-sentence label
                eos (int): id number of end-of-sentence label
                unk (int): id number of unknown-word label
                maxlen (int): list of target sequences
                beam (int): list of target sequences
                penalty (float): penalty added to log probabilities
                                 of each output label.
                nbest (int): number of n-best hypotheses to be output
            Return:
                list of tuples (hyp, score): n-best hypothesis list
                 - hyp (list): generated word Id sequence
                 - score (float): hypothesis score
                pair of ~chainer.Variable(s)): decoder state of best hypothesis
        """
        # encode
        ei, ei_len = self.input_encoder(None, x)

        q_prior = torch.zeros(ei.size(0), ei.size(1)).cuda()
        idx = torch.from_numpy(ei_len - 1).long().cuda()
        batch_index = torch.arange(0, ei_len.shape[0]).long().cuda()
        q_prior[batch_index, idx] = 1
        num_samples = s.shape[0]
        s = s.view(-1, s.size(2), s.size(3)).transpose(1, 2)
        s = self.emb_s(s)

        s = s.view(num_samples, -1, s.size(1), s.size(2)).transpose(2, 3)

        a = mx[0].cuda().permute(1, 2, 0)
        a = self.emb_a(a)
        a = a.transpose(1, 2)

        ei = self.atten(utils=[ei, s[0], s[1], s[2], s[3], a], priors=[q_prior, None, None, None, None, None])

        a_q = ei[0]
        a_s = [ei[1], ei[2], ei[3], ei[4]]
        a_a = ei[5]

        a_a_s = torch.cat([a_a.unsqueeze(1)] + [u.unsqueeze(1) for u in a_s], dim=1)
        _, hidden_temporal_state = self.emb_temporal_sp(a_a_s)
        eh = self.history_encoder(None, hx)

        # concatenate encodings
        es = torch.cat((a_q, eh[-1]), dim=1)

        # beam search
        ds = self.response_decoder.initialize(hidden_temporal_state, es, torch.from_numpy(np.asarray([sos])).cuda())
        hyplist = [([], 0., ds)]
        best_state = None
        comp_hyplist = []
        for l in six.moves.range(maxlen):
            new_hyplist = []
            argmin = 0
            for out, lp, st in hyplist:
                logp = self.response_decoder.predict(st)
                lp_vec = logp.cpu().data.numpy() + lp
                lp_vec = np.squeeze(lp_vec)
                if l >= minlen:
                    new_lp = lp_vec[eos] + penalty * (len(out) + 1)
                    new_st = self.response_decoder.update(st, torch.from_numpy(np.asarray([eos])).cuda())
                    comp_hyplist.append((out, new_lp))
                    if best_state is None or best_state[0] < new_lp:
                        best_state = (new_lp, new_st)

                for o in np.argsort(lp_vec)[::-1]:
                    if o == unk or o == eos:  # exclude <unk> and <eos>
                        continue
                    new_lp = lp_vec[o]
                    if len(new_hyplist) == beam:
                        if new_hyplist[argmin][1] < new_lp:
                            new_st = self.response_decoder.update(st, torch.from_numpy(np.asarray([o])).cuda())
                            new_hyplist[argmin] = (out + [o], new_lp, new_st)
                            argmin = min(enumerate(new_hyplist), key=lambda h: h[1][1])[0]
                        else:
                            break
                    else:
                        new_st = self.response_decoder.update(st, torch.from_numpy(np.asarray([o])).cuda())
                        new_hyplist.append((out + [o], new_lp, new_st))
                        if len(new_hyplist) == beam:
                            argmin = min(enumerate(new_hyplist), key=lambda h: h[1][1])[0]

            hyplist = new_hyplist

        if len(comp_hyplist) > 0:
            maxhyps = sorted(comp_hyplist, key=lambda h: -h[1])[:nbest]
            return maxhyps, best_state[1]
        else:
            return [([], 0)], None
