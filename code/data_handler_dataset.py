#!/usr/bin/env python
"""Functions for feature data handling
   Adapted from 2018 Mitsubishi Electric Research Labs
   Used in: A Simple Baseline for Audio-Visual Scene-Aware Dialog
   https://arxiv.org/abs/1904.05876v1
"""

import copy
import logging
import sys
import time
import os
import six
import pickle
import json
import numpy as np
import torch.utils.data as data

def words2ids(str_in, vocab, unk=0, eos=-1):
    words = str_in.split()
    if eos >= 0:
        sentence = np.ndarray(len(words)+1, dtype=np.int32)
    else:
        sentence = np.ndarray(len(words), dtype=np.int32)
    for i,w in enumerate(words):
        if w in vocab:
            sentence[i] = vocab[w]
        else:
            sentence[i] = unk
    if eos >= 0:
	sentence[len(words)] = eos
    return sentence

def get_npy_shape(filename):
    # read npy file header and return its shape
    with open(filename, 'rb') as f:
        if filename.endswith('.pkl'):
            shape = pickle.load(f).shape
        else:
            major, minor = np.lib.format.read_magic(f)
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
    return shape


class AVSDDataSet(data.Dataset):
    def __init__(self, dataset_file, fea_types, fea_path, include_caption=False, vocab_path = '', vocabfile='', dictmap=None,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True, test_mode=False):

        self.vocab_path = vocab_path
        self.include_caption = include_caption
        self.vocabfile = vocabfile

        logging.info('Extracting words from ' + vocab_path)
        if self.vocabfile == '':
            self._get_vocabulary()

        self.dataset_file  = dataset_file
        self.dictmap = dictmap
        self.fea_types = fea_types
        self.fea_path = fea_path
        logging.info('Loading data from ' + dataset_file)


        self._make_batch_indices()

        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode

        if self.modality == 'RGBDiff':
            self.new_length += 1# Diff needs one more image to calculate diff

        self._parse_list()

    def _get_vocabulary(self, cutoff=1):
        self.vocab = {'<unk>': 0, '<sos>': 1, '<eos>': 2}
        if self.vocabfile != '':
            vocab_from_file = json.load(open(self.vocabfile, 'r'))
            for w in vocab_from_file:
                if w not in self.vocab:
                    self.vocab[w] = len(self.vocab)
            return
        dialog_data = json.load(open(self.vocab_path, 'r'))
        word_freq = {}
        for dialog in dialog_data['dialogs']:
            if self.include_caption:
                for word in dialog['caption'].split():
                    if word in word_freq:
                        word_freq[word] += 1
                    else:
                        word_freq[word] = 1
            for key in ['question', 'answer']:
                for turn in dialog['dialog']:
                    for word in turn[key].split():
                        if word in word_freq:
                            word_freq[word] += 1
                        else:
                            word_freq[word] = 1
        for word, freq in word_freq.items():
            if freq > cutoff:
                self.vocab[word] = len(self.vocab)

    # Load text data
    def _load(self):

        dialog_data = json.load(open(self.dataset_file, 'r'))

        unk = self.vocab['<unk>']
        eos = self.vocab['<eos>']
        dialog_list = []
        vid_set = set()
        qa_id = 0
        for dialog in dialog_data['dialogs']:
            if self.include_caption:
                caption = [words2ids(dialog['caption'], self.vocab, eos=eos)]
            else:
                caption = [np.array([eos], dtype=np.int32)]

            questions = [words2ids(d['question'], self.vocab) for d in dialog['dialog']]
            answers = [words2ids(d['answer'], self.vocab) for d in dialog['dialog']]
            qa_pair = [np.concatenate((q, a, [eos])).astype(np.int32) for q, a in zip(questions, answers)]

            vid = self.dictmap[dialog['image_id']] if self.dictmap is not None else dialog['image_id']
            vid_set.add(vid)
            for n in six.moves.range(len(questions)):
                history = copy.copy(caption)
                for m in six.moves.range(n):
                    history.append(qa_pair[m])
                question = np.concatenate((questions[n], [eos])).astype(np.int32)
                answer_in = np.concatenate(([eos], answers[n])).astype(np.int32)
                answer_out = np.concatenate((answers[n], [eos])).astype(np.int32)
                dialog_list.append((vid, qa_id, history, question, answer_in, answer_out))
                qa_id += 1

        self.data = {'dialogs': dialog_list, 'vocab': self.vocab, 'features': [],
                'original': dialog_data}
        for ftype in self.fea_types:
            basepath = self.fea_path.replace('<FeaType>', ftype)
            features = {}
            for vid in vid_set:
                filepath = basepath.replace('<ImageID>', vid)
                shape = get_npy_shape(filepath)
                features[vid] = (filepath, shape[0])
            self.data['features'].append(features)

    def _make_batch_indices(self):
        # Setup mini-batches
        self.idxlist = []
        for n, dialog in enumerate(self.data['dialogs']):
            vid = dialog[0]  # video ID
        x_len = []
        for feat in data['features']:
            value = feat[vid]
            size = value[1] if isinstance(value, tuple) else len(value)
            x_len.append(size)

            qa_id = dialog[1]  # QA-pair id
            h_len = len(dialog[2])  # history length
            q_len = len(dialog[3])  # question length
            a_len = len(dialog[4])  # answer length
            self.idxlist.append((vid, qa_id, x_len, h_len, q_len, a_len))

        if self.batchsize > 1:
            self.idxlist = sorted(self.idxlist, key=lambda s: (-s[3], -s[2][0], -s[4], -s[5]))


    def __len__(self):
        return len(self.idxlist)

    def __getitem__(self, item):
        , batchsize = 100, max_length = 20
        batch_indices = []
        bs = 0
        while bs < n_samples:
            in_len = idxlist[bs][3]
            bsize = batchsize / (in_len / max_length + 1)
            be = min(bs + bsize, n_samples) if bsize > 0 else bs + 1
            x_len = [max(idxlist[bs:be], key=lambda s: s[2][j])[2][j]
                     for j in six.moves.range(len(x_len))]
            h_len = max(idxlist[bs:be], key=lambda s: s[3])[3]
            q_len = max(idxlist[bs:be], key=lambda s: s[4])[4]
            a_len = max(idxlist[bs:be], key=lambda s: s[5])[5]
            vids = [s[0] for s in idxlist[bs:be]]
            qa_ids = [s[1] for s in idxlist[bs:be]]
            batch_indices.append((vids, qa_ids, x_len, h_len, q_len, a_len, be - bs))
            bs = be

        return batch_indices, n_samples




















def make_batch(data, index, eos=1):
    x_len, h_len, q_len, a_len, n_seqs = index[2:]
    feature_info = data['features']
    for j in six.moves.range(n_seqs):
        vid = index[0][j]
        fea = [np.load(fi[vid][0]) for fi in feature_info]
        for i in six.moves.range(len(x_len)):
            if fea[i].ndim > 3:
                fea[i] = fea[i].mean(axis=(2, 3)).reshape((-1, 2048))
        if j == 0:
            x_batch = [np.zeros((x_len[i], n_seqs, fea[i].shape[-1]), 
                       dtype=np.float32) for i in six.moves.range(len(x_len))]

        for i in six.moves.range(len(feature_info)):
            print(fea[i].shape)
            x_batch[i][:len(fea[i]), j] = fea[i]

    empty_sentence = np.array([eos], dtype=np.int32)
    h_batch = [ [] for _ in six.moves.range(h_len) ]
    q_batch = []
    a_batch_in = []
    a_batch_out = []
    dialogs = data['dialogs']
    for i in six.moves.range(n_seqs):
        qa_id = index[1][i]
        history, question, answer_in, answer_out = dialogs[qa_id][2:]
	for j in six.moves.range(h_len):
	    if j < len(history):
	        h_batch[j].append(history[j])
	    else:
	        h_batch[j].append(empty_sentence)
        q_batch.append(question)
        a_batch_in.append(answer_in)
        a_batch_out.append(answer_out)

    return x_batch, h_batch, q_batch, a_batch_in, a_batch_out


def feature_shape(data):
    dims = []
    for features in data["features"]:
        sample_feature = features.values()[0]
        if isinstance(sample_feature, tuple):
	    dims.append(2048)#np.load(sample_feature[0]).shape[-1])
        else:
            dims.append(sample_feature.shape[-1])
    return dims
