# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss, BCELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling_ori_dis import BertForSequenceClassification, BertConfig, WEIGHTS_NAME, CONFIG_NAME
import pytorch_pretrained_bert.modeling_fast_dis as modeling_fast
#import pytorch_pretrained_bert.modeling_both as modeling_fast
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

import time
import copy
import json

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

wr_target = None

split = None
intv = None
wr_now = 1.

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")),
            "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test_matched.tsv")),
            "test")

    def get_testmm_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test_mismatched.tsv")),
            "test")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            if set_type == 'test':
                label = 'contradiction'
            else:
                label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")),
            "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test_mismatched.tsv")),
            "test")


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if set_type == 'test':
                text_a = line[1]
                label = '1'
            else:
                text_a = line[3]
                label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            if set_type == 'test':
                text_a = line[1]
                label = '0'
            else:
                text_a = line[0]
                label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QqpProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        if set_type == 'test':
            for (i, line) in enumerate(lines):
                if i == 0:
                    continue
                guid = "%s-%s" % (set_type, line[0])
                try:
                    text_a = line[1]
                    text_b = line[2]
                    if set_type == 'test':
                        label = '0'
                    # label = line[5]
                except IndexError:
                    continue
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            return examples
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:
                text_a = line[3]
                text_b = line[4]
                if set_type == 'test':
                    label = '0'
                else:
                    label = line[5]
                # label = line[5]
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QnliProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            if set_type == 'test':
                label = 'entailment'
            else:
                label = line[-1]
            # label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if False and ex_index % 100000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        '''if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))'''

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    tp = (outputs * labels).sum()
    tn = ((1 - outputs) * (1 - labels)).sum()
    fp = (outputs * (1 - labels)).sum()
    fn = ((1 - outputs) * labels).sum()
    mc = 1.0 * (tp * tn - fp * fn) / (((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5)
    return np.sum(outputs == labels), mc


def do_sparse(w, ratio, param_tensor, model):
    '''d1,d2 = w.shape
    # size = list(matrix.size())[0] * list(matrix.size())[1]
    # bottom k
    k = int(ratio * float(d1))
    #print('k',k,'d',d1,d2)
    # print(sparsity, size, k)
    bottom_k, indices = torch.topk(w.abs(), k, largest=False,dim=0)
    # topk_cpu = topk.cpu()
    # indices_cpu = indices.cpu()
    #w = torch.nn.Parameter(w.detach().scatter_(0, indices, torch.cuda.FloatTensor(k,d2).fill_(0)))
    model.state_dict()[param_tensor] = w.detach().scatter_(0, indices, torch.cuda.FloatTensor(k,d2).fill_(0))'''
    d1_old, d2_old = w.shape
    w = w.reshape(-1, 1)
    d1, d2 = w.shape
    # size = list(matrix.size())[0] * list(matrix.size())[1]
    # bottom k
    k = int(ratio * float(d1))
    # print('k',k,'d',d1,d2)
    # print(sparsity, size, k)
    bottom_k, indices = torch.topk(w.abs(), k, largest=False, dim=0)
    # topk_cpu = topk.cpu()
    # indices_cpu = indices.cpu()
    # w = torch.nn.Parameter(w.detach().scatter_(0, indices, torch.cuda.FloatTensor(k,d2).fill_(0)))
    w = w.detach().scatter_(0, indices, torch.cuda.FloatTensor(k, d2).fill_(0))
    w = w.reshape(d1_old, d2_old)
    return w
    # print((w==0).sum())
    # model.state_dict()[param_tensor] = w


def svd(mat, rank):
    U, sigma, VT = np.linalg.svd(mat)
    diag = np.sqrt(np.diag(sigma[:rank]))
    return torch.nn.Parameter(torch.from_numpy(np.matmul(U[:, :rank], diag)[:,:184]).float().cuda()), torch.nn.Parameter(
        torch.from_numpy(np.matmul(diag, VT[:rank, :])[:184,:]).float().cuda())


class prune_function:
    def __init__(self, args):
        self.args = args
        processors = {
            "cola": ColaProcessor,
            "mnli": MnliProcessor,
            "mnli-mm": MnliMismatchedProcessor,
            "mrpc": MrpcProcessor,
            "sst-2": Sst2Processor,
            "sts-b": StsbProcessor,
            "qqp": QqpProcessor,
            "qnli": QnliProcessor,
            "rte": RteProcessor,
            "wnli": WnliProcessor,
        }

        output_modes = {
            "cola": "classification",
            "mnli": "classification",
            "mrpc": "classification",
            "sst-2": "classification",
            "sts-b": "regression",
            "qqp": "classification",
            "qnli": "classification",
            "rte": "classification",
            "wnli": "classification",
        }

        if args.local_rank == -1 or args.no_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
            n_gpu = torch.cuda.device_count()
        else:
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
            n_gpu = 1
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.distributed.init_process_group(backend='nccl')
        logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            device, n_gpu, bool(args.local_rank != -1), args.fp16))
        self.device = device

        if args.gradient_accumulation_steps < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                args.gradient_accumulation_steps))

        args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)

        if not args.do_train and not args.do_eval:
            raise ValueError("At least one of `do_train` or `do_eval` must be True.")

        # if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        #    raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        task_name = args.task_name.lower()

        if task_name not in processors:
            raise ValueError("Task not found: %s" % (task_name))

        processor = processors[task_name]()
        output_mode = output_modes[task_name]

        label_list = processor.get_labels()
        self.label_list = label_list
        num_labels = len(label_list)

        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

        train_examples = None
        num_train_optimization_steps = None
        if args.do_train:
            train_examples = processor.get_train_examples(args.data_dir)
            self.num_train_optimization_steps = int(
                len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
            if args.local_rank != -1:
                self.num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        # f = open(output_eval_file, "w")
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, output_mode)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", self.num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)

        if output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        elif output_mode == "regression":
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        self.train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        self.eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        test_examples = processor.get_test_examples(args.data_dir)
        test_features = convert_examples_to_features(
            test_examples, label_list, args.max_seq_length, tokenizer, output_mode)
        all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
        # Run prediction for full data
        test_sampler = SequentialSampler(test_data)
        self.test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)
        if args.task_name == 'mnli':
            test_examples = processor.get_testmm_examples(args.data_dir)
            test_features = convert_examples_to_features(
                test_examples, label_list, args.max_seq_length, tokenizer, output_mode)
            all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
            test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
            # Run prediction for full data
            test_sampler = SequentialSampler(test_data)
            self.testmm_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

        '''cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                       'distributed_{}'.format(args.local_rank))
        model = BertForSequenceClassification.from_pretrained(args.bert_model,
                                                              cache_dir=cache_dir,
                                                              num_labels=num_labels)'''

        if args.bert_model == 'bert-base-uncased':
            layer_num = 12  # others not implemented
            old_dim = 768

        if args.svd_weight_dir is None:
            cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                           'distributed_{}'.format(args.local_rank))
            model = modeling_fast.BertForSequenceClassification.from_pretrained(args.bert_model,
                                                                                cache_dir=cache_dir,
                                                                                num_labels=num_labels)
            for i in range(layer_num):
                pt = model.bert.encoder.layer[i].attention.self
                pt.qmat1, pt.qmat2 = svd(pt.query.weight.detach().cpu().numpy(), old_dim)
                pt.kmat1, pt.kmat2 = svd(pt.key.weight.detach().cpu().numpy(), old_dim)
                pt.vmat1, pt.vmat2 = svd(pt.value.weight.detach().cpu().numpy(), old_dim)
                pt = model.bert.encoder.layer[i].attention.output
                pt.dmat1, pt.dmat2 = svd(pt.dense.weight.detach().cpu().numpy(), old_dim)
                pt = model.bert.encoder.layer[i].intermediate
                pt.dmat1, pt.dmat2 = svd(pt.dense.weight.detach().cpu().numpy(), old_dim)
                pt = model.bert.encoder.layer[i].output
                pt.dmat1, pt.dmat2 = svd(pt.dense.weight.detach().cpu().numpy(), old_dim)
                print('init weight finish')

            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
            statedict = model_to_save.state_dict()
            torch.save(statedict, output_model_file)
            #torch.save(statedict, '/home/yayaming/maoyh/mnli_svd_weight_184/'+WEIGHTS_NAME)
            output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
            f1 = open(output_config_file, 'w+')
            f1.write(model_to_save.config.to_json_string())
            f1.close()
            # Load a trained model and config that you have fine-tuned
            config = modeling_fast.BertConfig(output_config_file)
            model = modeling_fast.BertForSequenceClassification(config, num_labels=num_labels)
            model.load_state_dict(torch.load(output_model_file))
        else:
            if args.bert_model == 'bert-base-uncased':
                svd_weight = args.svd_weight_dir
                # if num_labels==2:
                #    svd_weight+='_2'
                output_model_file = os.path.join(svd_weight, WEIGHTS_NAME)
                output_config_file = os.path.join(svd_weight, CONFIG_NAME)
                if args.cont_model!='':
                    output_model_file=os.path.join(args.cont_model, WEIGHTS_NAME)
            '''else:
                output_model_file = os.path.join('/home/yujwang/maoyh/svd_weight_large', WEIGHTS_NAME)
                output_config_file = os.path.join('/home/yujwang/maoyh/svd_weight_large', CONFIG_NAME)'''
            config = modeling_fast.BertConfig(output_config_file)
            model = modeling_fast.BertForSequenceClassification(config, num_labels=num_labels)
            model.load_state_dict(torch.load(output_model_file), strict=False)

        if args.fp16:
            model.half()
        model.to(device)
        if args.local_rank != -1:
            try:
                from apex.parallel import DistributedDataParallel as DDP
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            model = DDP(model)
        elif n_gpu > 1:
            model = torch.nn.DataParallel(model)
        self.model = model
        self.output_mode = output_mode
        self.num_labels = num_labels
        self.n_gpu = n_gpu

        # model_t
        distill_weight = args.distill_dir  # '/home/yujwang/maoyh/sst_distill_weight'
        distill_weight_file = os.path.join(distill_weight, WEIGHTS_NAME)
        config = BertConfig(output_config_file)
        model_t = BertForSequenceClassification(config, num_labels=num_labels)
        model_t.eval()
        model_t.load_state_dict(torch.load(distill_weight_file), strict=False)
        model_t.cuda()
        self.model_t = model_t
        print('init finish')

    def eval_after_train(self, prune_type, target_prune_rate):
        args = self.args
        device = self.device
        output_mode = self.output_mode
        num_labels = self.num_labels
        n_gpu = self.n_gpu

        now_step = 0
        layer_num = 12

        model = self.model  # copy.deepcopy(self.model)
        model_t = self.model_t
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        if args.fp16:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=args.learning_rate,
                                  bias_correction=False,
                                  max_grad_norm=1.0)
            if args.loss_scale == 0:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                optimizer = FP16_Optimizer(optimizer, static_loss_scale=self.args.loss_scale)

        else:
            optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=args.learning_rate,
                                 warmup=args.warmup_proportion,
                                 t_total=self.num_train_optimization_steps)

        global_step = 0
        nb_tr_steps = 0
        tr_loss = 0
        model.train()

        best_acc = 0
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        f = open(output_eval_file, "a")

        global wr_now, intv
        loss_mse = MSELoss()
        def soft_cross_entropy(predicts, targets):
            student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
            targets_prob = torch.nn.functional.softmax(targets, dim=-1)
            return (- targets_prob * student_likelihood).mean()

        for epoch_i in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0.
            tr_att_loss = 0.
            tr_rep_loss = 0.
            tr_cls_loss = 0.
            nb_tr_examples, nb_tr_steps = 0, 0
            start = time.time()
            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):
                if False and global_step > 0 and global_step % 50 == 0:

                    model.eval()
                    eval_loss, eval_accuracy = 0, 0
                    eval_mc = 0
                    nb_eval_steps, nb_eval_examples = 0, 0

                    for input_ids, input_mask, segment_ids, label_ids in tqdm(self.eval_dataloader, desc="Evaluating"):
                        input_ids = input_ids.to(device)
                        input_mask = input_mask.to(device)
                        segment_ids = segment_ids.to(device)
                        label_ids = label_ids.to(device)

                        with torch.no_grad():
                            tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids,
                                                  p_type=prune_type, p_rate=prune_rate)
                            logits = model(input_ids, segment_ids, input_mask, p_type=prune_type, p_rate=prune_rate)

                        logits = logits.detach().cpu().numpy()
                        label_ids = label_ids.to('cpu').numpy()
                        tmp_eval_accuracy, mc = accuracy(logits, label_ids)

                        eval_loss += tmp_eval_loss.mean().item()
                        eval_accuracy += tmp_eval_accuracy
                        eval_mc += mc

                        nb_eval_examples += input_ids.size(0)
                        nb_eval_steps += 1

                    eval_loss = eval_loss / nb_eval_steps
                    eval_accuracy = eval_accuracy / nb_eval_examples

                if epoch_i == 0:
                    if wr_now < step // intv and wr_now < split:  # sr_now, wr_now are increased here
                        if wr_now < wr_target:
                            wr_now += 1
                else:
                    wr_now = wr_target
                prune_rate = [1.]*48# sr's temporary prune rate is assigned here, unused

                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                att_loss = 0.
                rep_loss = 0.
                cls_loss = 0.

                student_logits, student_atts, student_reps = model(input_ids, segment_ids, input_mask,
                                                                        p_type=prune_type, p_rate=prune_rate)

                with torch.no_grad():
                    teacher_logits, teacher_atts, teacher_reps = model_t(input_ids, segment_ids, input_mask)

                # if not args.pred_distill:
                teacher_layer_num = len(teacher_atts)
                student_layer_num = len(student_atts)
                assert teacher_layer_num % student_layer_num == 0
                layers_per_block = int(teacher_layer_num / student_layer_num)
                new_teacher_atts = [teacher_atts[i * layers_per_block + layers_per_block - 1]
                                    for i in range(student_layer_num)]

                for student_att, teacher_att in zip(student_atts, new_teacher_atts):
                    student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(device),
                                              student_att)
                    teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(device),
                                              teacher_att)

                    tmp_loss = loss_mse(student_att, teacher_att)
                    att_loss += tmp_loss

                new_teacher_reps = [teacher_reps[i * layers_per_block] for i in range(student_layer_num)]
                new_student_reps = student_reps
                for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
                    tmp_loss = loss_mse(student_rep, teacher_rep)
                    rep_loss += tmp_loss

                loss = rep_loss + att_loss
                tr_att_loss += att_loss.item()
                tr_rep_loss += rep_loss.item()
                # else:
                if output_mode == "classification":
                    cls_loss = soft_cross_entropy(student_logits / 1,#args.temperature,
                                                  teacher_logits / 1)#args.temperature)
                elif output_mode == "regression":
                    loss_mse = MSELoss()
                    cls_loss = loss_mse(student_logits.view(-1), label_ids.view(-1))

                loss += cls_loss
                tr_cls_loss += cls_loss.item()


                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(
                            global_step / self.num_train_optimization_steps,
                            args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                # sparse
                if step > 0:  # weight pruning is done here
                    # prune embd
                    r = args.embd_r
                    embd = model.bert.embeddings.word_embeddings.weight
                    do_sparse(embd, r, None, model)

                    for layer_now in range(layer_num):
                        svd_ch = [model.bert.encoder.layer[layer_now].attention.self.qmat1[:,
                                  :model.bert.encoder.layer[layer_now].attention.self.to_dim],
                                  model.bert.encoder.layer[layer_now].attention.self.qmat2[
                                  :model.bert.encoder.layer[layer_now].attention.self.to_dim, :],
                                  model.bert.encoder.layer[layer_now].attention.self.kmat1[:,
                                  :model.bert.encoder.layer[layer_now].attention.self.to_dim],
                                  model.bert.encoder.layer[layer_now].attention.self.kmat2[
                                  :model.bert.encoder.layer[layer_now].attention.self.to_dim, :],
                                  model.bert.encoder.layer[layer_now].attention.self.vmat1[:,
                                  :model.bert.encoder.layer[layer_now].attention.self.to_dim],
                                  model.bert.encoder.layer[layer_now].attention.self.vmat2[
                                  :model.bert.encoder.layer[layer_now].attention.self.to_dim, :],
                                  model.bert.encoder.layer[layer_now].attention.output.dmat1[:,
                                  :model.bert.encoder.layer[layer_now].attention.output.to_dim],
                                  model.bert.encoder.layer[layer_now].attention.output.dmat2[
                                  :model.bert.encoder.layer[layer_now].attention.output.to_dim, :],
                                  model.bert.encoder.layer[layer_now].intermediate.dmat1[:,
                                  :model.bert.encoder.layer[layer_now].intermediate.to_dim],
                                  model.bert.encoder.layer[layer_now].intermediate.dmat2[
                                  :model.bert.encoder.layer[layer_now].intermediate.to_dim, :],
                                  model.bert.encoder.layer[layer_now].output.dmat1[:,
                                  :model.bert.encoder.layer[layer_now].output.to_dim],
                                  model.bert.encoder.layer[layer_now].output.dmat2[
                                  :model.bert.encoder.layer[layer_now].output.to_dim, :]]
                        svd_ch_temp = [model.bert.encoder.layer[layer_now].attention.self.qmat1,
                                       model.bert.encoder.layer[layer_now].attention.self.qmat2,
                                       model.bert.encoder.layer[layer_now].attention.self.kmat1,
                                       model.bert.encoder.layer[layer_now].attention.self.kmat2,
                                       model.bert.encoder.layer[layer_now].attention.self.vmat1,
                                       model.bert.encoder.layer[layer_now].attention.self.vmat2,
                                       model.bert.encoder.layer[layer_now].attention.output.dmat1,
                                       model.bert.encoder.layer[layer_now].attention.output.dmat2,
                                       model.bert.encoder.layer[layer_now].intermediate.dmat1,
                                       model.bert.encoder.layer[layer_now].intermediate.dmat2,
                                       model.bert.encoder.layer[layer_now].output.dmat1,
                                       model.bert.encoder.layer[layer_now].output.dmat2]
                        id = [0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3]
                        for i in range(12):
                            r = 1 - target_prune_rate[layer_now * 4 + id[i]] ** (
                                    1. * wr_now / split)  # 1-target_r^{sr_now/split} for each layer
                            svd_ch_temp[i] = torch.nn.Parameter(do_sparse(svd_ch_temp[i], r, None, model))

                            '''print(i, (model.bert.encoder.layer[layer_now].attention.self.qmat1[:,
                                      :model.bert.encoder.layer[layer_now].attention.self.to_dim] == 0).sum())
                            print(i, (model.bert.encoder.layer[layer_now].attention.self.qmat2[
                                      :model.bert.encoder.layer[layer_now].attention.self.to_dim, :] == 0).sum())'''

                        svd_ch_check = [model.bert.encoder.layer[layer_now].attention.self.qmat1,
                                        model.bert.encoder.layer[layer_now].attention.self.qmat2,
                                        model.bert.encoder.layer[layer_now].attention.self.kmat1,
                                        model.bert.encoder.layer[layer_now].attention.self.kmat2,
                                        model.bert.encoder.layer[layer_now].attention.self.vmat1,
                                        model.bert.encoder.layer[layer_now].attention.self.vmat2,
                                        model.bert.encoder.layer[layer_now].attention.output.dmat1,
                                        model.bert.encoder.layer[layer_now].attention.output.dmat2,
                                        model.bert.encoder.layer[layer_now].intermediate.dmat1,
                                        model.bert.encoder.layer[layer_now].intermediate.dmat2,
                                        model.bert.encoder.layer[layer_now].output.dmat1,
                                        model.bert.encoder.layer[layer_now].output.dmat2]
                now_step += 1
                # if now_step == all_steps:
                # break
                if global_step%2000==0:

                    model.eval()
                    eval_loss, eval_accuracy = 0, 0
                    eval_mc = 0
                    nb_eval_steps, nb_eval_examples = 0, 0

                    for input_ids, input_mask, segment_ids, label_ids in tqdm(self.eval_dataloader, desc="Evaluating"):
                        input_ids = input_ids.to(device)
                        input_mask = input_mask.to(device)
                        segment_ids = segment_ids.to(device)
                        label_ids = label_ids.to(device)

                        with torch.no_grad():
                            logits,_,_ = model(input_ids, segment_ids, input_mask, p_type=prune_type, p_rate=prune_rate)

                        logits = logits.detach().cpu().numpy()
                        label_ids = label_ids.to('cpu').numpy()
                        tmp_eval_accuracy, mc = accuracy(logits, label_ids)

                        eval_accuracy += tmp_eval_accuracy
                        eval_mc += mc

                        nb_eval_examples += input_ids.size(0)
                        nb_eval_steps += 1

                    eval_accuracy = eval_accuracy / nb_eval_examples
                    if eval_accuracy > best_acc:
                        best_acc = eval_accuracy
                        to_test = True
                    else:
                        to_test = False
                    eval_mc = eval_mc / nb_eval_steps
                    result = {'eval_accuracy': eval_accuracy,
                              'mc': eval_mc,'step':global_step}
                    logger.info("Eval Loss: %s", result)
                    f.write("Eval Loss: %s\n" % result)
                    f.close()
                    f = open(output_eval_file, "a")

                    if to_test:  # below is the output of test dataset
                        model_to_save = model.module if hasattr(model,
                                                                'module') else model  # Only save the model it-self
                        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
                        statedict = model_to_save.state_dict()
                        torch.save(statedict, output_model_file)

                        model.eval()
                        ans = np.array([])
                        for input_ids, input_mask, segment_ids in tqdm(self.test_dataloader, desc="test"):
                            input_ids = input_ids.to(device)
                            input_mask = input_mask.to(device)
                            segment_ids = segment_ids.to(device)

                            with torch.no_grad():
                                logits,_,_ = model(input_ids, segment_ids, input_mask, p_type=prune_type, p_rate=prune_rate)

                            logits = logits.detach().cpu().numpy()
                            outputs = np.argmax(logits, axis=1)
                            ans = np.concatenate((ans, outputs))

                        if args.task_name == 'cola':
                            f1 = open('tst.tsv', 'w')
                            f1.write('index\tprediction\n')
                            for i in range(1, ans.shape[0]):
                                f1.write(str(i - 1) + '\t' + str(int(ans[i])) + '\n')
                            f1.close()
                            f1 = open(os.path.join(args.output_dir, 'tst.tsv'), "w")
                            f1.write('index\tprediction\n')
                            for i in range(1, ans.shape[0]):
                                f1.write(str(i - 1) + '\t' + str(int(ans[i])) + '\n')
                            f1.close()
                        elif args.task_name == 'mnli':
                            model.eval()
                            ans = np.array([])
                            for input_ids, input_mask, segment_ids in tqdm(self.test_dataloader, desc="test"):
                                input_ids = input_ids.to(device)
                                input_mask = input_mask.to(device)
                                segment_ids = segment_ids.to(device)

                                with torch.no_grad():
                                    logits,_,_ = model(input_ids, segment_ids, input_mask, p_type=prune_type,
                                                   p_rate=prune_rate)

                                logits = logits.detach().cpu().numpy()
                                outputs = np.argmax(logits, axis=1)
                                ans = np.concatenate((ans, outputs))

                            f1 = open('tst.tsv', 'w')
                            f1.write('index\tprediction\n')
                            for i in range(ans.shape[0]):
                                f1.write(str(i) + '\t' + self.label_list[int(ans[i])] + '\n')
                            f1.close()
                            f1 = open(os.path.join(args.output_dir, 'tst.tsv'), "w")
                            f1.write('index\tprediction\n')
                            for i in range(ans.shape[0]):
                                f1.write(str(i) + '\t' + self.label_list[int(ans[i])] + '\n')
                            f1.close()

                            model.eval()
                            ans = np.array([])
                            for input_ids, input_mask, segment_ids in tqdm(self.testmm_dataloader, desc="test"):
                                input_ids = input_ids.to(device)
                                input_mask = input_mask.to(device)
                                segment_ids = segment_ids.to(device)

                                with torch.no_grad():
                                    logits,_,_ = model(input_ids, segment_ids, input_mask, p_type=prune_type,
                                                   p_rate=prune_rate)

                                logits = logits.detach().cpu().numpy()
                                outputs = np.argmax(logits, axis=1)
                                ans = np.concatenate((ans, outputs))

                            f1 = open('tstmm.tsv', 'w')
                            f1.write('index\tprediction\n')
                            for i in range(ans.shape[0]):
                                f1.write(str(i) + '\t' + self.label_list[int(ans[i])] + '\n')
                            f1.close()
                            f1 = open(os.path.join(args.output_dir, 'tstmm.tsv'), "w")
                            f1.write('index\tprediction\n')
                            for i in range(ans.shape[0]):
                                f1.write(str(i) + '\t' + self.label_list[int(ans[i])] + '\n')
                            f1.close()
                        else:
                            f1 = open('tst.tsv', 'w')
                            f1.write('index\tprediction\n')
                            for i in range(ans.shape[0]):
                                f1.write(str(i) + '\t' + self.label_list[int(ans[i])] + '\n')
                            f1.close()
                            f1 = open(os.path.join(args.output_dir, 'tst.tsv'), "w")
                            f1.write('index\tprediction\n')
                            for i in range(ans.shape[0]):
                                f1.write(str(i) + '\t' + self.label_list[int(ans[i])] + '\n')
                            f1.close()
            out = {'epoch': epoch_i, 'loss': tr_loss / (step + 1), 'time': time.time() - start}
            logger.info("Train Loss: %s", out)


        return best_acc


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument("--p_encoder",
                        default=0.1,
                        type=float,
                        help="Target overall ratio of pruning.")
    parser.add_argument("--split",
                        default=64,
                        type=int,
                        help="Total rounds of iterative pruning. Don't change.")
    parser.add_argument("--svd_weight_dir",
                        default=None,
                        type=str,
                        help="Here it's /home/yujwang/maoyh/svd_weight. By default for sst-2 there're 3 categories. For MNLI, it's /home/yujwang/maoyh/svd_weight_2, 2 categories.")
    parser.add_argument("--svd_ratio",
                        default=18,
                        type=int,
                        help="svd_ratio times in all 64 times.")
    parser.add_argument("--layerwise",
                        default=None,
                        type=str,
                        help="Whether apply different ratio to different layers")
    parser.add_argument("--p_embd",
                        default=0.4,
                        type=float,
                        help="svd_ratio times in all 64 times.")
    parser.add_argument("--lw",
                        default=0.4,
                        type=float,
                        help="1:lw in ratio alignment.")
    parser.add_argument("--distill_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="Distilled model directory")
    parser.add_argument("--cont_model",
                        default='',
                        type=str,
                        help="continue model")
    parser.add_argument("--svd_dim",
                        default=256,
                        type=int,
                        help="?/768")
    args = parser.parse_args()
    args.embd_r=1.-args.p_embd
    args.target_r=args.p_encoder

    def balance(prune_rate, target):
        rate_all = 0
        whole_param = [768 * args.svd_dim*2 * 3, 768 * args.svd_dim*2, args.svd_dim * (3072+768), args.svd_dim * (3072+768)]
        rate_one = 0
        for i in range(48):
            rate_all += prune_rate[i] * whole_param[i % 4]
            rate_one += whole_param[i % 4]
        rate_all /= rate_one
        for i in range(48):
            prune_rate[i] = prune_rate[i] / rate_all * target
        return prune_rate

    to_test_sr = [args.svd_ratio]  # [16,18,20,22,24,26,28,30] # grid search of sr(svd ratio), given sr+wr=split
    result = {}

    global split, intv, psteps
    task_step = {'mnli': 10000, 'sst-2': 2100, 'qnli': 3000, 'qqp': 10000, 'mrpc': 100, 'cola': 200}
    psteps = task_step[args.task_name]
    split = args.split
    intv = psteps // (split + 1)

    for tst in to_test_sr:
        global sr_target, wr_target
        sr_target, wr_target = 0,64#tst, split - tst
        if args.layerwise is None:
            prune_type = ['svd'] * 48  # default: combine svd and weight pruning
            prune_rate = [0.1]*48#[0.4,0.43,0.4,0.43] * 12
            result['rate']=prune_rate
            prune_rate = balance(prune_rate, args.target_r)
        else:  # to use a given layerwise assignment of pruning ratio
            filename = args.layerwise
            sp = json.load(open(filename))
            print("use layerwise ratio from: " + filename)
            prune_rate = []
            prune_type = ['svd'] * 48
            for i in range(48):
                prune_rate.append((sp['pr' + str(i)] - 0.3) / 0.4 * args.lw + 1.)
            prune_rate = balance(prune_rate, args.target_r)
        print(prune_rate, prune_type)
        func = prune_function(args)
        result[str(sr_target)] = func.eval_after_train(prune_type, prune_rate)
        print(result)
        json.dump(result, open('result.json', 'a'))


if __name__ == "__main__":
    main()
