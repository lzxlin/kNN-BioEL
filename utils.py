import json
import random

import numpy as np
import pickle
import torch
import itertools
import torch.nn.functional as F
from options import args
from tqdm import tqdm
from collections import defaultdict
from transformers import BertTokenizer


def flatten(lists):
    """
    flat a list of lists
    ----
    [[1, 1, 1, 3], [1]] -> [1,1,1,3,1]
    """
    merged = list(itertools.chain(*lists))
    return merged


class Tokenizer:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_dir)

    def tokenize(self, mentions):
        mentions = list(mentions)
        input_ids = []
        attention_mask = []
        for mention in mentions:
            mention_tokens = self.tokenizer.tokenize(mention)
            tokens_max_len = int(args.mentions_max_length - 2)
            if len(mention_tokens) > tokens_max_len:
                mention_tokens = mention_tokens[:tokens_max_len]
            mention_tokens.insert(0, '[CLS]')
            mention_tokens.append('[SEP]')
            mention_tokens_id = self.tokenizer.convert_tokens_to_ids(mention_tokens)

            mention_masks = [1] * len(mention_tokens_id)
            mention_padding = [0] * (args.mentions_max_length - len(mention_tokens_id))
            mention_tokens_id += mention_padding
            mention_masks += mention_padding

            input_ids.append(mention_tokens_id)
            attention_mask.append(mention_masks)
        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask)
        }


def preprocess(text):
    return stringQ2B(text).lower()


def prepare_instance(filename, id2words, word2ids):
    instances = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            items = line.strip().split('||')
            # assert len(items) == 4
            if len(items) == 4:
                mention, _id, word, context = items
                mention = preprocess(mention)
                word = preprocess(word)
                all_cuis = word2ids[word] + [_id]
                gold_set = [word]
                for cui in all_cuis:
                    for id in cui.split('|'):
                        gold_set += id2words[id]
                gold_set = list(set(gold_set))
            elif len(items) == 2:
                # for bc5cdr-disease/chemical
                mention, _id = items
                mention = preprocess(mention)
                all_cuis = [_id]
                gold_set = id2words.get(_id, [])
                if not gold_set:
                    for cui in all_cuis:
                        for id in cui.split('|'):
                            gold_set += id2words[id]
                gold_set = list(set(gold_set))
                # print(mention, _id, len(gold_set))
                # if not gold_set:
                #     print(mention, _id)
                #     raise NotImplementedError
                word = random.choice(gold_set)  # 随机选择一个gold
                # word = list(gold_set)[0]
                word = preprocess(word)
                context = ''
                # print(mention, _id, word)
            else:
                raise NotImplementedError
            mention_gold = {
                'mention': mention,
                'gold': word,
                'context': context,
                'cui': _id,
                'gold_set': gold_set,
            }
            instances.append(mention_gold)
    return instances


def load_dictionaries(filename):
    id2words = defaultdict(list)
    word2ids = defaultdict(list)
    with open(filename, "r", encoding='utf-8') as f:
        for line in tqdm(f, desc='load dictionaries'):
            line = line.strip()
            if not line:
                continue
            cui, name = line.split("||")
            name = preprocess(name)
            cuis = [cui] + cui.split('|')
            for cui in set(cuis):
                id2words[cui].append(name)
                word2ids[name].append(cui)
    return id2words, word2ids


from torch.utils.data import Dataset


class TrainDataset(Dataset):

    def __init__(self, X, D):
        self.X = X
        self.D = D  # id --> words
        self.neg_samples = None  # for hard negative samples
        self.tokenizer = Tokenizer()
        self.label2id = {}
        count = 0
        for cui, names in self.D.items():
            for name in names:
                self.label2id[name] = count
            count += 1

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        res = self.X[idx]
        res['label'] = self.label2id[res['gold']]
        if self.neg_samples is not None:
            res['neg'] = self.neg_samples[idx]
            res['neg_label'] = [self.label2id[word] for word in self.neg_samples[idx]]
        return res

    def set_neg_samples(self, neg_samples):
        self.neg_samples = neg_samples

    def collate_fn(self, batch):
        mentions = [b['mention'] for b in batch]
        golds = [b['gold'] for b in batch]
        labels = [b['label'] for b in batch]
        labels = torch.tensor(labels)
        mention_inputs = self.tokenizer.tokenize(mentions)
        gold_inputs = self.tokenizer.tokenize(golds)
        if self.neg_samples is not None:
            negs = flatten([b['neg'] for b in batch])
            neg_inputs = self.tokenizer.tokenize(negs)
            neg_labels = flatten([b['neg_label'] for b in batch])
            neg_labels = torch.tensor(neg_labels)
        else:
            neg_inputs = None
            neg_labels = None
        return mention_inputs, gold_inputs, neg_inputs, labels, neg_labels


class EvalDataset(Dataset):

    def __init__(self, X):
        self.X = X
        self.tokenizer = Tokenizer()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        res = self.X[idx]
        return res

    def collate_fn(self, batch):
        mentions = [b['mention'] for b in batch]
        mention_inputs = self.tokenizer.tokenize(mentions)
        return mention_inputs


class FlatDataset(Dataset):

    def __init__(self, X):
        self.X = X
        self.tokenizer = Tokenizer()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        res = self.X[idx]
        return res

    def collate_fn(self, batch):
        mention_inputs = self.tokenizer.tokenize(batch)
        return mention_inputs


def pad_sequence(x, max_len, type=int):
    padded_x = np.zeros((len(x), max_len), dtype=type)
    for i, row in enumerate(x):
        padded_x[i][:len(row)] = row

    return padded_x


def my_collate_bert(x):
    mention_inputs_id = [x_['mention_tokens_id'] for x_ in x]
    mention_masks = [x_['mention_masks'] for x_ in x]
    gold = [x_['gold'] for x_ in x]

    return mention_inputs_id, mention_masks, gold


def get_positive(targets):
    positive = []
    for target in targets:
        positive += [target] * args.hard
    return positive


def get_negative_hard(targets, model, tokens, masks, use_random=False):
    negative = []
    hard_number = args.hard
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    if use_random:
        single = []
        for target in targets:
            single
    with torch.no_grad():
        tokens = torch.LongTensor(tokens).to(device)
        masks = torch.LongTensor(masks).to(device)
        descriptions = model.get_descriptions(tokens, masks)
        single = []
        for target in targets:
            distance = F.pairwise_distance(descriptions[target], descriptions)
            sorted, indices = torch.sort(distance, descending=False)
            indices = indices.cpu().numpy().tolist()
            for indice in indices:
                if indice not in targets:
                    single.append(indice)
            single = single[:hard_number]
            negative.extend(single)
    return negative


def get_description():
    descriptions = list()
    with open("./data/yidu-n7k/code.txt", "r", encoding='utf-8') as f:
        for line in f.readlines():
            entity = line.split('\t')[1].strip()
            descriptions.append(entity)
    # print(f'total {len(descriptions)} descriptions')

    wp_tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
    all_tokens = []
    all_masks = []
    for description in descriptions:
        tokens = wp_tokenizer.tokenize(description)

        tokens_max_len = args.candidates_max_length - 2
        if len(tokens) > tokens_max_len:
            tokens = tokens[:tokens_max_len]

        tokens.insert(0, '[CLS]')
        tokens.append('[SEP]')

        tokens_id = wp_tokenizer.convert_tokens_to_ids(tokens)
        masks = [1] * len(tokens)
        candidate_padding = [0] * (args.candidates_max_length - len(tokens))
        tokens_id += candidate_padding
        masks += candidate_padding
        all_tokens.append(tokens_id)
        all_masks.append(masks)

    return all_tokens, all_masks


def Q2B(uchar):
    """单个字符 全角转半角"""
    inside_code = ord(uchar)
    if inside_code == 0x3000:
        inside_code = 0x0020
    else:
        inside_code -= 0xfee0
    if inside_code < 0x0020 or inside_code > 0x7e:  # 转完之后不是半角字符返回原来的字符
        return uchar
    return chr(inside_code)


def stringQ2B(ustring):
    """把字符串全角转半角"""
    return "".join([Q2B(uchar) for uchar in ustring])


def transfer_inputs_to_cuda(inputs, device):
    # inputs must be a list or tensor
    if inputs is None:
        return inputs
    if isinstance(inputs, torch.Tensor):
        return inputs.to(device)
    # tokenizer处理得到每个模态特征列表
    elif isinstance(inputs, dict):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        return inputs
    else:
        raise NotImplementedError


def concatenate(lists, dim=0):
    # filter None
    lists = [l for l in lists if l is not None]
    if len(lists) < 1:
        return torch.tensor([])
    return torch.cat(lists, dim=dim)


def get_split_sections(lists, dim=0):
    # 返回需要切分的大小
    return [l.shape[dim] for l in lists if l is not None]


def cos_similarity(a, b, temperature=1.0):
    # 二维矩阵相乘
    return torch.matmul(a, b.T) / temperature


def calculate_nearest_k(sim_mat, k):
    sorted_mat = -np.partition(-sim_mat, k + 1, axis=1)  # -np.sort(-sim_mat1)
    nearest_k = sorted_mat[:, 0:k]
    return np.mean(nearest_k, axis=1)


def csls_sim(sim_mat, k):
    """
    Compute pairwise csls similarity based on the input similarity matrix.

    Parameters
    ----------
    sim_mat : matrix-like
        A pairwise similarity matrix.
    k : int
        The number of nearest neighbors.

    Returns
    -------
    csls_sim_mat : A csls similarity matrix of n1*n2.
    """

    nearest_values1 = calculate_nearest_k(sim_mat, k)
    nearest_values2 = calculate_nearest_k(sim_mat.T, k)
    csls_sim_mat = 2 * sim_mat.T - nearest_values1
    csls_sim_mat = csls_sim_mat.T - nearest_values2
    return csls_sim_mat


def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
