import csv
import sys
import os
import gc
import time
import random
import torch
import argparse
import numpy as np
from tqdm import tqdm
from loguru import logger
from torch.utils.data import DataLoader
from train_test import train, test
from utils import *
from models import pick_model
from transformers import AdamW, get_linear_schedule_with_warmup
from loss import MetricLearning


def run(args):

    start = time.time()
    if args.random_seed != 0:
        seed = args.random_seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False  # 设置为True可以让卷积加速
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True

    logger.info(args)

    # prepare for output
    if not os.path.exists(args.save_result_dir):
        os.makedirs(args.save_result_dir)

    model = pick_model(args)

    if args.load_model:
        state_dict = torch.load(os.path.join(args.load_model, 'model.pt'), map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)

    id2words, word2ids = load_dictionaries(args.data_dir + '/' + 'test_dictionary.txt')
    print(f'dictionaries {len(id2words)} (concepts); {sum([len(v) for k, v in id2words.items()])} (words)')
    train_instances = prepare_instance(args.data_dir + '/' + 'train_queries.txt', id2words, word2ids)
    print("train_instances {}".format(len(train_instances)))
    dev_instances = prepare_instance(args.data_dir + '/' + 'dev_queries.txt', id2words, word2ids)
    print("dev_instances {}".format(len(dev_instances)))
    test_instances = prepare_instance(args.data_dir + '/' + 'test_queries.txt', id2words, word2ids)
    print("test_instances {}".format(len(test_instances)))

    # 构造retrieval store
    train_id2words, train_word2ids = defaultdict(set), defaultdict(set)
    for sample in train_instances:
        mention = sample['mention']
        cui = sample['cui']
        cuis = list(set([cui] + cui.split('|')))
        for cui in cuis:
            train_id2words[cui].add(mention)
        train_word2ids[mention] |= set(cuis)
    train_word2ids = {k: list(v) for k, v in train_word2ids.items()}
    # print(train_word2ids)

    def check_queries(instances):
        for sample in instances:
            mention = sample['mention']
            word = sample['gold']
            cui = sample['cui']
            # print(cui, mention, 1, word, word2ids[word])
            if word not in word2ids:
                print(f'word `{word}` not in dict.')
            elif cui not in word2ids[word]:
                print(f'the label `{cui}` of word `{word}` maybe not right.')

    check_queries(train_instances)
    check_queries(dev_instances)
    check_queries(test_instances)

    logger.info('Start Evaluation !!!')

    # word2ids = {k: word2ids[k] for i, k in enumerate(word2ids) if i < 10}

    dict_embs = None
    train_result, train_accs, dict_embs = test(
        args, model, train_instances, list(word2ids.keys()), args.top_k, word2ids, dict_embs=dict_embs
    )
    logger.info(f'[train] acc1: {train_accs["acc1"]}; acc5: {train_accs["acc5"]}')
    dev_result, dev_accs, dict_embs = test(
        args, model, dev_instances, list(word2ids.keys()), args.top_k, word2ids, dict_embs=dict_embs
    )
    logger.info(f'[dev] acc1: {dev_accs["acc1"]}; acc5: {dev_accs["acc5"]}')
    test_result, test_accs, dict_embs = test(
        args, model, test_instances, list(word2ids.keys()), args.top_k, word2ids, dict_embs=dict_embs
    )
    logger.info(f'[test] acc1: {test_accs["acc1"]}; acc5: {test_accs["acc5"]}')

    knn_dict_embs = None
    train_knn_result, train_knn_accs, knn_dict_embs = test(
        args, model, train_instances, list(train_word2ids.keys()), args.top_k, train_word2ids, dict_embs=knn_dict_embs
    )
    logger.info(f'[train-knn] acc1: {train_knn_accs["acc1"]}; acc5: {train_knn_accs["acc5"]}')
    dev_knn_result, dev_knn_accs, knn_dict_embs = test(
        args, model, dev_instances, list(train_word2ids.keys()), args.top_k, train_word2ids, dict_embs=knn_dict_embs
    )
    logger.info(f'[dev-knn] acc1: {dev_knn_accs["acc1"]}; acc5: {dev_knn_accs["acc5"]}')
    test_knn_result, test_knn_accs, knn_dict_embs = test(
        args, model, test_instances, list(train_word2ids.keys()), args.top_k, train_word2ids, dict_embs=knn_dict_embs
    )
    logger.info(f'[test-knn] acc1: {test_knn_accs["acc1"]}; acc5: {test_knn_accs["acc5"]}')

    save_json(train_result, os.path.join(args.save_result_dir, 'train_recall_result.json'))  # too large
    save_json(train_knn_result, os.path.join(args.save_result_dir, 'train_knn_result.json'))

    save_json(dev_result, os.path.join(args.save_result_dir, 'dev_recall_result.json'))
    save_json(test_result, os.path.join(args.save_result_dir, 'test_recall_result.json'))
    save_json(dev_knn_result, os.path.join(args.save_result_dir, 'dev_knn_result.json'))
    save_json(test_knn_result, os.path.join(args.save_result_dir, 'test_knn_result.json'))

    end = time.time()
    training_time = end - start
    training_hour = int(training_time / 60 / 60)
    training_minute = int(training_time / 60 % 60)
    training_second = int(training_time % 60)
    logger.info(
        "Evaluation Time!{} hours {} minutes {} seconds".format(training_hour, training_minute, training_second))

    gc.collect()
    torch.cuda.empty_cache()

    sys.stdout.flush()


def run_aap(args):
    src_data_dir = args.data_dir
    arc_save_result_dir = args.save_result_dir
    src_load_model = args.load_model
    for i in tqdm(range(10)):
        args.data_dir = os.path.join(src_data_dir, f'fold{i}', 'formatted')
        args.save_result_dir = os.path.join(arc_save_result_dir, f'fold{i}')
        if args.load_model:
            args.load_model = os.path.join(os.path.dirname(src_load_model), f'fold{i}', os.path.split(src_load_model)[-1])

        run(args)


if __name__ == "__main__":
    from options import args

    if 'aap' in args.data_dir:
        run_aap(args)
    else:
        run(args)
