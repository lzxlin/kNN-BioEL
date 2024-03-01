import sys
import os
import gc
import time
import copy
from tqdm import tqdm
from loguru import logger
from scipy.special import softmax
from torch.utils.data import DataLoader
from train_test import train, test, evaluate_topk_acc
from utils import *


def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def get_retrieval_map(data):
    res = defaultdict(list)
    for idx, sample in enumerate(data):
        mention, candidates = sample['mention'], sample['candidates']
        res[mention] = candidates
    return res


def evaluate_aap(eval_data_dict, topk=5, mode='test'):
    avg_acc = defaultdict(list)
    for key in eval_data_dict:
        eval_data = evaluate_topk_acc(eval_data_dict[key], topk=topk)
        for k in range(1, topk + 1):
            avg_acc[k].append(eval_data[f"acc{k}"])
    res = {}
    for k, v in avg_acc.items():
        print('[{}] acc@{}: {:.2f}'.format(mode, k, np.average(v) * 100))
        res[f'acc{k}'] = np.average(v)
    return res


def print_acc(res, mode='train'):
    print(f'[{mode}] acc1: {res["acc1"]}; acc5: {res["acc5"]}')


def combine_dists(test_data, knn_map, topk, agg='max', alpha=1.0, cand_temp=1.0, knn_temp=1.0, cand_num=10000):
    inf = -100000

    test_data2 = copy.deepcopy(test_data)

    for idx, sample in tqdm(enumerate(test_data2)):
        mention = str(sample['mention'])
        candidates = sample['candidates'][:cand_num]

        # if candidates[0]['label'] != 1:
        #     print(mention, golden_cui)

        # 预测topk分布
        cand_dist = [val['score'] for val in candidates]
        src_dist = cand_dist.copy()

        # 构造knn分布
        label2idx = defaultdict(list)
        for idx, val in enumerate(candidates):
            label = val['labelcui']  # cui list
            for l in label:
                label2idx[l].append(idx)
        knn_dist = [inf for _ in range(len(candidates))]
        for demo in knn_map[mention][:topk]:
            label = demo['labelcui']
            score = demo['score']
            for l in label:
                for _id in label2idx[l]:
                    if agg == 'max':
                        knn_dist[_id] = max(knn_dist[_id], score)
                    elif agg == 'sum':
                        if knn_dist[_id] == inf:
                            knn_dist[_id] = 0
                        knn_dist[_id] += score
                    elif agg == 'mean':
                        pass
                    else:
                        raise NotImplementedError()

        cand_dist = softmax(np.array(cand_dist) / cand_temp)
        knn_dist = softmax(np.array(knn_dist) / knn_temp)

        dist = alpha * knn_dist + (1 - alpha) * cand_dist
        rank_ids = np.argsort(dist)[::-1].tolist()
        dist = dist.tolist()

        # rerank candidates
        new_candidates = [sample['candidates'][_id] for _id in rank_ids]
        # update prob
        # for _id, cand in zip(rank_ids, new_candidates):
        #     cand['score'] = dist[_id]
        sample['candidates'] = new_candidates

        # sample['candidates'] = knn_map[mention]  # lazy learning

        # print(src_dist)
        # print(knn_dist.tolist())
        # print(dist)
        # print(rank_ids)
        # print('===')
    return test_data2


def run(args):
    start = time.time()

    # train_data = load_data(os.path.join(args.save_result_dir, 'train_recall_result.json'))
    dev_data = load_data(os.path.join(args.save_result_dir, 'dev_recall_result.json'))
    test_data = load_data(os.path.join(args.save_result_dir, 'test_recall_result.json'))
    dev_knn_data = load_data(os.path.join(args.save_result_dir, 'dev_knn_result.json'))
    test_knn_data = load_data(os.path.join(args.save_result_dir, 'test_knn_result.json'))

    print(f'Load data finished. cost time {round(time.time() - start)}/s')

    # train_res = evaluate_topk_acc(train_data, args.top_k)
    # print_acc(train_res, mode='train')
    dev_res = evaluate_topk_acc(dev_data, args.top_k)
    print_acc(dev_res, mode='eval')
    test_res = evaluate_topk_acc(test_data, args.top_k)
    print_acc(test_res, mode='test')
    dev_knn_res = evaluate_topk_acc(dev_knn_data, args.top_k)
    print_acc(dev_knn_res, mode='eval-knn')
    test_knn_res = evaluate_topk_acc(test_knn_data, args.top_k)
    print_acc(test_knn_res, mode='test-knn')

    dev_knn_map = get_retrieval_map(dev_knn_data)
    test_knn_map = get_retrieval_map(test_knn_data)

    best_hypers = []
    cand_num = 50  # 默认所有数据集都是top-50的分布
    for agg in ['max']:
        for cand_num in [500]:
            for alpha in [args.alpha]:
                # 0.01, 0.05, 0.1, 0.15, 0.2, 0.5, 1.0
                for temp1 in [args.beta1]:
                # for temp1 in [1.0, 0.1, 0.01, 0.001]:
                #     for temp2 in [1.0, 0.1, 0.2, 0.5, 0.01, 0.05]:
                    for temp2 in [args.beta2]:
                        for k in [args.knn]:
                            print('alpha: {:.4f}; cand_temp: {:.4f}; knn_temp: {:.4f}; '
                                  'topk: {}; cand_num: {}; agg: {}'.format(
                                alpha, temp1, temp2, k, cand_num, agg))

                            # eval_data2 = combine_dists(dev_data, dev_knn_map, topk=k, agg=agg, alpha=alpha,
                            #                            cand_temp=temp1,
                            #                            knn_temp=temp2,
                            #                            cand_num=cand_num)
                            # eval_data2 = evaluate_topk_acc(eval_data2, topk=10)
                            # eval_acc1 = eval_data2['acc1']
                            # print_acc(eval_data2, mode='eval')
                            eval_acc1 = 0

                            test_data2 = combine_dists(test_data, test_knn_map, topk=k, agg=agg, alpha=alpha,
                                                       cand_temp=temp1,
                                                       knn_temp=temp2,
                                                       cand_num=cand_num)
                            test_data2 = evaluate_topk_acc(test_data2, topk=10)
                            test_acc1 = test_data2['acc1']
                            print_acc(test_data2, mode='test')

                            best_hypers.append(
                                [eval_acc1 * 100, test_acc1 * 100, alpha, temp1, temp2, k, cand_num, agg])

                            if len(best_hypers) % 10 == 0:
                                print_hypers = sorted(best_hypers, key=lambda x: x[1], reverse=True)[:50]
                                print(f'evaluation at: {len(best_hypers)};')
                                for hyper in print_hypers:
                                    print('eval_acc: {:.2f}; test_acc: {:.2f}; alpha: {:.4f}; cand_temp: {:.2f}; '
                                          'knn_temp: {:.2f}; topk: {}; cand_num: {}; agg: {}'.format(*hyper))

    print_hypers = sorted(best_hypers, key=lambda x: x[1], reverse=True)[:50]
    print(f' atevaluation at last;')
    for hyper in print_hypers:
        print('eval_acc: {:.2f}; test_acc: {:.2f}; alpha: {:.4f}; cand_temp: {:.2f}; '
              'knn_temp: {:.2f}; topk: {}; cand_num: {}; agg: {}'.format(*hyper))

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
    start = time.time()

    dev_data_dict, test_data_dict, dev_knn_dict, test_knn_dict = {}, {}, {}, {}
    dev_knn_map, test_knn_map = {}, {}

    for i in tqdm(range(10), desc='load dataset'):
        path = os.path.join(args.save_result_dir, f'fold{i}')
        # train_data = load_data(os.path.join(args.save_result_dir, 'train_recall_result.json'))
        dev_data = load_data(os.path.join(path, 'dev_recall_result.json'))
        test_data = load_data(os.path.join(path, 'test_recall_result.json'))
        dev_knn_data = load_data(os.path.join(path, 'dev_knn_result.json'))
        test_knn_data = load_data(os.path.join(path, 'test_knn_result.json'))
        test_data_dict[i] = test_data
        dev_data_dict[i] = dev_data
        test_knn_dict[i] = test_knn_data
        dev_knn_dict[i] = dev_knn_data
        dev_knn_map[i] = get_retrieval_map(dev_knn_data)
        test_knn_map[i] = get_retrieval_map(test_knn_data)

    print(f'Load data finished. cost time {round(time.time() - start)}/s')

    evaluate_aap(dev_data_dict, topk=args.top_k, mode='eval')
    evaluate_aap(dev_knn_dict, topk=args.top_k, mode='eval-knn')
    evaluate_aap(test_data_dict, topk=args.top_k, mode='test')
    evaluate_aap(test_knn_dict, topk=args.top_k, mode='test-knn')


    best_hypers = []
    cand_num = 50  # 默认所有数据集都是top-50的分布
    for agg in ['max']:
        for cand_num in [500]:
            for alpha in [args.alpha]:
                # 0.01, 0.05, 0.1, 0.15, 0.2, 0.5, 1.0
                for temp1 in [args.beta1]:
                    for temp2 in [args.beta2]:
                        for k in [args.knn]:
                            print('alpha: {:.4f}; cand_temp: {:.2f}; knn_temp: {:.2f}; '
                                  'topk: {}; cand_num: {}; agg: {}'.format(
                                alpha, temp1, temp2, k, cand_num, agg))

                            test_data_dict2 = {}
                            for i  in range(10):
                                # eval_data2 = combine_dists(dev_data, dev_knn_map, topk=k, agg=agg, alpha=alpha,
                                #                            cand_temp=temp1,
                                #                            knn_temp=temp2,
                                #                            cand_num=cand_num)
                                # eval_data2 = evaluate_topk_acc(eval_data2, topk=10)
                                # eval_acc1 = eval_data2['acc1']
                                # print_acc(eval_data2, mode='eval')
                                eval_acc1 = 0

                                test_data2 = combine_dists(test_data_dict[i], test_knn_map[i], topk=k, agg=agg, alpha=alpha,
                                                           cand_temp=temp1,
                                                           knn_temp=temp2,
                                                           cand_num=cand_num)
                                test_data_dict2[i] = test_data2
                            test_res = evaluate_aap(test_data_dict2, topk=args.top_k, mode='test')
                            test_acc1 = test_res['acc1']
                            print_acc(test_res, mode='test')

                            best_hypers.append(
                                [eval_acc1 * 100, test_acc1 * 100, alpha, temp1, temp2, k, cand_num, agg])

                            if len(best_hypers) % 10 == 0:
                                print_hypers = sorted(best_hypers, key=lambda x: x[1], reverse=True)[:50]
                                print(f'evaluation at: {len(best_hypers)};')
                                for hyper in print_hypers:
                                    print('eval_acc: {:.2f}; test_acc: {:.2f}; alpha: {:.4f}; cand_temp: {:.2f}; '
                                          'knn_temp: {:.2f}; topk: {}; cand_num: {}; agg: {}'.format(*hyper))

    print_hypers = sorted(best_hypers, key=lambda x: x[1], reverse=True)[:50]
    print(f' atevaluation at last;')
    for hyper in print_hypers:
        print('eval_acc: {:.2f}; test_acc: {:.2f}; alpha: {:.4f}; cand_temp: {:.2f}; '
              'knn_temp: {:.2f}; topk: {}; cand_num: {}; agg: {}'.format(*hyper))

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


if __name__ == "__main__":
    from options import args

    if 'aap' in args.save_result_dir:
        run_aap(args)
    else:
        run(args)
