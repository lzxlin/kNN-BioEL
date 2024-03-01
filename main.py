import csv
import sys
import os
import gc
import time
import datetime
import random
import torch
import argparse
import numpy as np
from loguru import logger
from torch.utils.data import DataLoader
from train_test import train, test
from utils import *
from models import pick_model
from transformers import AdamW, get_linear_schedule_with_warmup
from loss import MetricLearning


def mean(data):
    return sum(data) / len(data)


def run(args):

    seed = args.random_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

    logger.info(args)

    now = datetime.datetime.now()
    time_formatted = str(datetime.date.today()) + '-' + now.strftime("%H-%M-%S")
    logger.info(f'time_formatted: {time_formatted}')

    # prepare for output
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')

    model = pick_model(args)

    if args.load_model:
        pretrained_model_path = args.load_model + '/' + model.name
        state_dict = torch.load(pretrained_model_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)

    if 'bc5cdr-disease' in args.data_dir or 'bc5cdr-chemical' in args.data_dir:
        is_bc5cdr = True
    else:
        is_bc5cdr = False

    if is_bc5cdr:
        id2words, word2ids = load_dictionaries(args.data_dir + '/' + 'train_dictionary.txt')
        eval_id2words, eval_word2ids = load_dictionaries(args.data_dir + '/' + 'dev_dictionary.txt')
        test_id2words, test_word2ids = load_dictionaries(args.data_dir + '/' + 'test_dictionary.txt')
    else:
        id2words, word2ids = load_dictionaries(args.data_dir + '/' + 'test_dictionary.txt')
        eval_id2words, eval_word2ids = id2words, word2ids
        test_id2words, test_word2ids = id2words, word2ids
    print(f'dictionaries {len(id2words)} (concepts); {sum([len(v) for k, v in id2words.items()])} (words)')
    train_instances = prepare_instance(args.data_dir + '/' + 'train_queries.txt', id2words, word2ids)
    print("train_instances {}".format(len(train_instances)))
    dev_instances = prepare_instance(args.data_dir + '/' + 'dev_queries.txt', id2words, word2ids)
    print("dev_instances {}".format(len(dev_instances)))
    test_instances = prepare_instance(args.data_dir + '/' + 'test_queries.txt', id2words, word2ids)
    print("test_instances {}".format(len(test_instances)))

    def check_queries(instances):
        for sample in instances:
            mention = sample['mention']
            word = sample['gold']
            cui = sample['cui']
            # print(cui, mention, 1, word, word2ids[word])
            if word not in word2ids:
                # print(f'word `{word}` not in dict.')
                pass
            elif cui not in word2ids[word]:
                # print(f'the label `{cui}` of word `{word}` maybe not right.')
                pass

    check_queries(train_instances)
    check_queries(dev_instances)
    check_queries(test_instances)

    total_train_num = len(train_instances)
    train_num = int(len(train_instances) * args.train_ratio)
    train_instances = train_instances[:train_num]
    logger.info(f'used train_instances: {len(train_instances)}/{total_train_num}; ratio: {args.train_ratio}')

    train_dataset = TrainDataset(train_instances, id2words)
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)

    test_dataset = EvalDataset(test_instances)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)

    optimizer = AdamW(
        model.parameters(),
        # optimizer_grouped_parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
        # eps=1e-8
    )

    metric_loss = MetricLearning(
        use_cuda=True if args.gpu else False,
        temperature=args.temperature,
        broadcast=args.broadcast,
    )
    if args.gpu and torch.cuda.is_available():
        metric_loss = metric_loss.to(device=device)

    total_steps = len(train_loader) * args.n_epochs

    # Create the learning rate scheduler.
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=0,
    #     num_training_steps=total_steps
    # )
    scheduler = None

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_param_names = [n for n, p in model.named_parameters() if p.requires_grad]
    # logger.info('trainable layer name are: {}'.format(trainable_param_names))
    logger.info("total params num is {}".format(total_params))

    logger.info('Start training !!!')
    best_eval_acc1 = 0
    best_eval_acc5 = 0
    best_test_acc1 = 0
    best_test_acc5 = 0
    best_epoch = 0
    step_global = 0
    start = time.time()

    dict_embs = None
    for epoch in range(args.n_epochs):
        epoch_start = time.time()

        if args.online_neg_num > 0:
            neg_samples = model.online_hard_negative_mining(
                queries=train_instances,
                dictionaries=list(word2ids.keys()),
                topk=args.online_neg_num,
                show_progress=True,
                dict_embeds=dict_embs
            )
        else:
            neg_samples = None
        train_dataset.set_neg_samples(neg_samples=neg_samples)

        train_loss, step_global = train(
            args, epoch, train_loader, model, metric_loss, optimizer, scheduler, step_global=step_global
        )
        epoch_finish = time.time()
        logger.info('loss/train_per_epoch={:4}/{}, cost time: {}/s'.format(
            train_loss, epoch, round(epoch_finish - epoch_start, 3))
        )

        # save model every epoch
        if args.save_checkpoint_all:
            checkpoint_dir = os.path.join(args.output_dir, "checkpoint_{}".format(epoch))
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'model.pt'))

        # save model for last epoch
        if epoch == args.n_epochs:
            torch.save(args.output_dir, os.path.join(args.output_dir, 'model.pt'))

        if args.gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        dict_embs = None
        if args.use_eval:
            logger.info('Eval at epoch {}'.format(epoch))
            if is_bc5cdr:
                eval_result, eval_accs, dict_embs = test(
                    args, model, dev_instances, list(eval_word2ids.keys()), args.top_k, eval_word2ids
                )
            else:
                eval_result, eval_accs, dict_embs = test(
                    args, model, dev_instances, list(eval_word2ids.keys()), args.top_k, eval_word2ids,
                    dict_embs=dict_embs
                )
            eval_acc1, eval_acc5 = eval_accs[f'acc1'], eval_accs[f'acc5']
            logger.info(f'epoch: {epoch}; [eval] eval_acc1: {eval_acc1}; eval_acc5: {eval_acc5}')

            if eval_acc1 > best_eval_acc1:
                best_eval_acc1 = eval_acc1
                best_eval_acc5 = eval_acc5
                best_epoch = epoch
                if is_bc5cdr:
                    test_result, test_accs, dict_embs = test(
                        args, model, test_instances, list(test_word2ids.keys()), args.top_k, test_word2ids,
                    )
                else:
                    test_result, test_accs, dict_embs = test(
                        args, model, test_instances, list(test_word2ids.keys()), args.top_k, test_word2ids,
                        dict_embs=dict_embs
                    )
                best_test_acc1, best_test_acc5 = test_accs[f'acc1'], test_accs[f'acc5']
                logger.info(f'epoch: {epoch}; [test] test_acc1: {best_test_acc1}; test_acc5: {best_test_acc5}')

                # save model for best epoch
                if args.save_checkpoint_best:
                    if args.add_save_params:
                        checkpoint_dir = os.path.join(
                            args.output_dir,
                            f"checkpoint_best_{args.batch_size}_{args.lr}_{args.online_neg_num}_"
                            f"{args.temperature}_{args.train_ratio}_{time_formatted}"
                        )
                    else:
                        checkpoint_dir = os.path.join(args.output_dir, "checkpoint_best")
                    if not os.path.exists(checkpoint_dir):
                        os.makedirs(checkpoint_dir)
                    torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'model.pt'))
                    # save recall
                    save_json(eval_result, os.path.join(checkpoint_dir, 'recall_result.json'))

            logger.info('[eval] best acc1/5: {}/{}, [test] best acc1/5: {}/{}, best epoch: {}'.
                        format(round(best_eval_acc1, 4),
                               round(best_eval_acc5, 4),
                               round(best_test_acc1, 4),
                               round(best_test_acc5, 4),
                               best_epoch))

    end = time.time()
    training_time = end - start
    training_hour = int(training_time / 60 / 60)
    training_minute = int(training_time / 60 % 60)
    training_second = int(training_time % 60)
    logger.info("Training Time!{} hours {} minutes {} seconds".format(training_hour, training_minute, training_second))

    gc.collect()
    torch.cuda.empty_cache()

    sys.stdout.flush()

    return best_eval_acc1, best_eval_acc5, best_test_acc1, best_test_acc5


def run_aap(args):
    eval_acc1_list = []
    eval_acc5_list = []
    test_acc1_list = []
    test_acc5_list = []
    start = time.time()
    src_data_dir, src_output_dir = args.data_dir, args.output_dir
    for i in range(10):

        logger.info(f'\nEval fold {i}>>>>>>')

        args.data_dir = os.path.join(args.data_dir, f'fold{i}', 'formatted')
        args.output_dir = os.path.join(args.output_dir, f'fold{i}')
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        best_eval_acc1, best_eval_acc5, best_test_acc1, best_test_acc5 = run(args)
        eval_acc1_list.append(best_eval_acc1)
        eval_acc5_list.append(best_eval_acc5)
        test_acc1_list.append(best_test_acc1)
        test_acc5_list.append(best_test_acc5)

        args.data_dir = src_data_dir
        args.output_dir = src_output_dir

    logger.info('=='*30)
    logger.info('eval result:')
    logger.info(f'eval_acc1_list: {eval_acc1_list}')
    logger.info(f'eval_acc5_list: {eval_acc5_list}')
    logger.info('test result:')
    logger.info(f'test_acc1_list: {test_acc1_list}')
    logger.info(f'test_acc5_list: {test_acc5_list}')
    logger.info('avg result:')
    logger.info(f'[eval] acc1: {round(mean(eval_acc1_list), 4)}; acc5: {round(mean(eval_acc5_list), 4)}; '
                f'[test] acc1: {round(mean(test_acc1_list), 4)}; acc5: {round(mean(test_acc5_list), 4)};')
    end = time.time()
    training_time = end - start
    training_hour = int(training_time / 60 / 60)
    training_minute = int(training_time / 60 % 60)
    training_second = int(training_time % 60)
    logger.info(" Fold Training Time!{} hours {} minutes {} seconds".format(training_hour, training_minute, training_second))
    logger.info('=='*30)


if __name__ == "__main__":
    from options import args

    if 'aap' in args.data_dir:
        run_aap(args)
    else:
        run(args)
