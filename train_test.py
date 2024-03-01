import torch
import numpy as np
from tqdm import tqdm
from loguru import logger
from utils import transfer_inputs_to_cuda


def train(args, epoch, data_loader, model, metric_loss, optimizer, scheduler, step_global=0):
    print("EPOCH %d" % epoch)

    losses = []
    train_loss = 0
    train_steps = 0

    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')

    model.train()
    optimizer.zero_grad()

    # loader
    data_iter = iter(data_loader)
    num_iter = len(data_loader)
    # all_tokens = torch.tensor(all_tokens).to(device)
    # all_masks = torch.tensor(all_masks).to(device)
    for i in tqdm(range(num_iter)):
        mention_inputs, gold_inputs, neg_inputs, labels, neg_labels = next(data_iter)
        mention_inputs = transfer_inputs_to_cuda(mention_inputs, device)
        gold_inputs = transfer_inputs_to_cuda(gold_inputs, device)
        neg_inputs = transfer_inputs_to_cuda(neg_inputs, device)
        labels = transfer_inputs_to_cuda(labels, device)
        neg_labels = transfer_inputs_to_cuda(neg_labels, device)
        train_embs, train_labels, split_sections = model([mention_inputs, gold_inputs, neg_inputs, labels, neg_labels])
        loss = metric_loss(train_embs, train_labels, split_sections)
        loss = loss / args.gradient_accumulation_step
        loss.backward()

        if args.extra_step:  # it can work sightly better and train stably, why?
            optimizer.step()

        train_loss += loss.item()
        train_steps += 1
        step_global += 1

        # optimizer.zero_grad()
        # optimizer.step()
        # scheduler.step()

        if (i + 1) % args.gradient_accumulation_step == 0:
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            optimizer.zero_grad()

        if step_global % args.checkpoint_step == 0:
            logger.info('Global Step: {}, train loss: {}'.format(step_global, train_loss / train_steps))
    train_loss /= (train_steps + 1e-9)

    return train_loss, step_global


def test(args, model, eval_data, dictionaries, topk, word2ids, batch_size=1024, use_norm=True, csls_k=0,
         dict_embs=None):
    y, yhat, ysort = [], [], []

    model.eval()
    with torch.no_grad():

        if dict_embs is None:
            dict_embs = model.embed_dense(names=dictionaries, show_progress=True, norm=use_norm, batch_size=batch_size)
        logger.info('dict dense embeds shape {}'.format(dict_embs.shape))

        queries = []

        for idx in tqdm(range(0, len(eval_data), batch_size)):
            batch = eval_data[idx:idx + batch_size]
            mentions = [b['mention'] for b in batch]

            men_embs = model.embed_dense(names=mentions, show_progress=False, norm=use_norm, batch_size=batch_size)
            # get score matrix
            score_matrix = model.get_score_matrix(
                query_embeds=men_embs,
                dict_embeds=dict_embs,
                csls_k=csls_k
            )

            # candidate_idxs = model.retrieve_candidate(
            #     score_matrix=score_matrix,
            #     topk=topk * 10
            # )
            candidate_idxs = model.retrieve_candidate_cuda(
                score_matrix=score_matrix,
                topk=topk * 10
            )

            # topk_scores = score_matrix[:, candidate_idxs].squeeze()  # (batch, topK)
            # assert len(np_candidates) == len(topk_scores)

            for idj, data in enumerate(batch):
                mention = data['mention']
                gold = data['gold']  # a list
                gold_set = data['gold_set']
                cui = data['cui']
                topk_scores = score_matrix[idj, candidate_idxs[idj]].squeeze()
                candidates = [dictionaries[idx] for idx in candidate_idxs[idj]]
                assert len(topk_scores) == len(candidates)
                dict_candidates = []
                for cand, score in zip(candidates, topk_scores):
                    dict_candidates.append({
                        'name': cand,
                        'labelcui': word2ids[cand],
                        'score': score.item(),  # np.float不能写入json,要转成python内置的
                        'label': 1 if sum([check_label(_id, cui) for _id in word2ids[cand]]) > 0 else 0
                    })
                queries.append({
                    'mention': mention,
                    'gold': gold,
                    'cui': cui,
                    'gold_set': gold_set,
                    'candidates': dict_candidates
                })
    res = evaluate_topk_acc(queries, topk)
    # print(len(queries))
    # print(queries[:2])
    # print(res)
    return queries, res, dict_embs


def test_bc5cdr(args, model, eval_data, dictionaries, topk, word2ids, batch_size=1024, use_norm=True, csls_k=0,
                dict_embs=None):
    y, yhat, ysort = [], [], []

    model.eval()
    with torch.no_grad():

        if dict_embs is None:
            dict_embs = model.embed_dense(names=dictionaries, show_progress=True, norm=use_norm, batch_size=batch_size)
        logger.info('dict dense embeds shape {}'.format(dict_embs.shape))

        queries = []

        for idx in tqdm(eval_data, total=len(eval_data), desc='eval bc5cdr'):
            mention = eval_data[idx]['mention']
            mentions = mention.replace('+', '|').split('|')
            cui = eval_data[idx]['cui']
            gold = eval_data[idx]['gold']
            gold_set = eval_data[idx]['gold_set']

            men_embs = model.embed_dense(names=mentions, show_progress=False, norm=use_norm, batch_size=batch_size)
            # get score matrix
            score_matrix = model.get_score_matrix(
                query_embeds=men_embs,
                dict_embeds=dict_embs,
                csls_k=csls_k
            )

            # candidate_idxs = model.retrieve_candidate(
            #     score_matrix=score_matrix,
            #     topk=topk * 10
            # )
            candidate_idxs = model.retrieve_candidate_cuda(
                score_matrix=score_matrix,
                topk=topk * 10
            )

            # topk_scores = score_matrix[:, candidate_idxs].squeeze()  # (batch, topK)
            # assert len(np_candidates) == len(topk_scores)

            dict_mentions = []  # 每个mention的候选集
            for idj in range(len(mentions)):
                topk_scores = score_matrix[idj, candidate_idxs[idj]].squeeze()
                candidates = [dictionaries[idx] for idx in candidate_idxs[idj]]
                assert len(topk_scores) == len(candidates)
                dict_candidates = []
                for cand, score in zip(candidates, topk_scores):
                    dict_candidates.append({
                        'name': cand,
                        'labelcui': word2ids[cand],
                        'score': score.item(),  # np.float不能写入json,要转成python内置的
                        'label': 1 if sum([check_label(_id, cui) for _id in word2ids[cand]]) > 0 else 0
                    })
                dict_mentions.append({
                    'split_mention': mention,
                    'candidates': dict_candidates
                })
            queries.append({
                'mention': mention,
                'gold': gold,
                'cui': cui,
                'gold_set': gold_set,
                'candidates': dict_candidates
            })
    res = evaluate_bc5cdr_topk_acc(queries, topk)
    # print(len(queries))
    # print(queries[:2])
    # print(res)
    return queries, res, dict_embs


def check_label(predicted_cui, golden_cui):
    """
    Some composite annotation didn't consider orders
    So, set label '1' if any cui is matched within composite cui (or single cui)
    Otherwise, set label '0'
    """
    return int(len(set(predicted_cui.split("|")).intersection(set(golden_cui.split("|")))) > 0)


def evaluate_topk_acc(data, topk):
    """
    evaluate acc@1~acc@k
    """
    queries = data
    k = topk
    res = {}

    for i in range(0, k):
        hit = 0
        flat_acc = 0
        mention_num = 0
        for query in queries:
            candidates = query['candidates'][:i + 1]  # to get acc@(i+1)
            label = np.any([candidate['label'] for candidate in candidates])
            hit += label
            mention_num += 1
            flat_acc += label

        res['acc{}'.format(i + 1)] = hit / len(queries)
        res['flat_acc{}'.format(i + 1)] = flat_acc / mention_num

    return res


def evaluate_bc5cdr_topk_acc(data, topk=None):
    """
    evaluate acc@1~acc@k
    """
    queries = data
    k = topk
    res = {}

    for i in range(0, k):
        hit = 0
        flat_acc = 0
        mention_num = 0
        for query in queries:
            mentions = query['candidates']
            mention_hit = 0
            for mention in mentions:
                candidates = mention['candidates'][:i + 1]  # to get acc@(i+1)
                label = np.any([candidate['label'] for candidate in candidates])
                mention_hit += label
                mention_num += 1
                flat_acc += label

            # When all mentions in a query are predicted correctly,
            # we consider it as a hit
            if mention_hit == len(mentions):
                hit += 1

        res['acc{}'.format(i + 1)] = hit / len(queries)
        res['flat_acc{}'.format(i + 1)] = flat_acc / mention_num

    return res