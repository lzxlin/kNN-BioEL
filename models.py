import faiss
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from transformers import BertModel
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from utils import *
from options import args
from sklearn.metrics.pairwise import cosine_similarity


class Bert(nn.Module):

    def __init__(self, args):
        super(Bert, self).__init__()

        self.name = args.model
        self.gpu = args.gpu
        self.agg_mode = args.agg_mode
        self.bert = BertModel.from_pretrained(args.bert_dir)
        self.loss = F.triplet_margin_loss
        self.device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')

    def get_embedding(self, inputs):
        out = self.bert(**inputs, output_hidden_states=True, return_dict=True)
        if self.agg_mode == "cls":
            dense_emb = out.last_hidden_state[:, 0]  # query : [batch_size, hidden]
        elif self.agg_mode == "cls-mlp":
            dense_emb = out.pooler_output  # query : [batch_size, hidden]
        elif self.agg_mode == "mean-all-tok":
            dense_emb = out.last_hidden_state.mean(1)  # query : [batch_size, hidden]
        elif self.agg_mode == "mean":
            dense_emb = (out.last_hidden_state * inputs['attention_mask'].unsqueeze(-1)).sum(1) / \
                        inputs['attention_mask'].sum(-1).unsqueeze(-1)
        elif self.agg_mode == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)
            dense_emb = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)
        elif self.agg_mode == 'first-last-avg':
            all_hidden_states = out.hidden_states
            lengths = inputs['attention_mask'].sum(dim=1, keepdim=True)  # (bsz, 1)
            dense_emb = (
                    (all_hidden_states[-0] * inputs['attention_mask'].unsqueeze(-1)).sum(dim=1) +
                    (all_hidden_states[-1] * inputs['attention_mask'].unsqueeze(-1)).sum(dim=1)
            ).div(2 * lengths)  # (bsz, hdim)
        else:
            raise NotImplementedError()
        return dense_emb

    def forward(self, batch):

        # query_inputs, std_inputs, neg_inputs
        query_inputs, std_inputs, neg_inputs, labels, neg_labels = batch

        query_embs = self.get_embedding(query_inputs)
        std_embs = self.get_embedding(std_inputs)

        # encoding negative samples
        if neg_inputs is not None:
            neg_embs = self.get_embedding(neg_inputs)
            batch_size = len(query_embs)
            neg_num = int(len(neg_embs) / batch_size)
        else:
            neg_embs = None

        train_embs = concatenate([query_embs, std_embs, neg_embs], dim=0)
        train_labels = concatenate([labels, labels, neg_labels], dim=0)
        split_sections = get_split_sections([labels, labels, neg_labels], dim=0)
        return train_embs, train_labels, split_sections

    def retrieve_candidate(self, score_matrix, topk):
        """
        Return sorted topk idxes (descending order)
        Parameters
        ----------
        score_matrix : np.array
            2d numpy array of scores
        topk : int
            The number of candidates
        Returns
        -------
        topk_idxs : np.array
            2d numpy array of scores [# of query , # of dict]
        """

        # logging.info('retrieve candidate')

        def indexing_2d(arr, cols):
            rows = np.repeat(np.arange(0, cols.shape[0])[:, np.newaxis], cols.shape[1], axis=1)
            return arr[rows, cols]

        # get topk indexes without sorting
        topk_idxs = np.argpartition(score_matrix, -topk)[:, -topk:]

        # get topk indexes with sorting
        topk_score_matrix = indexing_2d(score_matrix, topk_idxs)
        topk_argidxs = np.argsort(-topk_score_matrix)
        topk_idxs = indexing_2d(topk_idxs, topk_argidxs)

        return topk_idxs

    def retrieve_candidate_cuda(self, score_matrix, topk, batch_size=128, show_progress=False):
        """
        Return sorted topk idxes (descending order) using cuda

        Parameters
        ----------
        score_matrix : np.array
            2d numpy array of scores
        topk : int
            The number of candidates
        batch_size: int
            pass
        show_progress: bool
            pass
        Returns
        -------
        topk_idxs : np.array
            2d numpy array of scores [# of query , # of dict]
        """

        res = None
        for i in tqdm(np.arange(0, score_matrix.shape[0], batch_size), disable=not show_progress):
            score_matrix_tmp = torch.tensor(score_matrix[i:i + batch_size]).to(self.device)
            matrix_sorted = torch.argsort(score_matrix_tmp, dim=1, descending=True)[:, :topk].cpu()
            if res is None:
                res = matrix_sorted
            else:
                res = torch.cat([res, matrix_sorted], dim=0)

        return res.numpy()

    def embed_dense(self, names, batch_size=1024, show_progress=False, norm=True):
        """
        Embedding data into dense representations
        """

        self.eval()  # prevent dropout

        if isinstance(names, np.ndarray):
            names = names.tolist()

        dense_embeds = []
        num = len(names)

        if num % batch_size:
            steps = num // batch_size + 1
        else:
            steps = num // batch_size

        with torch.no_grad():

            eval_datasets = FlatDataset(names)
            eval_dataloader = DataLoader(
                eval_datasets,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=eval_datasets.collate_fn
            )

            if show_progress:
                iterations = tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), desc='embed biostd dense')
            else:
                iterations = enumerate(eval_dataloader)

            # t3 = time.time()
            for idx, batch in iterations:
                batch = transfer_inputs_to_cuda(batch, self.device)
                batch_embeds = self.get_embedding(batch)
                batch_embeds = batch_embeds.cpu().detach().numpy()
                dense_embeds.append(batch_embeds)
            # t4 = time.time()
            # print('model encoder time: {:.3}/s'.format(t4-t3))

        dense_embeds = np.concatenate(dense_embeds, axis=0)
        if norm:
            dense_embeds = dense_embeds / np.linalg.norm(dense_embeds, axis=-1, keepdims=True)

        return dense_embeds

    def get_score_matrix(self, query_embeds, dict_embeds, cosine=False, normalise=False, csls_k=0):
        """
        Return score matrix (slow for sparse embedding)

        Parameters
        ----------
        query_embeds : np.array
            2d numpy array of query embeddings
        dict_embeds : np.array
            2d numpy array of query embeddings
        cosine: bool
            use sklearn cosine metric
        normalise: bool
            use normalization
        csls_k: int
            compute csls similarity if csls_k>0
        Returns
        -------
        score_matrix : np.array
            2d numpy array of scores
        """
        if cosine:
            score_matrix = cosine_similarity(query_embeds, dict_embeds)
        else:
            score_matrix = np.matmul(query_embeds, dict_embeds.T)

        if normalise:
            score_matrix = (score_matrix - score_matrix.min()) / (score_matrix.max() - score_matrix.min())

        if csls_k > 0:
            score_matrix = csls_sim(score_matrix, csls_k)

        return score_matrix

    def online_hard_negative_mining(
            self, queries, dictionaries,
            topk=5, batch_size=128,
            show_progress=False, random_sampling=False,
            dict_embeds=None
    ):
        """
        using faiss for quickly searching neighbor
        ----
        """
        query_list = [q['mention'] for q in queries]
        gold_label_list = [q['gold_set'] for q in queries]
        dict_list = np.array(dictionaries)
        with torch.no_grad():
            # embed dictionaries
            if dict_embeds is None:
                dict_embeds = self.embed_dense(names=dict_list, show_progress=show_progress, norm=True)
            d = dict_embeds.shape[1]
            index = faiss.IndexFlatIP(d)  # faiss
            index.add(dict_embeds)

            # embed queries
            if show_progress:
                iterations = tqdm(range(0, len(queries), batch_size), desc='online hard negative mining')
            else:
                iterations = range(0, len(queries), batch_size)

            neg_samples = []
            for idx in iterations:
                # embed batch
                query_batch = query_list[idx: idx + batch_size]
                gold_batch = gold_label_list[idx: idx + batch_size]
                embed_batch = self.embed_dense(names=query_batch, show_progress=False, norm=True)
                # search by faiss
                D, I = index.search(embed_batch, min(topk * 100, len(dict_embeds)))
                neg_batch = dict_list[I]

                # filter false negative samples, it also works for 1vN case
                if random_sampling:
                    # 在topN中随机采样topK
                    topn = topk * 25
                    _neg_batch = []
                    for neg, gold in zip(neg_batch, gold_batch):
                        candidates = neg[np.in1d(neg, gold, invert=True)][:topn]
                        np.random.shuffle(candidates)  # 就地打乱, 没有返回值
                        _neg_batch.append(candidates[:topk])
                    neg_batch = np.array(_neg_batch)
                else:
                    neg_batch = np.array(
                        [neg[np.in1d(neg, gold, invert=True)][:topk] for neg, gold in zip(neg_batch, gold_batch)]
                    )
                # for a, b, c in zip(query_batch, gold_batch, neg_batch):
                #     print(a, '++', b, '++', c)
                # print(neg_batch.shape)
                neg_samples.append(neg_batch)
            neg_samples = np.concatenate(neg_samples, axis=0)
            assert len(queries) == len(neg_samples)
        return neg_samples


def pick_model(args):
    if args.model == "bert":
        model = Bert(args)
    else:
        raise RuntimeError("wrong model name")
    if args.gpu >= 0:
        device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
        model.to(device)
    return model
