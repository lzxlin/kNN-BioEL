import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from tqdm import tqdm
from pytorch_metric_learning import (
    miners,
    losses
)
from utils import cos_similarity

LOGGER = logging.getLogger(__name__)


class NTXentLoss(nn.Module):

    def __init__(self,
                 device,
                 temperature=1.0,
                 broadcast=False,
                 bce_ratio=0.0):
        """
        ---
        param add_aa_neg: bool
        param add_bb_neg: bool
        param broadcast: bool
            broadcast hard negative for batch
        param bce_ratio: float
        """
        super(NTXentLoss, self).__init__()
        self.temp = temperature
        self.device = device
        self.broadcast = broadcast
        self.criterion = nn.BCEWithLogitsLoss()
        self.bce_ratio = bce_ratio

    def forward(self, embeded, labels, split_sections):
        # print(embeded.shape, labels.shape, split_sections)
        assert len(split_sections) > 1
        F.normalize(embeded)
        embeded_list = torch.split(embeded, split_size_or_sections=split_sections, dim=0)
        label_list = torch.split(labels, split_size_or_sections=split_sections, dim=0)
        cos_sims = []
        base = embeded_list[0]
        batch_size, hidden_dim = base.shape
        section_num = len(split_sections)
        for i in range(1, len(split_sections)):
            aug = embeded_list[i]
            if not self.broadcast:
                if (i == len(split_sections) - 1) and section_num > 2:
                    # hard negative
                    aug = torch.reshape(aug, shape=(batch_size, -1, hidden_dim))  # (batch, neg_num, hidden_dim)
                    # (batch, 1, hidden,dim) * (batch, hidden_dim, neg_num) -->  (batch, 1, neg_num)
                    cos_sim = torch.matmul(base.unsqueeze(1), torch.transpose(aug, 1, 2)).squeeze(1)
                else:
                    cos_sim = cos_similarity(base, aug)
            else:
                cos_sim = cos_similarity(base, aug)
            cos_sims.append(cos_sim)
        cos_sims = torch.cat(cos_sims, dim=1)

        # threshold = 0.85
        # pos_mask = (~(cos_sims > threshold)).float().to(self.device).detach()

        cos_sims_for_bce = cos_sims
        cos_sims_for_cl = cos_sims / self.temp

        # for numerical stability
        # https://github.com/HobbitLong/SupContrast/blob/a8a275b3a8b9b9bdc9c527f199d5b9be58148543/losses.py#L72
        logits_max, _ = torch.max(cos_sims_for_cl, dim=1, keepdim=True)
        cos_sims_for_cl = cos_sims_for_cl - logits_max.detach()

        # labels = torch.arange(cos_sims.size(0)).long().to(self.device)
        # loss = self.criteron(cos_sims, labels)

        if not self.broadcast:
            # labels = F.one_hot(torch.arange(start=0, end=cos_sims.shape[0], dtype=torch.int64),
            #                    num_classes=cos_sims.shape[1]).float().to(self.device)
            src_label = label_list[0].view(-1, 1)
            labels = []
            for i in range(1, len(split_sections)):
                tgt_label = label_list[i].view(batch_size, -1)
                if (i == len(split_sections) - 1) and section_num > 2:
                    labels.append(torch.eq(src_label, tgt_label))  # (B,1) * (B, neg_num) --> (B, neg_num)
                else:
                    labels.append(torch.eq(src_label, tgt_label.T))  # (B,1) * (1, B) --> (B, B)
            labels = torch.cat(labels, dim=-1).float().to(self.device)
        else:
            src_label = label_list[0].view(-1, 1)
            tgt_label = torch.cat([label_list[i] for i in range(1, len(split_sections))], dim=0)
            labels = torch.eq(src_label, tgt_label.T).float().to(self.device)  # (B, B*(neg_num+1)))

        bce_loss = self.criterion(cos_sims_for_bce, labels)

        # compute log_prob
        exp_logits = torch.exp(cos_sims_for_cl)
        log_prob = cos_sims_for_cl - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # labels = labels * pos_mask
        mean_log_prob_pos = (labels * log_prob).sum(1) / (labels.sum(1) + 1e-10)
        loss = -mean_log_prob_pos.mean()

        loss = bce_loss * self.bce_ratio + loss
        return loss


class MetricLearning(nn.Module):
    def __init__(self,
                 use_cuda,
                 use_miner=False,
                 miner_margin=0.2,
                 type_of_triplets="all",
                 temperature=1.0,
                 broadcast=False,
                 cosent_alpha=20,
                 pos_margin=0.0,
                 neg_margin=0.5):

        LOGGER.info("MetricLearning! use_cuda={}".format(use_cuda))
        super(MetricLearning, self).__init__()
        self.use_cuda = use_cuda
        self.use_miner = use_miner
        self.miner_margin = miner_margin
        self.type_of_triplets = type_of_triplets
        self.temperature = temperature
        self.broadcast = broadcast
        self.cosent_alpha = cosent_alpha
        self.neg_margin = neg_margin
        self.pos_margin = pos_margin
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')

        if self.use_miner:
            self.miner = miners.TripletMarginMiner(margin=miner_margin, type_of_triplets=type_of_triplets)
        else:
            self.miner = None

        self.criterion = marginal_nll

        self.loss = NTXentLoss(device=self.device,
                               temperature=self.temperature,
                               broadcast=self.broadcast)

        print("miner:", self.miner)
        print("loss:", self.loss)

    def forward(self, embeddings, labels, split_sections=None):
        """
        query : (N, h), candidates : (N, topk, h)
        output : (N, topk)
        modal_id:
        """
        embeddings = F.normalize(embeddings)
        if self.use_miner:
            hard_pairs = self.miner(embeddings, labels)
            retrieval_loss = self.loss(embeddings, labels, hard_pairs)
        else:
            if isinstance(self.loss, NTXentLoss):
                retrieval_loss = self.loss(embeddings, labels, split_sections)
            else:
                retrieval_loss = self.loss(embeddings, labels)
        return retrieval_loss

    def reshape_candidates_for_encoder(self, candidates):
        """
        reshape candidates for encoder input shape
        [batch_size, topk, max_length] => [batch_size*topk, max_length]
        """
        _, _, max_length = candidates.shape
        candidates = candidates.contiguous().view(-1, max_length)
        return candidates

    def get_loss(self, outputs, targets):
        # if self.use_cuda:
        #     targets = targets.cuda()
        targets = targets.to(self.device)
        loss = self.criterion(outputs, targets, self.temperature)
        return loss

    def get_embeddings(self, mentions, batch_size=1024):
        """
        Compute all embeddings from mention tokens.
        """
        embedding_table = []
        with torch.no_grad():
            for start in tqdm(range(0, len(mentions), batch_size)):
                end = min(start + batch_size, len(mentions))
                batch = mentions[start:end]
                batch_embedding = self.vectorizer(batch)
                batch_embedding = batch_embedding.cpu()
                embedding_table.append(batch_embedding)
        embedding_table = torch.cat(embedding_table, dim=0)
        return embedding_table


def marginal_nll(score, target, temperature=1.0):
    """
    sum all scores among positive samples
    """
    predict = F.softmax(score / temperature, dim=-1)
    loss = predict * target
    loss = loss.sum(dim=-1)  # sum all positive scores
    loss = loss[loss > 0]  # filter sets with at least one positives
    loss = torch.clamp(loss, min=1e-9, max=1)  # for numerical stability
    loss = -torch.log(loss)  # for negative log likelihood
    if len(loss) == 0:
        loss = loss.sum()  # will return zero loss
    else:
        loss = loss.mean()
    return loss
