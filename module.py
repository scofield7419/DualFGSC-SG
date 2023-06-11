import random
import math
import time
from transformers.modeling_bert import BertPreTrainedModel, BertModel

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pyemd import emd_samples
from sklearn.metrics import f1_score
from utils import *
import torch
from .modeling_bart import BartEncoder, BartDecoder, BartModel
from transformers import BartTokenizer
from fastNLP import seq_len_to_mask
from fastNLP.modules import Seq2SeqEncoder, Seq2SeqDecoder, State
import torch.nn.functional as F
from fastNLP.models import Seq2SeqModel
from torch import nn
import math


class Criterion(nn.Module):
    def __init__(self, model, reward_type, loss_weight,aspect_num=5,
                 supervised=True, rl_lambda=1.0, rl_alpha=0.5,
                 pretrain_epochs=0, total_epochs=-1, anneal_type='none',
                 LM=None, training_set_label_samples=None):
        super(Criterion, self).__init__()
        self.model = model
        self.reward_type = reward_type
        self.supervised = supervised
        self.rl_lambda = rl_lambda
        self.rl_alpha = rl_alpha
        self.pretrain_epochs = pretrain_epochs
        self.epoch = 0
        self.total_epochs = total_epochs
        self.anneal_type = anneal_type
        if anneal_type == 'linear' and (total_epochs is None):
            raise ValueError("Please set total_epochs if you want to " \
                             "use anneal_type='linear'")
        if anneal_type == 'switch' and pretrain_epochs == 0:
            raise ValueError("Please set pretrain_epochs > 0 if you want to " \
                             "use anneal_type='switch'")
        self.LM = LM
        self.aspect_num = aspect_num
        self.BCE = nn.BCEWithLogitsLoss(reduction='none')
        self.CE = nn.CrossEntropyLoss(weight=loss_weight, reduction='none')
        if 'em' in reward_type:
            samples = sum(training_set_label_samples, [])
            np.random.shuffle(samples)
            n = 10
            size = len(samples) // n
            self.samples = [
                samples[i*size:(i+1)*size]
                for i in range(n)
            ]

    def set_scorer(self, scorer):
        self.scorer = scorer

    def epoch_end(self):
        self.epoch += 1
        if self.anneal_type != 'none' and self.epoch == self.pretrain_epochs:
            print_time_info("loss scheduling started ({})".format(self.anneal_type))

    def earth_mover(self, decisions):
        # decisions.size() == (batch_size, sample_size, attr_vocab_size)
        length = decisions.size(-1)
        indexes = (decisions.float().numpy() >= 0.5)
        emd = [
            [
                emd_samples(
                    np.arange(length)[index].tolist(),
                    self.samples[0]
                ) if index.sum() > 0 else 1.0
                for index in indexes[bid]
            ]
            for bid in range(decisions.size(0))
        ]
        return torch.tensor(emd, dtype=torch.float, device=decisions.device)

    def get_scheduled_loss(self, sup_loss, rl_loss):
        if self.epoch < self.pretrain_epochs:
            return sup_loss, 0
        elif self.anneal_type == 'none':
            return sup_loss, rl_loss
        elif self.anneal_type == 'switch':
            return 0, rl_loss

        assert self.anneal_type == 'linear'
        rl_weight = (self.epoch - self.pretrain_epochs + 1) / (self.total_epochs - self.pretrain_epochs + 1)
        return (1-rl_weight) * sup_loss, rl_weight * rl_loss

    def get_scores(self, name, logits):
        size = logits.size(0)
        ret = torch.tensor(getattr(self.scorer, name)[-size:]).float()
        if len(ret.size()) == 2:
            ret = ret.mean(dim=-1)
        return ret

    def get_log_joint_prob_nlg(self, logits, decisions):
        """
        args:
            logits: tensor of shape [batch_size, beam_size, seq_length, vocab_size]
            decisions: tensor of shape [batch_size, beam_size, seq_length, vocab_size]
                       one-hot vector of decoded word-ids
        returns:
            log_joint_prob: tensor of shape [batch_size, beam_size]
        """
        logits = logits.contiguous().view(*decisions.size())
        probs = torch.softmax(logits, dim=-1)
        return (decisions * probs).sum(dim=-1).log().sum(dim=-1)

    def get_log_joint_prob_nlu(self, logits, decisions):
        """
        args:
            logits: tensor of shape [batch_size, attr_vocab_size]
                    or [batch_size, sample_size, attr_vocab_size]
            decisions: tensor of shape [batch_size, sample_size, attr_vocab_size]
                       decisions(0/1)
        returns:
            log_joint_prob: tensor of shape [batch_size, sample_size]
        """
        if len(logits.size()) == len(decisions.size()) - 1:
            logits = logits.unsqueeze(1).expand(-1, decisions.size(1), -1)

        probs = torch.sigmoid(logits)
        decisions = decisions.float()
        probs = probs * decisions + (1-probs) * (1-decisions)
        return probs.log().sum(dim=-1)

    def lm_log_prob(self, decisions):
        # decisions.size() == (batch_size, beam_size, seq_length, vocab_size)
        log_probs = [
            self.LM.get_log_prob(decisions[:, i])
            for i in range(decisions.size(1))
        ]
        return torch.stack(log_probs, dim=0).transpose(0, 1)

    def sentiment_log_prob(self, decisions):
        log_probs = [
            1 / self.aspect_num
            for i in range(decisions.size(1))
        ]
        return torch.stack(log_probs, dim=0).transpose(0, 1)

    def nlg_loss(self, logits, targets):
        bs = targets.size(0)
        loss = [
            self.CE(logits[:, i].contiguous().view(-1, logits.size(-1)), targets.view(-1)).view(bs, -1).mean(-1)
            for i in range(logits.size(1))
        ]
        return torch.stack(loss, dim=0).transpose(0, 1)

    def nlg_score(self, decisions, targets, func):
        scores = [
            func(targets, np.argmax(decisions.detach().cpu().numpy()[:, i], axis=-1))
            for i in range(decisions.size(1))
        ]
        scores = torch.tensor(scores, dtype=torch.float, device=decisions.device).transpose(0, 1)
        if len(scores.size()) == 3:
            scores = scores.mean(-1)

        return scores

    def nlu_loss(self, logits, targets):
        loss = [
            self.BCE(logits[:, i], targets).mean(-1)
            for i in range(logits.size(1))
        ]
        return torch.stack(loss, dim=0).transpose(0, 1)

    def nlu_score(self, decisions, targets, average):
        device = decisions.device
        decisions = decisions.detach().cpu().long().numpy()
        targets = targets.detach().cpu().long().numpy()
        scores = [
            [
                f1_score(y_true=np.array([label]), y_pred=np.array([pred]), average=average)
                for label, pred in zip(targets, decisions[:, i])
            ]
            for i in range(decisions.shape[1])
        ]
        return torch.tensor(scores, dtype=torch.float, device=device).transpose(0, 1)

    def get_reward(self, logits, targets, decisions=None):
        reward = 0
        if decisions is not None:
            decisions = decisions.detach()

        if self.model == "nlu":
            if self.reward_type == "loss":
                reward = self.nlu_loss(logits, targets)
            elif self.reward_type == "micro-f1":
                reward = -self.nlu_score(decisions, targets, 'micro')
            elif self.reward_type == "weighted-f1":
                reward = -self.nlu_score(decisions, targets, 'weighted')
            elif self.reward_type == "f1":
                reward = -(self.nlu_score(decisions, targets, 'micro') + self.nlu_score(decisions, targets, 'weighted'))
            elif self.reward_type == "em":
                reward = self.earth_mover(decisions)
            elif self.reward_type == "made":
                reward = -self.sentiment_log_prob(decisions)
            elif self.reward_type == "loss-em":
                reward = self.nlu_loss(logits, targets) + self.earth_mover(decisions)
        elif self.model == "nlg":
            if self.reward_type == "loss":
                reward = self.nlg_loss(logits, targets)
            elif self.reward_type == "lm":
                reward = -self.lm_log_prob(decisions)
            elif self.reward_type == "bleu":
                reward = -self.nlg_score(decisions, targets, func=single_BLEU)
            elif self.reward_type == "rouge":
                reward = -self.nlg_score(decisions, targets, func=single_ROUGE)
            elif self.reward_type == "bleu-rouge":
                reward = -(self.nlg_score(decisions, targets, func=single_BLEU) + self.nlg_score(decisions, targets, func=single_ROUGE))
            elif self.reward_type == "loss-lm":
                reward = self.nlg_loss(logits, targets) - self.lm_log_prob(decisions)

        return reward

    def forward(self, logits, targets, decisions=None, n_supervise=1,
                log_joint_prob=None, supervised=True, last_reward=0.0, calculate_reward=True):
        """
        args:
            logits: tensor of shape [batch_size, sample_size, * ]
            targets: tensor of shape [batch_size, *]
            decisions: tensor of shape [batch_size, sample_size, *]
        """
        if not self.supervised:
            supervised = False

        logits = logits.contiguous()
        targets = targets.contiguous()

        sup_loss = rl_loss = 0
        reward = 0.0
        if self.epoch >= self.pretrain_epochs and calculate_reward:
            reward = self.rl_lambda * self.get_reward(logits, targets, decisions)
        if isinstance(last_reward, torch.Tensor):
            reward = self.rl_alpha * last_reward + (1 - self.rl_alpha) * reward

        if self.model == "FGSC":
            if supervised:
                splits = logits.split(split_size=1, dim=1)
                for i in range(n_supervise):
                    sup_loss += self.BCE(splits[i].squeeze(1), targets).mean()
            X = self.get_log_joint_prob_nlu(logits, decisions) if log_joint_prob is None else log_joint_prob
        elif self.model == "FGSG":
            if supervised:
                splits = logits.split(split_size=1, dim=1)
                for i in range(n_supervise):
                    sup_loss += self.CE(splits[i].contiguous().view(-1, logits.size(-1)), targets.view(-1)).mean()
            X = self.get_log_joint_prob_nlg(logits, decisions) if log_joint_prob is None else log_joint_prob

        if isinstance(reward, torch.Tensor):
            rl_loss = (reward * X).mean()

        sup_loss, rl_loss = self.get_scheduled_loss(sup_loss, rl_loss)

        return sup_loss, rl_loss, X, reward


class RNNModel(nn.Module):
    def __init__(self,
                 dim_embedding,
                 dim_hidden,
                 attr_vocab_size,
                 vocab_size,
                 n_layers=1,
                 bidirectional=False):
        super(RNNModel, self).__init__()
        if attr_vocab_size and attr_vocab_size > dim_hidden:
            raise ValueError(
                "attr_vocab_size ({}) should be no larger than "
                "dim_hidden ({})".format(attr_vocab_size, dim_hidden)
            )
        self.dim_embedding = dim_embedding
        self.dim_hidden = dim_hidden
        self.attr_vocab_size = attr_vocab_size
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.n_directions = 2 if bidirectional else 1

        self.embedding = nn.Embedding(vocab_size, dim_embedding)
        self.rnn = nn.GRU(dim_embedding,
                          dim_hidden,
                          num_layers=n_layers,
                          batch_first=True,
                          bidirectional=bidirectional)

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def _init_hidden(self, inputs):
        """
        args:
            inputs: shape [batch_size, *]
                    a input tensor with correct device

        returns:
            hidden: shpae [n_layers*n_directions, batch_size, dim_hidden]
                    all-zero hidden state
        """
        batch_size = inputs.size(0)
        return torch.zeros(self.n_layers*self.n_directions,
                           batch_size,
                           self.dim_hidden,
                           dtype=torch.float,
                           device=inputs.device)

    def _init_hidden_with_attrs(self, attrs):
        """
        args:
            attrs: shape [batch_size, attr_vocab_size], a n-hot vector

        returns:
            hidden: shape [n_layers*n_directions, batch_size, dim_hidden]
        """
        batch_size = attrs.size(0)
        hidden = torch.cat(
            [
                attrs,
                torch.zeros(batch_size,
                            self.dim_hidden - self.attr_vocab_size,
                            dtype=attrs.dtype,
                            device=attrs.device)
            ], 1)
        '''
        # ignore _UNK and _PAD
        hidden[:, 0:2] = 0
        '''
        return hidden.unsqueeze(0).expand(self.n_layers*self.n_directions, -1, -1).float()


class NLGRNN(RNNModel):
    def __init__(self, *args, **kwargs):
        super(NLGRNN, self).__init__(*args, **kwargs)
        if self.n_directions != 1:
            raise ValueError("RNN must be uni-directional in NLG model.")

        self.transform = nn.Linear(self.attr_vocab_size, self.dim_hidden)
        self.linear = nn.Linear(self.dim_hidden, self.vocab_size)

    def _st_softmax(self, logits, hard=False, dim=-1):
        y_soft = logits.softmax(dim)

        if hard:
            # Straight through.
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            ret = y_soft

        return ret

    def _st_onehot(self, logits, indices, hard=True, dim=-1):
        y_soft = logits.softmax(dim)
        if isinstance(indices, np.ndarray):
            indices = torch.from_numpy(indices).long().to(logits.device)
        if len(logits.size()) == len(indices.size()) + 1:
            indices = indices.unsqueeze(-1)
        y_hard = torch.zeros_like(logits).scatter_(dim, indices, 1.0)
        if hard:
            return y_hard - y_soft.detach() + y_soft, y_hard
        else:
            return y_soft, y_hard

    def forward(self, attrs, bos_id, labels=None,
                tf_ratio=0.5, max_decode_length=50, beam_size=5, st=True):
        """
        args:
            attrs: shape [batch_size, attr_vocab_size]
            bos_id: integer
            labels: shape [batch_size, seq_length]

        outputs:
            logits: shape [batch_size, beam_size, seq_length, vocab_size]
            outputs: shape [batch_size, beam_size, seq_length, vocab_size]
                     output words as one-hot vectors (maybe soft)
            decisions: shape [batch_size, beam_size, seq_length, vocab_size]
                       output words as one-hot vectors (hard)
        """
        if beam_size == 1:
            logits, outputs = self.forward_greedy(
                attrs, bos_id, labels,
                tf_ratio=tf_ratio, max_decode_length=max_decode_length,
                st=st
            )
            return logits.unsqueeze(1), outputs.unsqueeze(1), outputs.unsqueeze(1)

        decode_length = max_decode_length if labels is None else labels.size(1)

        batch_size = attrs.size(0)
        # hidden.size() should be (n_layers*n_directions, beam_size*batch_size, dim_hidden)
        hiddens = self.transform(attrs.float()).unsqueeze(0).unsqueeze(0)
        hiddens = hiddens.expand(self.n_layers*self.n_directions, beam_size, -1, -1)
        hiddens = hiddens.contiguous().view(-1, beam_size*batch_size, self.dim_hidden)
        last_output = torch.full_like(attrs[:, 0], bos_id, dtype=torch.long)
        # last_output.size() == (beam_size, batch_size)
        last_output = [last_output for _ in range(beam_size)]
        # logits.shape will be [seq_length, beam_size, batch_size, vocab_size]
        logits = []
        beam_probs = np.full((beam_size, batch_size), -math.inf)
        beam_probs[0, :] = 0.0
        # last_indices.shape will be [seq_length, batch_size, beam_size]
        last_indices = []
        output_ids = []
        for step in range(decode_length):
            curr_inputs = []
            for beam in range(beam_size):
                use_tf = False if step == 0 else random.random() < tf_ratio
                if use_tf:
                    curr_input = labels[:, step-1]
                else:
                    curr_input = last_output[beam].detach()

                if len(curr_input.size()) == 1:
                    # curr_input are ids
                    curr_input = self.embedding(curr_input).unsqueeze(1)
                else:
                    # curr_input are one-hot vectors
                    curr_input = torch.matmul(curr_input.float(), self.embedding.weight).unsqueeze(1)
                curr_inputs.append(curr_input)

            curr_inputs = torch.stack(curr_inputs, dim=0)
            # curr_inputs.size() == (beam_size, batch_size, 1, dim_embedding)
            curr_inputs = curr_inputs.view(-1, 1, self.dim_embedding)
            output, new_hiddens = self.rnn(curr_inputs, hiddens)
            output = self.linear(output.squeeze(1))
            output = output.view(beam_size, batch_size, -1)
            new_hiddens = new_hiddens.view(self.n_layers*self.n_directions, beam_size, batch_size, -1)
            probs = torch.log_softmax(output.detach(), dim=-1)
            # top_probs.size() == top_indices.size() == (beam_size, batch_size, k)
            top_probs, top_indices = torch.topk(probs, k=beam_size, dim=-1)
            top_probs = top_probs.detach().cpu().numpy()
            top_indices = top_indices.detach().cpu().numpy()
            last_index = []
            output_id = []
            for bid in range(batch_size):
                beam_prob = []
                for beam in range(beam_size):
                    beam_prob.extend([
                        (
                            beam,
                            top_indices[beam, bid, i],
                            beam_probs[beam][bid] + top_probs[beam, bid, i]
                        )
                        for i in range(beam_size)
                    ])
                topk = sorted(beam_prob, key=lambda x: x[2], reverse=True)[:beam_size]
                last_index.append([item[0] for item in topk])
                output_id.append([item[1] for item in topk])
                beam_probs[:, bid] = np.array([item[2] for item in topk])

            last_indices.append(last_index)
            output_ids.append(output_id)

            new_hiddens = new_hiddens.permute([2, 0, 1, 3]).split(split_size=1, dim=0)
            hiddens = torch.stack([
                new_hiddens[bid].squeeze(0).index_select(dim=1, index=torch.tensor(indices).to(new_hiddens[bid].device))
                for bid, indices in enumerate(last_index)
            ], dim=0).permute([1, 2, 0, 3]).contiguous().view(-1, beam_size*batch_size, self.dim_hidden)

            output = output.transpose(0, 1).split(split_size=1, dim=0)
            output = [
                output[bid].squeeze(0).index_select(dim=0, index=torch.tensor(indices).to(output[bid].device))
                for bid, indices in enumerate(last_index)
            ]
            logits.append(output)

            last_output = [
                torch.tensor(
                    [output_id[bid][beam] for bid in range(batch_size)],
                    dtype=torch.long, device=attrs.device
                )
                for beam in range(beam_size)
            ]

        last_indices = np.array(last_indices)
        output_ids = np.array(output_ids)
        # back-trace the beams to get outputs
        beam_outputs = []
        beam_logits = []
        beam_decisions = []
        for bid in range(batch_size):
            this_index = np.arange(beam_size)
            step_logits = []
            step_output_ids = []
            for step in range(decode_length-1, -1, -1):
                this_logits = logits[step][bid].index_select(dim=0, index=torch.from_numpy(this_index).to(logits[step][bid].device))
                step_logits.append(this_logits)
                step_output_ids.append(output_ids[step, bid, this_index])
                this_index = last_indices[step, bid, this_index]

            step_logits = torch.stack(step_logits[::-1], dim=0)
            step_outputs, step_decisions = self._st_onehot(step_logits, np.array(step_output_ids[::-1]), hard=st)
            beam_outputs.append(step_outputs)
            beam_logits.append(step_logits)
            beam_decisions.append(step_decisions)

        logits = torch.stack(beam_logits).transpose(1, 2)
        outputs = torch.stack(beam_outputs).transpose(1, 2)
        decisions = torch.stack(beam_decisions).transpose(1, 2)
        return logits, outputs, decisions

    def forward_greedy(self, attrs, bos_id, labels=None, sampling=False,
                       tf_ratio=0.5, max_decode_length=50, st=True):
        """
        args:
            attrs: shape [batch_size, attr_vocab_size]
            bos_id: integer
            labels: shape [batch_size, seq_length]

        outputs:
            logits: shape [batch_size, seq_length, vocab_size]
            outputs: shape [batch_size, seq_length, vocab_size]
                     output words as one-hot vectors
        """
        decode_length = max_decode_length if labels is None else labels.size(1)

        hidden = self.transform(attrs.float()).unsqueeze(0)
        hidden = hidden.expand(self.n_layers*self.n_directions, -1, -1).contiguous()
        last_output = torch.full_like(attrs[:, 0], bos_id, dtype=torch.long)
        logits = []
        outputs = []
        for step in range(decode_length):
            use_tf = False if step == 0 else random.random() < tf_ratio
            if use_tf:
                curr_input = labels[:, step-1]
            else:
                curr_input = last_output.detach()

            if len(curr_input.size()) == 1:
                curr_input = self.embedding(curr_input).unsqueeze(1)
            else:
                curr_input = torch.matmul(curr_input.float(), self.embedding.weight).unsqueeze(1)

            output, hidden = self.rnn(curr_input, hidden)
            output = self.linear(output.squeeze(1))
            logits.append(output)
            if sampling:
                last_output = F.gumbel_softmax(output, hard=True)
            else:
                last_output = self._st_softmax(output, hard=True, dim=-1)
            outputs.append(self._st_softmax(output, hard=st, dim=-1))

        logits = torch.stack(logits).transpose(0, 1)
        outputs = torch.stack(outputs).transpose(0, 1)
        return logits, outputs


class NLURNN(RNNModel):
    def __init__(self, *args, **kwargs):
        super(NLURNN, self).__init__(*args, **kwargs)
        self.linear = nn.Linear(
            self.n_layers * self.n_directions * self.dim_hidden,
            self.attr_vocab_size
        )

    def forward(self, inputs, sample_size=1):
        """
        args:
            inputs: shape [batch_size, seq_length]
                    or shape [batch_size, seq_length, attr_vocab_size] one-hot vectors

        outputs:
            logits: shape [batch_size, attr_vocab_size]
        """
        batch_size = inputs.size(0)
        if len(inputs.size()) == 2:
            inputs = self.embedding(inputs)
        else:
            # suppose the inputs are one-hot vectors
            inputs = torch.matmul(inputs.float(), self.embedding.weight)

        _, hidden = self.rnn(inputs)
        hidden = hidden.transpose(0, 1).contiguous().view(batch_size, -1)
        logits = self.linear(hidden)
        return logits


class LMRNN(RNNModel):
    def __init__(self, *args, **kwargs):
        super(LMRNN, self).__init__(*args, **kwargs)
        if self.n_directions != 1:
            raise ValueError("RNN must be uni-directional in LM model.")
        self.linear = nn.Linear(self.dim_hidden, self.vocab_size)

    def forward(self, inputs):
        inputs = self.embedding(inputs)
        output, _ = self.rnn(inputs)
        logits = self.linear(output)
        return logits


class MaskedLinear(nn.Linear):
    """ same as Linear except has a configurable mask on the weights """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))

    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))

    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, q, k, v, attn_mask):
        attn_score = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.d_k)
        attn_score.masked_fill_(attn_mask, -1e9)

        attn_weights = nn.Softmax(dim=-1)(attn_score)

        output = torch.matmul(attn_weights, v)

        return output, attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_k = self.d_v = d_model // n_heads

        self.WQ = nn.Linear(d_model, d_model)
        self.WK = nn.Linear(d_model, d_model)
        self.WV = nn.Linear(d_model, d_model)
        self.scaled_dot_product_attn = ScaledDotProductAttention(self.d_k)
        self.linear = nn.Linear(n_heads * self.d_v, d_model)

    def forward(self, Q, K, V, attn_mask):
        batch_size = Q.size(0)

        q_heads = self.WQ(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_heads = self.WK(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_heads = self.WV(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        attn, attn_weights = self.scaled_dot_product_attn(q_heads, k_heads, v_heads, attn_mask)

        attn = attn.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        output = self.linear(attn)

        return output, attn_weights


class PositionWiseFeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForwardNetwork, self).__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        output = self.relu(self.linear1(inputs))
        output = self.linear2(output)

        return output


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, p_drop, d_ff):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, n_heads)
        self.dropout1 = nn.Dropout(p_drop)
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)

        self.ffn = PositionWiseFeedForwardNetwork(d_model, d_ff)
        self.dropout2 = nn.Dropout(p_drop)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, inputs, attn_mask):
        attn_outputs, attn_weights = self.mha(inputs, inputs, inputs, attn_mask)
        attn_outputs = self.dropout1(attn_outputs)
        attn_outputs = self.layernorm1(inputs + attn_outputs)

        ffn_outputs = self.ffn(attn_outputs)
        ffn_outputs = self.dropout2(ffn_outputs)
        ffn_outputs = self.layernorm2(attn_outputs + ffn_outputs)

        return ffn_outputs, attn_weights


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, seq_len=300, d_model=768, n_layers=12, n_heads=8, p_drop=0.1, d_ff=500, pad_id=0):
        super(TransformerEncoder, self).__init__()
        self.pad_id = pad_id
        self.sinusoid_table = self.get_sinusoid_table(seq_len + 1, d_model)  # (seq_len+1, d_model)

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding.from_pretrained(self.sinusoid_table, freeze=True)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, p_drop, d_ff) for _ in range(n_layers)])

    def forward(self, inputs):
        positions = torch.arange(inputs.size(1), device=inputs.device, dtype=inputs.dtype).repeat(inputs.size(0), 1) + 1
        position_pad_mask = inputs.eq(self.pad_id)
        positions.masked_fill_(position_pad_mask, 0)

        outputs = self.embedding(inputs) + self.pos_embedding(positions)

        attn_pad_mask = self.get_attention_padding_mask(inputs, inputs, self.pad_id)

        for layer in self.layers:
            outputs, attn_weights = layer(outputs, attn_pad_mask)

        return outputs

    def get_attention_padding_mask(self, q, k, pad_id):
        attn_pad_mask = k.eq(pad_id).unsqueeze(1).repeat(1, q.size(1), 1)

        return attn_pad_mask

    def get_sinusoid_table(self, seq_len, d_model):
        def get_angle(pos, i, d_model):
            return pos / np.power(10000, (2 * (i // 2)) / d_model)

        sinusoid_table = np.zeros((seq_len, d_model))
        for pos in range(seq_len):
            for i in range(d_model):
                if i % 2 == 0:
                    sinusoid_table[pos, i] = np.sin(get_angle(pos, i, d_model))
                else:
                    sinusoid_table[pos, i] = np.cos(get_angle(pos, i, d_model))

        return torch.FloatTensor(sinusoid_table)


class LMTrm(TransformerEncoder):
    def __init__(self, *args, **kwargs):
        super(LMTrm, self).__init__(*args, **kwargs)
        if self.n_directions != 1:
            raise ValueError("RNN must be uni-directional in LM model.")
        self.linear = nn.Linear(self.dim_hidden, self.vocab_size)

    def forward(self, inputs):
        """
        args:
            inputs: shape [batch_size, seq_length]

        outputs:
            logits: shape [batch_size, seq_length, vocab_size]
        """
        positions = torch.arange(inputs.size(1), device=inputs.device, dtype=inputs.dtype).repeat(inputs.size(0), 1) + 1
        position_pad_mask = inputs.eq(self.pad_id)
        positions.masked_fill_(position_pad_mask, 0)

        outputs = self.embedding(inputs) + self.pos_embedding(positions)

        attn_pad_mask = self.get_attention_padding_mask(inputs, inputs, self.pad_id)

        for layer in self.layers:
            outputs, attn_weights = layer(outputs, attn_pad_mask)

        logits = self.linear(outputs)

        return logits


class OrderedCounter(Counter, OrderedDict):
    """Counter that remembers the order elements are first encountered"""
    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x


def idx2word(idx, i2w, pad_idx):
    sent_str = [str()]*len(idx)
    for i, sent in enumerate(idx):
        for word_id in sent:
            if word_id == pad_idx:
                break
            sent_str[i] += i2w[str(word_id.item())] + " "
        sent_str[i] = sent_str[i].strip()
    return sent_str


def interpolate(start, end, steps):

    interpolation = np.zeros((start.shape[0], steps + 2))

    for dim, (s, e) in enumerate(zip(start, end)):
        interpolation[dim] = np.linspace(s, e, steps+2)

    return interpolation.T


def expierment_name(args, ts):
    exp_name = str()
    exp_name += "BS=%i_" % args.batch_size
    exp_name += "LR={}_".format(args.learning_rate)
    exp_name += "EB=%i_" % args.embedding_size
    exp_name += "%s_" % args.rnn_type.upper()
    exp_name += "HS=%i_" % args.hidden_size
    exp_name += "L=%i_" % args.num_layers
    exp_name += "BI=%i_" % args.bidirectional
    exp_name += "LS=%i_" % args.latent_size
    exp_name += "WD={}_".format(args.word_dropout)
    exp_name += "ANN=%s_" % args.anneal_function.upper()
    exp_name += "K={}_".format(args.k)
    exp_name += "X0=%i_" % args.x0
    exp_name += "TS=%s" % ts

    return exp_name


def cosine_similarity(x, y, norm=False):
    assert len(x) == len(y), "len(x) != len(y)"
    zero_list = [0] * len(x)
    if x == zero_list or y == zero_list:
        return float(1) if x == y else float(0)

    res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))

    return 0.5 * cos + 0.5 if norm else cos


class Variables(nn.Module):
    def __init__(self, hidden_size, output_size, latent_size, latent_num, explicit_size, explicit_num):

        super(Variables).__init__()

        self.latent_size = latent_size
        self.explicit_size = explicit_size
        self.latent_num = latent_num
        self.explicit_num = explicit_num

        self.attention = MultiHeadAttention(hidden_size,1)
        self.hidden_size = hidden_size

        self.hidden2mean = nn.Linear(hidden_size , latent_size)
        self.hidden2logv = nn.Linear(hidden_size , latent_size)
        self.latent2hidden = nn.Linear(latent_size, hidden_size )
        self.outputs = nn.Linear(hidden_size , output_size)

    def forward_cross_referring(self, vars_src, vars_tgt):
        sim = cosine_similarity(vars_src, vars_tgt)
        return 1/nn.exp(sim)

    def forward(self, hidden):
        batch_size = hidden.size(0)
        hidden = hidden.squeeze()

        expvars = []
        for lat_k in range(self.latent_num):
            expvars.append(self.attention(hidden))

        latvars = []
        for lat_k in range(self.latent_num):
            mean = self.hidden2mean(hidden)
            logv = self.hidden2logv(hidden)
            std = torch.exp(0.5 * logv)

            z = to_var(torch.randn([batch_size, self.latent_size]))
            z = z * std + mean

            hidden = self.latent2hidden(z)
            hidden = hidden.unsqueeze(0)

            if self.word_dropout_rate > 0:
                prob = torch.rand(hidden.size())
                if torch.cuda.is_available():
                    prob=prob.cuda()
                prob[(hidden.data - self.sos_idx) * (hidden.data - self.pad_idx) == 0] = 1
                decoder_input_sequence = hidden.clone()
                decoder_input_sequence[prob < self.word_dropout_rate] = self.unk_idx

            # process outputs
            padded_outputs = rnn_utils.pad_packed_sequence(hidden, batch_first=True)[0]
            padded_outputs = padded_outputs.contiguous()
            _,reversed_idx = torch.sort(sorted_idx)
            padded_outputs = padded_outputs[reversed_idx]
            b,s,_ = padded_outputs.size()

            logp = nn.functional.log_softmax(self.outputs2vocab(padded_outputs.view(-1, padded_outputs.size(2))), dim=-1)
            logp = logp.view(b, s, self.embedding.num_embeddings)
            lats.append((logp, mean, logv, z))

        return expvars, latvars

    def var_inference(self, n, z=None):

        if z is None:
            batch_size = n
            z = to_var(torch.randn([batch_size, self.latent_size]))
        else:
            batch_size = z.size(0)

        hidden = self.latent2hidden(z)

        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)

        hidden = hidden.unsqueeze(0)

        sequence_idx = torch.arange(0, batch_size, out=self.tensor()).long()  # all idx of batch
        sequence_running = torch.arange(0, batch_size, out=self.tensor()).long()
        sequence_mask = torch.ones(batch_size, out=self.tensor()).bool()
        running_seqs = torch.arange(0, batch_size, out=self.tensor()).long()

        generations = self.tensor(batch_size, self.max_sequence_length).fill_(self.pad_idx).long()

        t = 0
        while t < self.max_sequence_length and len(running_seqs) > 0:

            if t == 0:
                input_sequence = to_var(torch.Tensor(batch_size).fill_(self.sos_idx).long())

            input_sequence = input_sequence.unsqueeze(1)

            input_embedding = self.embedding(input_sequence)

            output, hidden = self.decoder_rnn(input_embedding, hidden)

            logits = self.outputs2vocab(output)

            input_sequence = self._sample(logits)

            generations = self._save_sample(generations, input_sequence, sequence_running, t)

            sequence_mask[sequence_running] = (input_sequence != self.eos_idx)
            sequence_running = sequence_idx.masked_select(sequence_mask)

            running_mask = (input_sequence != self.eos_idx).data
            running_seqs = running_seqs.masked_select(running_mask)

            if len(running_seqs) > 0:
                input_sequence = input_sequence[running_seqs]
                hidden = hidden[:, running_seqs]

                running_seqs = torch.arange(0, len(running_seqs), out=self.tensor()).long()

            t += 1

        return generations, z

    def _sample(self, dist, mode='greedy'):

        if mode == 'greedy':
            _, sample = torch.topk(dist, 1, dim=-1)
        sample = sample.reshape(-1)

        return sample

    def _save_sample(self, save_to, sample, running_seqs, t):
        running_latest = save_to[running_seqs]
        running_latest[:,t] = sample.data
        save_to[running_seqs] = running_latest

        return save_to


class SCModel(BertPreTrainedModel):
    # BertForMultiLable
    def __init__(self, arg):
        super(SCModel, self).__init__(arg)
        self.bert = BertModel(arg)
        self.dropout = nn.Dropout(arg.hidden_dropout_prob)
        self.classifier = nn.Linear(arg.hidden_size, arg.num_labels)

        self.var_num = arg.num_labels + 1
        self.vars = [Variables(arg.hidden_size, arg.output_size, arg.latent_size, 1, arg.explicit_size, arg.num_labels) for _ in range(self.var_num)]

        self.init_weights()

    def _st_softmax(self, logits, hard=False, dim=-1):
        y_soft = logits.softmax(dim)

        if hard:
            # Straight through.
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            ret = y_soft

        return ret

    def _st_onehot(self, logits, indices, hard=True, dim=-1):
        y_soft = logits.softmax(dim)
        if isinstance(indices, np.ndarray):
            indices = torch.from_numpy(indices).long().to(logits.device)
        if len(logits.size()) == len(indices.size()) + 1:
            indices = indices.unsqueeze(-1)
        y_hard = torch.zeros_like(logits).scatter_(dim, indices, 1.0)
        if hard:
            return y_hard - y_soft.detach() + y_soft, y_hard
        else:
            return y_soft, y_hard

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,head_mask=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids,attention_mask=attention_mask, head_mask=head_mask)
        posterio_hiddens = []
        for var_k in range(self.var_num):
            expvars, latvars = self.vars[var_k](outputs)
            posterio_hiddens.append(latvars)
        pooled_output = self.dropout(posterio_hiddens)
        logits = self.classifier(pooled_output)
        return logits


class FBartEncoder(Seq2SeqEncoder):
    def __init__(self, encoder):
        super().__init__()
        assert isinstance(encoder, BartEncoder)
        self.bart_encoder = encoder

    def forward(self, src_tokens, src_seq_len):
        mask = seq_len_to_mask(src_seq_len, max_len=src_tokens.size(1))
        dict = self.bart_encoder(input_ids=src_tokens, attention_mask=mask, return_dict=True,
                                 output_hidden_states=True)
        encoder_outputs = dict.last_hidden_state
        hidden_states = dict.hidden_states
        return encoder_outputs, mask, hidden_states


class FBartDecoder(Seq2SeqDecoder):
    def __init__(self, decoder, pad_token_id, label_ids, use_encoder_mlp=True):
        super().__init__()
        assert isinstance(decoder, BartDecoder)
        self.decoder = decoder
        causal_mask = torch.zeros(512, 512).fill_(float('-inf'))
        causal_mask = causal_mask.triu(diagonal=1)
        self.register_buffer('causal_masks', causal_mask.float())
        self.pad_token_id = pad_token_id
        self.label_start_id = label_ids[0]
        self.label_end_id = label_ids[-1]+1

        mapping = torch.LongTensor([0, 2]+sorted(label_ids, reverse=False))
        self.register_buffer('mapping', mapping)
        self.src_start_index = len(mapping)
        hidden_size = decoder.embed_tokens.weight.size(1)
        if use_encoder_mlp:
            self.encoder_mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                             nn.Dropout(0.3),
                                             nn.ReLU(),
                                             nn.Linear(hidden_size, hidden_size))

    def forward(self, inputs, attrs=None, bos_id=None, labels=None,
        tf_ratio=0.5, max_decode_length=50, beam_size=5, st=True):

        if beam_size == 1:
            logits, outputs = self.forward_greedy(
                attrs, bos_id, labels,
                tf_ratio=tf_ratio, max_decode_length=max_decode_length,
                st=st
            )
            return logits.unsqueeze(1), outputs.unsqueeze(1), outputs.unsqueeze(1)

        decode_length = max_decode_length if labels is None else labels.size(1)

        batch_size = attrs.size(0)
        hiddens = self.transform(attrs.float()).unsqueeze(0).unsqueeze(0)
        hiddens = hiddens.expand(self.n_layers * self.n_directions, beam_size, -1, -1)
        hiddens = hiddens.contiguous().view(-1, beam_size * batch_size, self.dim_hidden)
        last_output = torch.full_like(attrs[:, 0], bos_id, dtype=torch.long)
        # last_output.size() == (beam_size, batch_size)
        last_output = [last_output for _ in range(beam_size)]
        # logits.shape will be [seq_length, beam_size, batch_size, vocab_size]
        logits = []
        beam_probs = np.full((beam_size, batch_size), -math.inf)
        beam_probs[0, :] = 0.0
        # last_indices.shape will be [seq_length, batch_size, beam_size]
        last_indices = []
        output_ids = []
        for step in range(decode_length):
            curr_inputs = []
            for beam in range(beam_size):
                use_tf = False if step == 0 else random.random() < tf_ratio
                if use_tf:
                    curr_input = labels[:, step - 1]
                else:
                    curr_input = last_output[beam].detach()

                if len(curr_input.size()) == 1:
                    # curr_input are ids
                    curr_input = self.embedding(curr_input).unsqueeze(1)
                else:
                    # curr_input are one-hot vectors
                    curr_input = torch.matmul(curr_input.float(), self.embedding.weight).unsqueeze(1)
                curr_inputs.append(curr_input)

            curr_inputs = torch.stack(curr_inputs, dim=0)
            # curr_inputs.size() == (beam_size, batch_size, 1, dim_embedding)
            curr_inputs = curr_inputs.view(-1, 1, self.dim_embedding)
            output, new_hiddens = self.rnn(curr_inputs, hiddens)
            output = self.linear(output.squeeze(1))
            output = output.view(beam_size, batch_size, -1)
            new_hiddens = new_hiddens.view(self.n_layers * self.n_directions, beam_size, batch_size, -1)
            probs = torch.log_softmax(output.detach(), dim=-1)
            # top_probs.size() == top_indices.size() == (beam_size, batch_size, k)
            top_probs, top_indices = torch.topk(probs, k=beam_size, dim=-1)
            top_probs = top_probs.detach().cpu().numpy()
            top_indices = top_indices.detach().cpu().numpy()
            last_index = []
            output_id = []
            for bid in range(batch_size):
                beam_prob = []
                for beam in range(beam_size):
                    beam_prob.extend([
                        (
                            beam,
                            top_indices[beam, bid, i],
                            beam_probs[beam][bid] + top_probs[beam, bid, i]
                        )
                        for i in range(beam_size)
                    ])
                topk = sorted(beam_prob, key=lambda x: x[2], reverse=True)[:beam_size]
                last_index.append([item[0] for item in topk])
                output_id.append([item[1] for item in topk])
                beam_probs[:, bid] = np.array([item[2] for item in topk])

            last_indices.append(last_index)
            output_ids.append(output_id)

            new_hiddens = new_hiddens.permute([2, 0, 1, 3]).split(split_size=1, dim=0)
            hiddens = torch.stack([
                new_hiddens[bid].squeeze(0).index_select(dim=1, index=torch.tensor(indices).to(new_hiddens[bid].device))
                for bid, indices in enumerate(last_index)
            ], dim=0).permute([1, 2, 0, 3]).contiguous().view(-1, beam_size * batch_size, self.dim_hidden)

            output = output.transpose(0, 1).split(split_size=1, dim=0)
            output = [
                output[bid].squeeze(0).index_select(dim=0, index=torch.tensor(indices).to(output[bid].device))
                for bid, indices in enumerate(last_index)
            ]
            logits.append(output)

            last_output = [
                torch.tensor(
                    [output_id[bid][beam] for bid in range(batch_size)],
                    dtype=torch.long, device=attrs.device
                )
                for beam in range(beam_size)
            ]

        last_indices = np.array(last_indices)
        output_ids = np.array(output_ids)
        # back-trace the beams to get outputs
        beam_outputs = []
        beam_logits = []
        beam_decisions = []
        for bid in range(batch_size):
            this_index = np.arange(beam_size)
            step_logits = []
            step_output_ids = []
            for step in range(decode_length - 1, -1, -1):
                this_logits = logits[step][bid].index_select(dim=0, index=torch.from_numpy(this_index).to(
                    logits[step][bid].device))
                step_logits.append(this_logits)
                step_output_ids.append(output_ids[step, bid, this_index])
                this_index = last_indices[step, bid, this_index]

            step_logits = torch.stack(step_logits[::-1], dim=0)
            step_outputs, step_decisions = self._st_onehot(step_logits, np.array(step_output_ids[::-1]), hard=st)
            beam_outputs.append(step_outputs)
            beam_logits.append(step_logits)
            beam_decisions.append(step_decisions)

        logits = torch.stack(beam_logits).transpose(1, 2)
        outputs = torch.stack(beam_outputs).transpose(1, 2)
        decisions = torch.stack(beam_decisions).transpose(1, 2)
        return logits, outputs, decisions


class SGModel(Seq2SeqModel):
    # BartSeq2SeqModel
    def __init__(self, arg):
        super(SGModel, self).__init__(arg)
        self.bert = BertModel(arg)
        self.dropout = nn.Dropout(arg.hidden_dropout_prob)
        self.classifier = nn.Linear(arg.hidden_size, arg.num_labels)

        self.var_num = arg.num_labels + 1
        self.vars = [Variables(arg.hidden_size, arg.output_size, arg.latent_size, arg.num_labels, arg.explicit_size, 1) for _ in range(self.var_num)]

        self.init_weights()


    @classmethod
    def build_model(cls, bart_model, tokenizer, label_ids, decoder_type=None,
                    use_encoder_mlp=False, use_recur_pos=False, tag_first=False):
        model = BartModel.from_pretrained(bart_model)
        num_tokens, _ = model.encoder.embed_tokens.weight.shape
        model.resize_token_embeddings(len(tokenizer.unique_no_split_tokens)+num_tokens)
        encoder = model.encoder
        decoder = model.decoder

        if use_recur_pos:
            decoder.set_position_embedding(label_ids[0], tag_first)

        _tokenizer = BartTokenizer.from_pretrained(bart_model)
        for token in tokenizer.unique_no_split_tokens:
            if token[:2] == '<<':
                index = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))
                if len(index)>1:
                    raise RuntimeError(f"{token} wrong split")
                else:
                    index = index[0]
                assert index>=num_tokens, (index, num_tokens, token)
                indexes = _tokenizer.convert_tokens_to_ids(_tokenizer.tokenize(token[2:-2]))
                embed = model.encoder.embed_tokens.weight.data[indexes[0]]
                for i in indexes[1:]:
                    embed += model.decoder.embed_tokens.weight.data[i]
                embed /= len(indexes)
                model.decoder.embed_tokens.weight.data[index] = embed

        encoder = FBartEncoder(encoder)
        label_ids = sorted(label_ids)
        decoder = FBartDecoder(decoder, pad_token_id=tokenizer.pad_token_id, label_ids=label_ids)

        return cls(encoder=encoder, decoder=decoder)

    def prepare_state(self, src_tokens, src_seq_len=None, first=None, tgt_seq_len=None):
        encoder_outputs, encoder_mask, hidden_states = self.encoder(src_tokens, src_seq_len)
        src_embed_outputs = hidden_states[0]
        state = BartState(encoder_outputs, encoder_mask, src_tokens, first, src_embed_outputs)
        # setattr(state, 'tgt_seq_len', tgt_seq_len)
        return state

    def forward(self, src_tokens, tgt_tokens, src_seq_len, tgt_seq_len, first):

        state = self.prepare_state(src_tokens, src_seq_len, first, tgt_seq_len)
        posterio_hiddens = []
        for var_k in range(self.var_num):
            expvars, latvars = self.vars[var_k](state)
            posterio_hiddens.append(latvars)
        decoder_output = self.decoder(tgt_tokens, posterio_hiddens)
        if isinstance(decoder_output, torch.Tensor):
            return {'pred': decoder_output}
        elif isinstance(decoder_output, (tuple, list)):
            return {'pred': decoder_output[0]}
        else:
            raise TypeError(f"Unsupported return type from Decoder:{type(self.decoder)}")


class BartState(State):
    def __init__(self, encoder_output, encoder_mask, src_tokens, first, src_embed_outputs):
        super().__init__(encoder_output, encoder_mask)
        self.past_key_values = None
        self.src_tokens = src_tokens
        self.first = first
        self.src_embed_outputs = src_embed_outputs

    def reorder_state(self, indices: torch.LongTensor):
        super().reorder_state(indices)
        self.src_tokens = self._reorder_state(self.src_tokens, indices)
        if self.first is not None:
            self.first = self._reorder_state(self.first, indices)
        self.src_embed_outputs = self._reorder_state(self.src_embed_outputs, indices)
        if self.past_key_values is not None:
            new = []
            for layer in self.past_key_values:
                new_layer = {}
                for key1 in list(layer.keys()):
                    new_layer_ = {}
                    for key2 in list(layer[key1].keys()):
                        if layer[key1][key2] is not None:
                            layer[key1][key2] = self._reorder_state(layer[key1][key2], indices)
                            # print(key1, key2, layer[key1][key2].shape)
                        new_layer_[key2] = layer[key1][key2]
                    new_layer[key1] = new_layer_
                new.append(new_layer)
            self.past_key_values = new