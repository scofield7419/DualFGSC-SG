import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

import time
import random
import numpy as np
import os
import math

from module import SCModel, SGModel
# from utils import single_BLEU, BLEU, single_ROUGE, ROUGE, best_ROUGE, print_time_info, check_dir, print_curriculum_status
from utils import *
from text_token import _UNK, _PAD, _BOS, _EOS
from model_utils import collate_fn_sg, collate_fn_sc, collate_fn_nl, collate_fn_sf, build_optimizer, get_device
from logger import Logger
from data_engine import DataEngineSplit

from tqdm import tqdm


class DualModel:
    def __init__(
            self,
            batch_size,
            optimizer,
            learning_rate,
            train_data_engine,
            test_data_engine,
            dim_hidden,
            dim_embedding,
            vocab_size=None,
            attr_vocab_size=None,
            n_layers=1,
            bidirectional=False,
            model_dir="./model",
            log_dir="./log",
            is_load=True,
            replace_model=True,
            model='FGSC-FGSG',
            schedule='iterative',
            device=None,
            dir_name='test',
            f1_per_sample=False,
    ):

        # Initialize attributes
        self.data_engine = train_data_engine
        self.n_layers = n_layers
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.dim_hidden = dim_hidden
        self.dim_embedding = dim_embedding
        self.vocab_size = vocab_size
        self.attr_vocab_size = attr_vocab_size
        self.dir_name = dir_name
        self.model = model
        self.schedule = schedule
        self.f1_per_sample = f1_per_sample

        self.device = get_device(device)

        self.sc = SCModel(
            dim_embedding=dim_embedding,
            dim_hidden=dim_hidden,
            attr_vocab_size=attr_vocab_size,
            vocab_size=vocab_size,
            n_layers=n_layers,
            bidirectional=bidirectional)

        self.sg = SGModel(
            dim_embedding=dim_embedding,
            dim_hidden=dim_hidden,
            attr_vocab_size=attr_vocab_size,
            vocab_size=vocab_size,
            n_layers=n_layers,
            bidirectional=False)

        self.sc.to(self.device)
        self.sg.to(self.device)

        self.model_dir, self.log_dir = handle_model_dirs(
            model_dir, log_dir, dir_name, replace_model, is_load
        )

        if is_load:
            self.load_model(self.model_dir)

        self.train_sc_data_loader = DataLoader(
            train_data_engine,
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
            drop_last=True,
            collate_fn=collate_fn_sc,
            pin_memory=True)
        self.train_sg_data_loader = DataLoader(
            train_data_engine,
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
            drop_last=True,
            collate_fn=collate_fn_sg,
            pin_memory=True)

        # Initialize data loaders and optimizers
        self.train_data_engine = train_data_engine
        self.test_data_engine = test_data_engine

        self.test_sc_data_loader = DataLoader(
            test_data_engine,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
            drop_last=True,
            collate_fn=collate_fn_sc,
            pin_memory=True)

        self.test_sg_data_loader = DataLoader(
            test_data_engine,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
            drop_last=True,
            collate_fn=collate_fn_sg,
            pin_memory=True)

        # sc parameters optimization
        self.sc_parameters = filter(
            lambda p: p.requires_grad, self.sc.parameters())
        self.sc_optimizer = build_optimizer(
            optimizer, self.sc_parameters,
            learning_rate)
        # sg parameters optimization
        self.sg_parameters = filter(
            lambda p: p.requires_grad, self.sg.parameters())
        self.sg_optimizer = build_optimizer(
            optimizer, self.sg_parameters,
            learning_rate)

        print_time_info("Model create complete")

        # Initialize the log files
        # self.logger = Logger(self.log_dir) # not used
        self.train_log_path = os.path.join(self.log_dir, "train_log.csv")
        self.valid_log_path = os.path.join(
            self.log_dir, "valid_log.csv")

        with open(self.train_log_path, 'w') as file:
            file.write("epoch,sc_loss,sg_loss,micro_f1,"
                       "bleu,rouge(1,2,L,BE)\n")
        with open(self.valid_log_path, 'w') as file:
            file.write("epoch,sc_loss,sg_loss,micro_f1, "
                       "bleu,rouge(1,2,L,BE)\n")

        # Initialize batch count
        self.sc_batches = self.sg_batches = 0

    def train(self, epochs, batch_size, criterion_sc, criterion_sg,
              verbose_epochs=1, verbose_batches=1,
              valid_epochs=1, valid_batches=1000,
              save_epochs=10,
              teacher_forcing_ratio=0.5,
              tf_decay_rate=0.9,
              max_norm=0.25,
              mid_sample_size=1,
              dual_sample_size=1,
              sc_st=True,
              sg_st=True,
              primal_supervised=True,
              dual_supervised=True,
              primal_reinforce=False,
              dual_reinforce=True):

        if mid_sample_size > 1 and dual_sample_size > 1:
            raise ValueError("mid_sample_size > 1 and dual_sample_size > 1 "
                             "is not allowed")

        self.sc_batches = self.sg_batches = 0

        def train_sc():
            epoch_sc_loss = 0
            batch_amount_sc = 0
            scorer = MultilabelScorer(f1_per_sample=self.f1_per_sample)
            criterion_sc.set_scorer(scorer)
            pbar = tqdm(self.train_sc_data_loader, dynamic_ncols=True)
            for b_idx, batch in enumerate(pbar):
                self.sc_batches += 1
                batch_loss, batch_logits, _, _, _ = self.run_sc_batch(
                    batch,
                    criterion_sc,
                    scorer=scorer,
                    testing=False,
                    max_norm=max_norm,
                    sample_size=mid_sample_size,
                    supervised=primal_supervised,
                    reinforce=primal_reinforce
                )
                epoch_sc_loss += batch_loss.item()
                batch_amount_sc += 1
                pbar.set_postfix(ULoss="{:.5f}".format(epoch_sc_loss / batch_amount_sc))

            scorer.print_avg_scores()
            return epoch_sc_loss / batch_amount_sc, scorer

        def train_sg():
            epoch_sg_loss = 0
            batch_amount_sg = 0
            scorer = SequenceScorer()
            criterion_sg.set_scorer(scorer)
            pbar = tqdm(self.train_sg_data_loader, dynamic_ncols=True)
            for b_idx, batch in enumerate(pbar):
                self.sg_batches += 1
                batch_loss, batch_logits, batch_decode_result, _, _ = self.run_sg_batch(
                    batch,
                    criterion_sg,
                    scorer=scorer,
                    testing=False,
                    teacher_forcing_ratio=teacher_forcing_ratio,
                    max_norm=max_norm,
                    beam_size=mid_sample_size,
                    supervised=primal_supervised,
                    reinforce=primal_reinforce
                )
                epoch_sg_loss += batch_loss.item()
                batch_amount_sg += 1
                pbar.set_postfix(GLoss="{:.5f}".format(epoch_sg_loss / batch_amount_sg))

            scorer.print_avg_scores()
            return epoch_sg_loss / batch_amount_sg, scorer

        def train_joint():
            sg_loss = sc_loss = 0
            sg_loss_gen = sc_loss_gen = 0
            batch_amount = 0
            sc_scorer = MultilabelScorer(f1_per_sample=self.f1_per_sample)
            sc_scorer_gen = MultilabelScorer(f1_per_sample=self.f1_per_sample)
            sg_scorer = SequenceScorer()
            sg_scorer_gen = SequenceScorer()
            criterion_sg.set_scorer(sg_scorer)
            criterion_sc.set_scorer(sc_scorer)
            pbar = tqdm(
                zip(self.train_sg_data_loader, self.train_sc_data_loader),
                total=len(self.train_sg_data_loader),
                dynamic_ncols=True
            )

            for batch_sg, batch_sc in pbar:
                batch_loss_sg, batch_loss_sc = train_sg_sc_joint_batch(
                    batch_sg,
                    sc_scorer=sc_scorer_gen,
                    sg_scorer=sg_scorer)
                sg_loss += batch_loss_sg
                sc_loss_gen += batch_loss_sc

                batch_loss_sg, batch_loss_sc = train_sc_sg_joint_batch(
                    batch_sc,
                    sc_scorer=sc_scorer,
                    sg_scorer=sg_scorer_gen)
                sg_loss_gen += batch_loss_sg
                sc_loss += batch_loss_sc

                batch_amount += 1

                pbar.set_postfix(
                    UT="{:.4f}".format(sc_loss / batch_amount),
                    UF="{:.4f}".format(sc_loss_gen / batch_amount),
                    GT="{:.3f}".format(sg_loss / batch_amount),
                    GF="{:.3f}".format(sg_loss_gen / batch_amount)
                )

            print_time_info("True SG scores:")
            sg_scorer.print_avg_scores()
            print_time_info("Generated SG scores:")
            sg_scorer_gen.print_avg_scores()

            print_time_info("True SC scores:")
            sc_scorer.print_avg_scores()
            print_time_info("Generated SC scores:")
            sc_scorer_gen.print_avg_scores()

            return (
                sg_loss / batch_amount,
                sg_loss_gen / batch_amount,
                sc_loss / batch_amount,
                sc_loss_gen / batch_amount,
                sg_scorer,
                sg_scorer_gen,
                sc_scorer,
                sc_scorer_gen
            )

        def train_sg_sc_joint_batch(batch, sc_scorer=None, sg_scorer=None):
            criterion_sg.set_scorer(sg_scorer)
            criterion_sc.set_scorer(sc_scorer)
            encoder_input, decoder_label, refs, sf_data = batch
            self.sg_batches += 1
            batch_loss_sg, batch_logits, batch_decode_result, \
                sg_joint_prob, last_reward = self.run_sg_batch(
                batch,
                criterion_sg,
                scorer=sg_scorer,
                testing=False,
                teacher_forcing_ratio=teacher_forcing_ratio,
                max_norm=max_norm,
                retain_graph=True,
                optimize=False,
                beam_size=mid_sample_size,
                sg_st=sg_st,
                supervised=primal_supervised,
                reinforce=primal_reinforce
            )
            generated_batch = [batch_decode_result, encoder_input, refs, sf_data]
            self.sc_batches += 1
            batch_loss_sc, batch_logits, _, _, _ = self.run_sc_batch_dual(
                generated_batch,
                criterion_sc,
                scorer=sc_scorer,
                joint_prob_other=sg_joint_prob if mid_sample_size > 1 else None,
                max_norm=max_norm,
                last_reward=last_reward,
                sample_size=dual_sample_size,
                supervised=dual_supervised,
                reinforce=dual_reinforce
            )

            return batch_loss_sg.item(), batch_loss_sc.item()

        def train_sc_sg_joint_batch(batch, sc_scorer=None, sg_scorer=None):
            criterion_sg.set_scorer(sg_scorer)
            criterion_sc.set_scorer(sc_scorer)
            encoder_input, decoder_label, refs, sf_data = batch
            self.sc_batches += 1
            batch_loss_sc, batch_logits, samples, \
                sc_joint_prob, last_reward = self.run_sc_batch(
                batch,
                criterion_sc,
                scorer=sc_scorer,
                testing=False,
                max_norm=max_norm,
                retain_graph=True,
                optimize=False,
                sample_size=mid_sample_size,
                supervised=primal_supervised,
                reinforce=primal_reinforce
            )

            if sc_st:
                generated_batch = [samples, encoder_input, refs, sf_data]
            else:
                generated_batch = [self._st_sigmoid(batch_logits, hard=False).unsqueeze(1).expand(-1, mid_sample_size, -1), encoder_input, refs,
                                   sf_data]

            self.sg_batches += 1
            batch_loss_sg, batch_logits, batch_decode_result, _, _ = self.run_sg_batch_dual(
                generated_batch,
                criterion_sg,
                scorer=sg_scorer,
                joint_prob_other=sc_joint_prob if mid_sample_size > 1 else None,
                teacher_forcing_ratio=teacher_forcing_ratio,
                max_norm=max_norm,
                last_reward=last_reward,
                beam_size=dual_sample_size,
                supervised=dual_supervised,
                reinforce=dual_reinforce
            )

            return batch_loss_sg.item(), batch_loss_sc.item()

    def test_dump(self, criterion_sc, criterion_sg):
        with torch.no_grad():
            with open(os.path.join(self.log_dir, 'validation', 'test_dump_NSN.txt'), 'w') as fw:
                data_count = 0
                sg_loss = 0
                batch_amount = 0
                for b_idx, batch in enumerate(tqdm(self.test_sc_data_loader)):
                    batch_amount += 1
                    encoder_input, decoder_label, refs, sf_data = batch
                    _, _, pred, _, _ = self.run_sc_batch(
                        batch,
                        criterion_sc,
                        testing=True,
                    )

                    generated_batch = [pred, encoder_input, refs, sf_data]

                    loss, _, decode_result, _, _ = self.run_sg_batch_dual(
                        generated_batch,
                        criterion_sg,
                        teacher_forcing_ratio=0.0
                    )
                    sg_loss += loss.item()

                    for i, (x, y_true, y_pred, x_pred) in enumerate(zip(
                            encoder_input,
                            decoder_label,
                            pred.cpu().long().numpy()[:, 0],
                            decode_result.cpu().long().numpy()[:, 0])):
                        x = self.data_engine.tokenizer.untokenize(x, sf_data[i])
                        y_true = self.data_engine.tokenizer.untokenize(sorted(y_true), sf_data[i], is_token=True)
                        y_pred = self.data_engine.tokenizer.untokenize(sorted(np.where(y_pred == 1)[0]), sf_data[i], is_token=True)
                        x_pred = self.data_engine.tokenizer.untokenize(np.argmax(x_pred, axis=-1), sf_data[i])
                        fw.write("Data {}\n".format(data_count + i))
                        fw.write("NL input: {}\n".format(" ".join(x)))
                        fw.write("SF label: {}\n".format(" / ".join(y_true)))
                        fw.write("SF pred: {}\n".format(" / ".join(y_pred)))
                        fw.write("NL output: {}\n\n".format(" ".join(x_pred)))
                    data_count += len(encoder_input)
                print('sg reconstruction loss {}'.format(sg_loss / batch_amount))
            with open(os.path.join(self.log_dir, 'validation', 'test_dump_SNS.txt'), 'w') as fw:
                data_count = 0
                sc_loss = 0
                batch_amount = 0
                for b_idx, batch in enumerate(tqdm(self.test_sg_data_loader)):
                    batch_amount += 1
                    encoder_input, decoder_label, refs, sf_data = batch
                    _, _, decode_result, _, _ = self.run_sg_batch(
                        batch,
                        criterion_sg,
                        teacher_forcing_ratio=0.0,
                        testing=True,
                    )

                    generated_batch = [decode_result, encoder_input, refs, sf_data]

                    loss, _, pred, _, _ = self.run_sc_batch_dual(
                        generated_batch,
                        criterion_sc,
                    )
                    sc_loss += loss.item()

                    for i, (x, y_true, y_pred, x_pred) in enumerate(zip(
                            encoder_input,
                            decoder_label,
                            decode_result.cpu().long().numpy()[:, 0],
                            pred.cpu().long().numpy()[:, 0])):
                        x = self.data_engine.tokenizer.untokenize(sorted(x), sf_data[i], is_token=True)
                        y_true = self.data_engine.tokenizer.untokenize(y_true, sf_data[i])
                        y_pred = self.data_engine.tokenizer.untokenize(np.argmax(y_pred, axis=-1), sf_data[i])
                        x_pred = self.data_engine.tokenizer.untokenize(sorted(np.where(x_pred == 1)[0]), sf_data[i], is_token=True)
                        fw.write("Data {}\n".format(data_count + i))
                        fw.write("SF input: {}\n".format(" / ".join(x)))
                        fw.write("NL label: {}\n".format(" ".join(y_true)))
                        fw.write("NL pred: {}\n".format(" ".join(y_pred)))
                        fw.write("SF output: {}\n\n".format(" / ".join(x_pred)))
                    data_count += len(encoder_input)
                print('sc reconstruction loss: {}'.format(sc_loss / batch_amount))

    def test(self, batch_size,
             criterion_sc, criterion_sg,
             test_sc=True, test_sg=True,
             sample_size=1, epoch=-1):

        sc_loss = sg_loss = None
        sc_scorer = sg_scorer = None

        batch_amount = 0

        if test_sc:
            sc_scorer = MultilabelScorer()
            sc_loss = 0
            for b_idx, batch in enumerate(tqdm(self.test_sc_data_loader)):
                with torch.no_grad():
                    batch_loss, batch_logits, _, _, _ = self.run_sc_batch(
                        batch,
                        criterion_sc,
                        scorer=sc_scorer,
                        testing=True
                    )
                sc_loss += batch_loss.item()
                batch_amount += 1

            sc_loss /= batch_amount
            sc_scorer.print_avg_scores()

        batch_amount = 0

        if test_sg:
            sg_scorer = SequenceScorer()
            sg_loss = 0
            for b_idx, batch in enumerate(tqdm(self.test_sg_data_loader)):
                with torch.no_grad():
                    batch_loss, batch_logits, batch_decode_result, _, _ = self.run_sg_batch(
                        batch,
                        criterion_sg,
                        scorer=sg_scorer,
                        testing=True,
                        teacher_forcing_ratio=0.0,
                        beam_size=sample_size,
                        result_path=os.path.join(
                            os.path.join(self.log_dir, "validation"),
                            "test.txt"
                        )
                    )

                sg_loss += batch_loss.item()
                batch_amount += 1

            sg_loss /= batch_amount
            sg_scorer.print_avg_scores()

        self._record_log(
            epoch=epoch,
            testing=True,
            sc_loss=sc_loss,
            sg_loss=sg_loss,
            sc_scorer=sc_scorer,
            sg_scorer=sg_scorer
        )

        with open("test_results.txt", 'a') as file:
            if test_sc or test_sg:
                file.write("{}\n".format(self.dir_name))
            if test_sg:
                sg_scorer.write_avg_scores_to_file(file)
            if test_sc:
                sc_scorer.write_avg_scores_to_file(file)

    def run_sc_batch(self, batch, criterion, scorer=None,
                     testing=False, optimize=True, max_norm=None,
                     retain_graph=False, result_path=None, sample_size=1,
                     supervised=True, reinforce=False):
        if testing:
            self.sc.eval()
        else:
            self.sc.train()

        encoder_input, decoder_label, refs, sf_data = batch

        inputs = torch.from_numpy(encoder_input).to(self.device)
        targets = self._sequences_to_nhot(decoder_label, self.attr_vocab_size)
        targets = torch.from_numpy(targets).float()

        logits = self.sc(inputs)
        prediction = (torch.sigmoid(logits.detach().cpu()) >= 0.5)
        prediction = prediction.clone().numpy()
        if scorer:
            targets_clone = targets.detach().cpu().long().numpy()
            scorer.update(targets_clone, prediction)

        if sample_size > 1:
            samples = self._sample_sc_output(logits, sample_size)
        else:
            samples = self._st_sigmoid(logits, hard=True).unsqueeze(1)

        sup_loss, rl_loss, sc_joint_prob, reward = criterion(
            logits.cpu().unsqueeze(1).expand(-1, sample_size, -1),
            targets.cpu(),
            decisions=samples.cpu(),
            n_supervise=1,
            calculate_reward=reinforce
        )
        has_rl = isinstance(rl_loss, torch.Tensor)

        if not testing:
            if supervised and isinstance(sup_loss, torch.Tensor):
                sup_loss.backward(retain_graph=(retain_graph or has_rl))
            if reinforce and has_rl:
                rl_loss.backward(retain_graph=retain_graph)
            if optimize:
                if max_norm:
                    clip_grad_norm_(self.sc_parameters, max_norm)
                self.sc_optimizer.step()
                self.sc_optimizer.zero_grad()

        if testing and result_path:
            self._record_sc_test_result(
                result_path,
                encoder_input,
                decoder_label,
                prediction
            )
        return sup_loss + rl_loss, logits, samples, sc_joint_prob, reward

    def run_sc_batch_dual(self, batch, criterion, scorer=None,
                          max_norm=None, joint_prob_other=None,
                          last_reward=0.0, sample_size=1, supervised=True, reinforce=True):
        self.sc.train()

        encoder_input, decoder_label, refs, sf_data = batch

        inputs = encoder_input.to(self.device)
        targets = self._sequences_to_nhot(decoder_label, self.attr_vocab_size)
        targets = torch.from_numpy(targets).float()

        sampled_input = (inputs.size(1) > 1)

        bs, ss, sl, vs = inputs.size()
        inputs = inputs.contiguous().view(bs * ss, sl, vs)
        logits = self.sc(inputs).view(bs, ss, -1)
        prediction = (torch.sigmoid(logits[:, 0].detach().cpu()) >= 0.5).clone().numpy()

        if sampled_input:
            samples = (torch.sigmoid(logits) >= 0.5).long()
        else:
            samples = self._sample_sc_output(logits.squeeze(1), sample_size)

        if scorer:
            targets_clone = targets.detach().cpu().long().numpy()
            scorer.update(targets_clone, prediction)

        sup_loss, rl_loss, sc_joint_prob, reward = criterion(
            logits.cpu(),
            targets.cpu(),
            decisions=samples.cpu().detach(),
            log_joint_prob=joint_prob_other,
            n_supervise=1,
            calculate_reward=reinforce
        )
        has_rl = isinstance(rl_loss, torch.Tensor)
        '''
        if is_dual and has_rl:
            _, dual_rl_loss, _, _ = criterion(
                logits.cpu().detach(), targets,
                supervised=False,
                log_joint_prob=joint_prob_other,
                last_reward=last_reward
            )
        '''
        if supervised and isinstance(sup_loss, torch.Tensor):
            sup_loss.backward(retain_graph=has_rl)
        if reinforce and has_rl:
            rl_loss.backward(retain_graph=False)
        if max_norm:
            clip_grad_norm_(self.sg_parameters, max_norm)
            clip_grad_norm_(self.sc_parameters, max_norm)

        self.sc_optimizer.step()
        self.sg_optimizer.step()
        self.sc_optimizer.zero_grad()
        self.sg_optimizer.zero_grad()

        return sup_loss + rl_loss, logits, samples, sc_joint_prob, reward

    def run_sg_batch(self, batch, criterion, scorer=None,
                     testing=False, optimize=True, teacher_forcing_ratio=0.5,
                     max_norm=None, retain_graph=False, result_path=None,
                     beam_size=1, sg_st=True, supervised=True, reinforce=False):
        if testing:
            self.sg.eval()
        else:
            self.sg.train()

        encoder_input, decoder_label, refs, sf_data = batch

        attrs = self._sequences_to_nhot(encoder_input, self.attr_vocab_size)
        attrs = torch.from_numpy(attrs).to(self.device)
        labels = torch.from_numpy(decoder_label).to(self.device)

        # logits.size() == (batch_size, beam_size, seq_length, vocab_size)
        # outputs.size() == (batch_size, beam_size, seq_length, vocab_size) one-hot vectors
        # Note that outputs are still in computational graph
        logits, outputs, decisions = self.sg(
            attrs, _BOS, labels, beam_size=beam_size,
            tf_ratio=teacher_forcing_ratio if not testing else 0.0,
            st=sg_st
        )

        batch_size, _, seq_length, vocab_size = logits.size()

        outputs_indices = decisions[:, 0].detach().cpu().clone().numpy()
        outputs_indices = np.argmax(outputs_indices, axis=-1)
        if scorer:
            labels_clone = labels.detach().cpu().numpy()
            scorer.update(labels_clone, refs, outputs_indices)

        sup_loss, rl_loss, sg_joint_prob, reward = criterion(
            logits.cpu(),
            labels.cpu(),
            decisions=decisions.cpu(),
            n_supervise=1,
            calculate_reward=reinforce
        )
        has_rl = isinstance(rl_loss, torch.Tensor)

        if not testing:
            if supervised and isinstance(sup_loss, torch.Tensor):
                sup_loss.backward(retain_graph=(retain_graph or has_rl))
            if reinforce and has_rl:
                rl_loss.backward(retain_graph=retain_graph)
            if optimize:
                if max_norm:
                    clip_grad_norm_(self.sg_parameters, max_norm)
                self.sg_optimizer.step()
                self.sg_optimizer.zero_grad()
                self.sc_optimizer.zero_grad()

        if testing and result_path:
            self._record_sg_test_result(
                result_path,
                encoder_input,
                decoder_label,
                sf_data,
                outputs_indices
            )
        # print(sup_loss, rl_loss)
        return sup_loss + rl_loss, logits, outputs, sg_joint_prob, reward
        # return sup_loss, logits, outputs, sg_joint_prob, reward

    def run_sg_batch_dual(self, batch, criterion, scorer=None,
                          joint_prob_other=None,
                          teacher_forcing_ratio=0.5,
                          max_norm=None, last_reward=0.0, beam_size=1,
                          supervised=True, reinforce=True):

        self.sg.train()

        encoder_input, decoder_label, refs, sf_data = batch

        attrs = encoder_input.to(self.device)
        labels = torch.from_numpy(decoder_label).to(self.device)

        sampled_input = (attrs.size(1) > 1)
        bs, ss, vs = attrs.size()
        attrs = attrs.contiguous().view(-1, vs)
        # logits.size() == (batch_size, beam_size, seq_length, vocab_size)
        # outputs.size() == (batch_size, beam_size, seq_length, vocab_size) one-hot vectors
        # Note that outputs are still in computational graph
        logits, outputs, decisions = self.sg(
            attrs, _BOS,
            labels.unsqueeze(1).expand(-1, ss, -1).contiguous().view(-1, labels.size(-1)),
            beam_size=beam_size,
            tf_ratio=teacher_forcing_ratio
        )

        if sampled_input:
            _, _, sl, vs = logits.size()
            logits = logits.view(bs, ss, sl, vs)
            outputs = outputs.view(bs, ss, sl, vs)
            decisions = decisions.view(bs, ss, sl, vs)

        batch_size, _, seq_length, vocab_size = logits.size()

        outputs_indices = decisions[:, 0].detach().cpu().clone().numpy()
        outputs_indices = np.argmax(outputs_indices, axis=-1)
        if scorer:
            labels_clone = labels.detach().cpu().numpy()
            scorer.update(labels_clone, refs, outputs_indices)

        sup_loss, rl_loss, sg_joint_prob, reward = criterion(
            logits.cpu().contiguous(),
            labels.cpu().contiguous(),
            decisions=decisions.cpu().detach(),
            log_joint_prob=joint_prob_other,
            n_supervise=1,
            calculate_reward=reinforce
        )
        has_rl = isinstance(rl_loss, torch.Tensor)
        '''
        if is_dual and has_rl:
            _, dual_rl_loss, _, _ = criterion(
                logits.cpu().detach().contiguous().view(-1, vocab_size),
                labels.cpu().contiguous().view(-1),
                outputs.cpu(),
                supervised=False,
                log_joint_prob=joint_prob_other,
                last_reward=last_reward
            )
        '''

        if supervised and isinstance(sup_loss, torch.Tensor):
            sup_loss.backward(retain_graph=has_rl)
        if reinforce and has_rl:
            rl_loss.backward(retain_graph=False)
        if max_norm:
            clip_grad_norm_(self.sc_parameters, max_norm)
            clip_grad_norm_(self.sg_parameters, max_norm)

        self.sg_optimizer.step()
        self.sc_optimizer.step()
        self.sg_optimizer.zero_grad()
        self.sc_optimizer.zero_grad()

        return sup_loss + rl_loss, logits, outputs, sg_joint_prob, reward

    def save_model(self, model_dir):
        sc_path = os.path.join(model_dir, "sc.ckpt")
        sg_path = os.path.join(model_dir, "sg.ckpt")
        torch.save(self.sc, sc_path)
        torch.save(self.sg, sg_path)
        print_time_info("Save model successfully")

    def load_model(self, model_dir):
        # Get the latest modified model (files or directory)
        sc_path = os.path.join(model_dir, "sc.ckpt")
        sg_path = os.path.join(model_dir, "sg.ckpt")

        if not os.path.exists(sc_path) or not os.path.exists(sg_path):
            print_time_info("Loading failed, start training from scratch...")
        else:
            self.sc = torch.load(sc_path, map_location=self.device)
            self.sg = torch.load(sg_path, map_location=self.device)
            print_time_info("Load model from {} successfully".format(model_dir))

    def _sequences_to_nhot(self, seqs, vocab_size):
        """
        args:
            seqs: list of list of word_ids
            vocab_size: int

        outputs:
            labels: np.array of shape [batch_size, vocab_size]
        """
        labels = np.zeros((len(seqs), vocab_size), dtype=np.int)
        for bid, seq in enumerate(seqs):
            for word in seq:
                labels[bid][word] = 1
        return labels

    def _sample_sc_output(self, logits, sample_size=1):
        """
        args:
            logits: tensor of shape (batch_size, vocab_size), unnormalized logits
        returns:
            samples: tensor of shape (batch_size, sample_size, vocab_size), 0/1 decisions
        """
        y_soft = logits.sigmoid()
        y_soft_clone = y_soft.detach().cpu().clone().numpy()
        samples = []
        for i in range(sample_size):
            sample = torch.tensor([
                [random.random() < y_soft_clone[b, v] for v in range(y_soft_clone.shape[1])]
                for b in range(y_soft_clone.shape[0])
            ], dtype=torch.float, device=logits.device)
            samples.append(sample)
        y_hard = torch.stack(samples, dim=0).transpose(0, 1)
        y_soft = y_soft.unsqueeze(1).expand(-1, sample_size, -1)
        return y_hard - y_soft.detach() + y_soft

    def _st_sigmoid(self, logits, hard=False):
        """
        args:
            logits: tensor of shape (*, vocab_size), unnormalized logits
            hard: boolean, whether to return one-hot decisions, or probabilities.
        returns:
            decisions: tensor of shape (*, vocab_size), 0/1 decisions
        """
        y_soft = logits.sigmoid()

        if hard:
            # Straight through.
            y_hard = (y_soft >= 0.5).float()
            ret = y_hard - y_soft.detach() + y_soft
        else:
            ret = y_soft

        return ret

    def _record_log(self,
                    epoch,
                    testing,
                    sc_loss=None,
                    sg_loss=None,
                    sc_scorer=None,
                    sg_scorer=None):
        filename = self.valid_log_path if testing else self.train_log_path
        sc_loss = 'None' if sc_loss is None else '{:.4f}'.format(sc_loss)
        sg_loss = 'None' if sg_loss is None else '{:.3f}'.format(sg_loss)
        if sc_scorer is not None:
            micro_f1, _ = sc_scorer.get_avg_scores()
            micro_f1 = '{:.4f}'.format(micro_f1)
        else:
            micro_f1 = '-1.0'
        if sg_scorer is not None:
            _, bleu, _, rouge, _ = sg_scorer.get_avg_scores()
            bleu = '{:.4f}'.format(bleu)
            rouge = ' '.join(['{:.4f}'.format(s) for s in rouge])
        else:
            bleu, rouge = '-1.0', '-1.0 -1.0 -1.0'
        with open(filename, 'a') as file:
            file.write("{},{},{},{},"
                       "{},{}\n".format(epoch, sc_loss, sg_loss, micro_f1, bleu, rouge))

    def _record_sc_test_result(self,
                               result_path,
                               encoder_input,
                               decoder_label,
                               prediction):
        pass

    def _record_sg_test_result(self, result_path, encoder_input,
                               decoder_label, sf_data, decoder_result):
        encoder_input = [
            self.data_engine.tokenizer.untokenize(sent, sf_data[idx], is_token=True)
            for idx, sent in enumerate(encoder_input)
        ]
        decoder_label = [
            self.data_engine.tokenizer.untokenize(sent, sf_data[idx])
            for idx, sent in enumerate(decoder_label)
        ]
        decoder_result = [
            self.data_engine.tokenizer.untokenize(sent, sf_data[idx])
            for idx, sent in enumerate(decoder_result)
        ]

        with open(result_path, 'a') as file:
            for idx in range(len(encoder_input)):
                file.write("---------\n")
                file.write("Data {}\n".format(idx))
                file.write("encoder input: {}\n".format(' '.join(encoder_input[idx])))
                file.write("decoder output: {}\n".format(' '.join(decoder_result[idx])))
                file.write("decoder label: {}\n".format(' '.join(decoder_label[idx])))


class DSLCriterion(nn.Module):
    def __init__(self, loss_weight, pretrain_epochs=0,
                 LM=None, LM2=None, lambda_xy=0.1, lambda_yx=0.1,
                 made_n_samples=1, propagate_other=False):
        super(DSLCriterion, self).__init__()
        self.pretrain_epochs = pretrain_epochs
        self.epoch = 0
        self.propagate_other = propagate_other
        self.lambda_xy = lambda_xy
        self.lambda_yx = lambda_yx
        self.LM = LM
        self.LM2 = LM2
        if LM is None:
            raise ValueError("Language model not provided")
        if LM2 is None:
            raise ValueError("Language model v2 not provided")

        self.made_n_samples = made_n_samples
        self.BCE = nn.BCEWithLogitsLoss(reduction='sum')
        self.CE = nn.CrossEntropyLoss(weight=loss_weight, reduction='sum')

    def get_log_joint_prob_nlg(self, logits, decisions):
        """
        args:
            logits: tensor of shape [batch_size, seq_length, vocab_size]
            decisions: tensor of shape [batch_size, seq_length, vocab_size]
                       one-hot vector of decoded word-ids
        returns:
            log_joint_prob: tensor of shape [batch_size]
        """
        probs = torch.softmax(logits, dim=-1)
        return (decisions * probs).sum(dim=-1).log().sum(dim=-1)

    def get_log_joint_prob_nlu(self, logits, decisions):
        """
        args:
            logits: tensor of shape [batch_size, attr_vocab_size]
            decisions: tensor of shape [batch_size, attr_vocab_size]
                       decisions(0/1)
        returns:
            log_joint_prob: tensor of shape [batch_size]
        """
        probs = torch.sigmoid(logits)
        decisions = decisions.float()
        probs = probs * decisions + (1 - probs) * (1 - decisions)
        return probs.log().sum(dim=-1)

    def epoch_end(self):
        self.epoch += 1
        if self.epoch == self.pretrain_epochs:
            print_time_info("pretrain finished, starting using duality loss")

    def get_scheduled_loss(self, dual_loss):
        if self.epoch < self.pretrain_epochs:
            return torch.tensor(0.0)
        return dual_loss

    def forward(self, nlg_logits, nlg_outputs, nlu_logits, nlg_targets, nlu_targets):
        """
        args:
            nlg_logits: tensor of shape [batch_size, seq_length, vocab_size]
            nlg_outputs: tensor of shape [batch_size, seq_length, vocab_size]
            nlg_targets: tensor of shape [batch_size, seq_length]
            nlu_logits: tensor of shape [batch_size, attr_vocab_size]
            nlu_targets: tensor of shape [batch_size, attr_vocab_size]
        """
        nlg_logits_1d = nlg_logits.contiguous().view(-1, nlg_logits.size(-1))
        nlg_targets_1d = nlg_targets.contiguous().view(-1)
        nlg_sup_loss = self.CE(nlg_logits_1d, nlg_targets_1d)
        nlu_sup_loss = self.BCE(nlu_logits, nlu_targets)

        log_p_x = self.LM.get_log_prob(nlg_targets)
        log_p_y = self.MADE.get_log_prob(nlu_targets, n_samples=self.made_n_samples)

        log_p_y_x = self.get_log_joint_prob_nlg(nlg_logits, nlg_outputs)
        nlu_decisions = (nlu_logits.sigmoid() >= 0.5).float()
        log_p_x_y = self.get_log_joint_prob_nlu(nlu_logits, nlu_decisions)

        if self.propagate_other:
            nlg_loss_dual = (log_p_x + log_p_y_x - log_p_y - log_p_x_y).pow(2).mean()
            nlu_loss_dual = (log_p_x + log_p_y_x - log_p_y - log_p_x_y).pow(2).mean()
        else:
            nlg_loss_dual = (log_p_x + log_p_y_x - log_p_y - log_p_x_y.detach()).pow(2).mean()
            nlu_loss_dual = (log_p_x + log_p_y_x.detach() - log_p_y - log_p_x_y).pow(2).mean()

        nlg_loss_dual = self.lambda_xy * self.get_scheduled_loss(nlg_loss_dual)
        nlu_loss_dual = self.lambda_yx * self.get_scheduled_loss(nlu_loss_dual)

        return nlg_sup_loss + nlg_loss_dual, nlu_sup_loss + nlu_loss_dual, nlg_loss_dual
