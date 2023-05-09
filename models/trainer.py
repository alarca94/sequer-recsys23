import math
import torch
import logging
import numpy as np
import pandas as pd

from torch import nn
from datetime import datetime, timezone

from utils import funcs
from . import SEQUER
from utils.data_utils import rouge_score, bleu_score, DataLoader, Batchify, now_time, ids2tokens, unique_sentence_percent, \
    root_mean_square_error, mean_absolute_error, feature_detect, feature_matching_ratio, feature_coverage_ratio, \
    feature_diversity, save_results
from utils.constants import *


class DummyScheduler:
    def __init__(self, optimizer):
        self.lr = [group['lr'] for group in optimizer.param_groups]

    def step(self):
        pass

    def get_last_lr(self):
        return self.lr


class Trainer:
    def __init__(self, args, cfg, device, prediction_path):
        gen_flg = not args.no_generate
        model_name = args.model_name
        model_suffix = args.model_suffix
        dataset = args.dataset
        fold = args.fold
        seed = args.seed

        self.use_feature = cfg.get('use_feature', False)
        self.out_words = TXT_LEN
        self.text_reg = cfg.get('text_reg', 0)
        self.context_reg = cfg.get('context_reg', 0)
        self.rating_reg = cfg.get('rating_reg', 0)
        self.item_reg = cfg.get('item_reg', 0)
        self.l2_reg = cfg.get('l2_reg', 0)
        self.clip_norm = cfg.get('clip_norm', None)
        self.epochs = cfg.get('epochs', 100)
        self.batch_size = cfg.get('batch_size', 128)
        self.endure_times = cfg.get('endure_times', 5)
        self.gen_flg = gen_flg
        self.test_flg = args.test
        self.seq_mode = cfg.get('seq_mode', 0)
        self.vocab_size = cfg.get('vocab_size', 5000)
        self.device = device
        self.load_data(dataset, fold, self.seq_mode, model_name, self.test_flg)
        test_str = '_test' if self.test_flg else ''
        self.model_path = os.path.join(CKPT_PATH, f'{model_name}{model_suffix}_{dataset}_{fold}_{seed}{test_str}.pth')
        self.prediction_path = prediction_path
        self.log_interval = cfg.get('log_interval', 'all')
        if self.log_interval == 'all':
            self.log_interval = self.train_data.sample_num + 1

        cfg['max_r'] = self.corpus.max_rating
        cfg['min_r'] = self.corpus.min_rating

        if self.use_feature:
            self.src_len = 2 + self.train_data.feature.size(1)  # [u, i, f]
        else:
            self.src_len = 2  # [u, i]
        self.tgt_len = self.train_data.seq.shape[-1] - 1
        self.ntokens = len(self.corpus.word_dict)
        self.nuser = len(self.corpus.user_dict)
        self.nitem = len(self.corpus.item_dict)

        self.build_model(model_name, cfg, self.device)
        self.text_criterion = nn.NLLLoss(ignore_index=self.word2idx[PAD_TOK])  # ignore the padding when computing loss
        self.nextit_criterion = nn.NLLLoss(ignore_index=self.word2idx[UNK_TOK])  # ignore the padding when computing loss
        self.rating_criterion = nn.MSELoss(reduction='none')
        if self.model is not None:
            self.optimizer = self.get_optim(self.model.parameters(), cfg=cfg)
            self.scheduler = self.get_scheduler(self.optimizer, cfg)

        cfg_txt = '_'.join([f'{k}:{v}' for k, v in cfg.items()])
        self.exp_metadata = {'model_name': model_name + model_suffix, 'dataset': dataset, 'split_ix': fold,
                             'config': cfg_txt, 'date': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                             'seed': seed}

    @staticmethod
    def get_optim(model_params, cfg):
        if cfg['optim'].lower() == 'sgd':
            return torch.optim.SGD(model_params, lr=cfg.get('lr', 1.0))
        elif cfg['optim'].lower() == 'rmsprop':
            torch.optim.RMSprop(model_params, lr=cfg.get('lr', 0.02), alpha=cfg.get('alpha', 0.95))
        elif cfg['optim'].lower() == 'adam':
            return torch.optim.Adam(model_params, lr=cfg.get('lr', 1e-4))
        elif cfg['optim'].lower() == 'adamw':
            return torch.optim.AdamW(model_params, lr=cfg.get('lr', 1e-3))

    @staticmethod
    def get_scheduler(optim, cfg):
        if 'scheduler' in cfg:
            if cfg['scheduler'].lower() == 'steplr':
                return torch.optim.lr_scheduler.StepLR(optim, step_size=cfg.get('step_size', 1),
                                                       gamma=cfg.get('gamma', 0.25))
        return DummyScheduler(optim)

    def load_data(self, dataset, fold, seq_mode, model_name, test_flg=False):
        logging.info(now_time() + f'Loading data {os.path.join(DATA_PATHS[dataset], DATA_MODE, str(fold))}')
        self.tokenizer = None
        self.corpus = DataLoader(dataset, fold, self.vocab_size, seq_mode, self.tokenizer, test_flg)
        self.word2idx = self.corpus.word_dict.word2idx
        self.idx2word = self.corpus.word_dict.idx2word
        self.feature_set = self.corpus.feature_set
        self.train_data = Batchify(self.corpus.train, self.word2idx, self.batch_size, seq_mode, shuffle=True)
        self.val_data = Batchify(self.corpus.valid, self.word2idx, self.batch_size, seq_mode)
        self.test_data = Batchify(self.corpus.test, self.word2idx, self.batch_size, seq_mode)

    def build_model(self, model_name, cfg, device):
        logging.info(now_time() + f'Building model {model_name.upper()}')
        self.model = None
        if 'sequer' in model_name.lower():
            self.model = SEQUER(self.src_len, self.tgt_len, self.word2idx[PAD_TOK],
                                self.nuser, self.nitem, self.ntokens, cfg).to(device)

    def compute_loss(self, pred, labels):
        c_loss, r_loss, t_loss, i_loss, l2_loss = torch.tensor([0] * 5, device=self.device)
        if 'context' in pred:
            context_dis = pred['context'].unsqueeze(0).repeat(
                (self.tgt_len - 1, 1, 1))  # (batch_size, ntoken) -> (tgt_len - 1, batch_size, ntoken)
            c_loss = self.text_criterion(context_dis.view(-1, self.ntokens), labels['seq'][1:-1].reshape((-1,)))
        if 'rating' in pred:
            mask = (labels['item_seq'] != self.corpus.item_dict.entity2idx[UNK_TOK])
            r_loss = (self.rating_criterion(pred['rating'], labels['rating']) * mask.float()).sum() / mask.sum()
        if 'word' in pred:
            t_loss = self.text_criterion(pred['word'].view(-1, self.ntokens), labels['seq'][1:].reshape((-1,)))
        if pred.get('item', None) is not None:
            i_loss = self.nextit_criterion(pred['item'].view(-1, self.nitem), labels['item_seq'][:, 1:].reshape((-1,)))
        if self.l2_reg > 0:
            l2_loss = torch.cat([x.view(-1) for x in self.model.parameters()]).pow(2.).sum()
        return c_loss, r_loss, t_loss, i_loss, l2_loss

    def train_epoch(self):
        # Turn on training mode which enables dropout.
        self.model.train()
        context_loss = 0.
        text_loss = 0.
        rating_loss = 0.
        total_sample = 0
        while True:
            user, item, rating, seq, feature = self.train_data.next_batch()  # (batch_size, seq_len), data.step += 1
            batch_size = user.size(0)
            user = user.to(self.device)  # (batch_size,)
            item = item.to(self.device)
            rating = rating.to(self.device)
            seq = seq.t().to(self.device)  # (tgt_len + 1, batch_size)
            labels = {'rating': rating, 'seq': seq, 'item_seq': item}
            feature = feature.t().to(self.device)  # (1, batch_size)
            if self.use_feature:
                text = torch.cat([feature, seq[:-1]], 0)  # (src_len + tgt_len - 2, batch_size)
            else:
                text = seq[:-1]  # (src_len + tgt_len - 2, batch_size)
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            self.optimizer.zero_grad()

            # (TGTL, BSZ, NTOK) vs. (BSZ, NTOK) vs. (BSZ, HISTL+1,) vs (BSZ, HISTL, NITEM)
            pred = self.model(user, item, text)
            c_loss, r_loss, t_loss, i_loss, l2_loss = self.compute_loss(pred, labels)
            loss = self.text_reg * t_loss + self.context_reg * c_loss + self.rating_reg * r_loss + \
                   self.item_reg * i_loss + self.l2_reg * l2_loss
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem.
            if self.clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
            self.optimizer.step()

            context_loss += batch_size * c_loss.item()
            text_loss += batch_size * t_loss.item()
            rating_loss += batch_size * r_loss.item()
            total_sample += batch_size

            if self.train_data.step % self.log_interval == 0 or self.train_data.step == self.train_data.total_step:
                cur_c_loss = context_loss / total_sample
                cur_t_loss = text_loss / total_sample
                cur_r_loss = rating_loss / total_sample
                logging.info(
                    now_time() + 'context ppl {:4.4f} | text ppl {:4.4f} | rating loss {:4.4f} | {:5d}/{:5d} batches'.format(
                        math.exp(cur_c_loss), math.exp(cur_t_loss), cur_r_loss, self.train_data.step, self.train_data.total_step))
                context_loss = 0.
                text_loss = 0.
                rating_loss = 0.
                total_sample = 0
            if self.train_data.step == self.train_data.total_step:
                break

    def evaluate(self, data):
        # Turn on evaluation mode which disables dropout.
        self.model.eval()
        context_loss = 0.
        text_loss = 0.
        rating_loss = 0.
        total_sample = 0
        with torch.no_grad():
            while True:
                user, item, rating, seq, feature = data.next_batch()  # (batch_size, seq_len), data.step += 1
                batch_size = user.size(0)
                user = user.to(self.device)  # (batch_size,)
                item = item.to(self.device)
                rating = rating.to(self.device)
                seq = seq.t().to(self.device)  # (tgt_len + 1, batch_size)
                labels = {'rating': rating, 'seq': seq, 'item_seq': item}
                feature = feature.t().to(self.device)  # (1, batch_size)
                if self.use_feature:
                    text = torch.cat([feature, seq[:-1]], 0)  # (src_len + tgt_len - 2, batch_size)
                else:
                    text = seq[:-1]  # (src_len + tgt_len - 2, batch_size)
                pred = self.model(user, item, text)  # (tgt_len, batch_size, ntoken) vs. (batch_size, ntoken) vs. (batch_size,)
                c_loss, r_loss, t_loss, i_loss, _ = self.compute_loss(pred, labels)

                context_loss += batch_size * c_loss.item()
                text_loss += batch_size * t_loss.item()
                rating_loss += batch_size * r_loss.item()
                total_sample += batch_size

                if data.step == data.total_step:
                    break
        return context_loss / total_sample, text_loss / total_sample, rating_loss / total_sample

    def get_start_ixs(self, batch):
        start_ixs = torch.zeros((batch.shape[0], 1), dtype=torch.long)
        ixs = torch.nonzero(batch == self.word2idx[SEP_TOK])
        # if len(ixs):
        last_sep_ixs = torch.nonzero(torch.cat((ixs[1:, 0], (ixs[-1, 0] + 1).unsqueeze(0))) - ixs[:, 0])
        start_ixs[ixs[last_sep_ixs, 0].squeeze()] = ixs[last_sep_ixs, 1]
        return start_ixs

    def generate(self, data):
        # Turn on evaluation mode which disables dropout.
        self.model.eval()
        self.model.seq_prediction = False  # Make sure it only predicts next token for the last token

        idss_predict = []
        context_predict = []
        rating_predict = []
        with torch.no_grad():
            while True:
                user, item, rating, seq, feature = data.next_batch()
                user = user.to(self.device)  # (batch_size,)
                item = item.to(self.device)
                # bos = seq[:, 0].unsqueeze(0).to(self.device)  # (batch_size, 1)
                if self.seq_mode >= HIST_REV_MODE:
                    bos_ixs = self.get_start_ixs(seq)
                else:
                    bos_ixs = torch.zeros((seq.shape[0], 1), dtype=int)
                seq.scatter_(dim=1, index=funcs.get_span_ixs(bos_ixs, span_size=self.out_words+2)[:, 1:],
                             value=self.word2idx[PAD_TOK])
                seq = seq.t().to(self.device)
                bos_ixs = bos_ixs.to(self.device)
                feature = feature.t().to(self.device)  # (1, batch_size)
                res = self.model.generate(user, item, seq, out_words=self.out_words, feature=feature,
                                          seq_mode=self.seq_mode, start_ixs=bos_ixs)
                idss_predict.extend(res['ids'])
                context_predict.extend(res['context'])
                rating_predict.extend(res['rating'])

                if data.step == data.total_step:
                    break

        if len(data.rating.shape) == 2:
            data.rating = data.rating[:, -1]
        text_out = self.compute_metrics(data, rating_predict, idss_predict, context_predict)

        return text_out

    def compute_metrics(self, data, rating_predict, idss_predict, context_predict):
        results = self.exp_metadata.copy()
        results.update({m: 0 for m in METRICS})
        # rating
        if rating_predict:
            predicted_rating = [(r, p) for (r, p) in zip(data.rating.tolist(), rating_predict)]
            results['RMSE'] = root_mean_square_error(predicted_rating, self.corpus.max_rating, self.corpus.min_rating)
            logging.info(now_time() + 'RMSE {:7.4f}'.format(results['RMSE']))
            results['MAE'] = mean_absolute_error(predicted_rating, self.corpus.max_rating, self.corpus.min_rating)
            logging.info(now_time() + 'MAE {:7.4f}'.format(results['MAE']))
        # text
        if self.seq_mode >= HIST_REV_MODE:
            gt_seq = funcs.gather_span(data.seq, start_ixs=self.get_start_ixs(data.seq), span_size=self.out_words + 2)
        else:
            gt_seq = data.seq
        tokens_test = [ids2tokens(ids[1:], self.word2idx, self.idx2word) for ids in gt_seq.tolist()]
        tokens_predict = [ids2tokens(ids, self.word2idx, self.idx2word) for ids in idss_predict]
        results['BLEU-1'] = bleu_score(tokens_test, tokens_predict, n_gram=1, smooth=False)
        logging.info(now_time() + 'BLEU-1 {:7.4f}'.format(results['BLEU-1']))
        results['BLEU-4'] = bleu_score(tokens_test, tokens_predict, n_gram=4, smooth=False)
        logging.info(now_time() + 'BLEU-4 {:7.4f}'.format(results['BLEU-4']))
        results['USR'], results['USN'] = unique_sentence_percent(tokens_predict)
        logging.info(now_time() + 'USR {:7.4f} | USN {:7}'.format(results['USR'], results['USN']))
        feature_batch = feature_detect(tokens_predict, self.feature_set)
        results['DIV'] = feature_diversity(feature_batch)  # time-consuming
        logging.info(now_time() + 'DIV {:7.4f}'.format(results['DIV']))
        results['FCR'] = feature_coverage_ratio(feature_batch, self.feature_set)
        logging.info(now_time() + 'FCR {:7.4f}'.format(results['FCR']))
        feature_test = [self.idx2word[i] for i in data.feature.squeeze(1).tolist()]  # ids to words
        results['FMR'] = feature_matching_ratio(feature_batch, feature_test, ignore=UNK_TOK)
        logging.info(now_time() + 'FMR {:7.4f}'.format(results['FMR']))
        text_test = [' '.join(tokens) for tokens in tokens_test]
        text_predict = [' '.join(tokens) for tokens in tokens_predict]
        ROUGE = rouge_score(text_test, text_predict)  # a dictionary
        for (k, v) in ROUGE.items():
            results[k] = v
            logging.info(now_time() + '{} {:7.4f}'.format(k, v))
        text_out = ''
        if not self.test_flg:
            save_results(results)
        if self.gen_flg:
            df = {'True': text_test, 'Pred': text_predict}
            if context_predict:
                df['Cntx'] = [' '.join([self.idx2word[i] for i in ids]) for ids in context_predict]
            #     tokens_context = [' '.join([self.idx2word[i] for i in ids]) for ids in context_predict]
            #     for (real, ctx, fake) in zip(text_test, tokens_context, text_predict):
            #         text_out += 'True: {}\nCntx: {}\nPred: {}\n\n'.format(real, ctx, fake)
            # else:
            #     for (real, fake) in zip(text_test, text_predict):
            #         text_out += 'True: {}\nPred: {}\n\n'.format(real, fake)
            idx2entity = np.array(self.corpus.item_dict.idx2entity)
            df['Items'] = [' '.join(idx2entity[ids].tolist()).replace(UNK_TOK, '').strip()
                           for ids in data.item.detach().cpu().numpy()]
            df['User'] = [self.corpus.user_dict.idx2entity[u] for u in data.user.cpu().numpy()]
            df = pd.DataFrame.from_dict(df, orient='columns')
            text_out = df.to_json(orient="records", indent=2)

        return text_out

    def train(self):
        best_val_loss = float('inf')
        endure_count = 0
        for epoch in range(1, self.epochs + 1):
            logging.info(now_time() + 'epoch {}'.format(epoch))
            self.train_epoch()
            val_c_loss, val_t_loss, val_r_loss = self.evaluate(self.val_data)
            if self.rating_reg == 0:
                val_loss = val_t_loss
            else:
                val_loss = val_t_loss + val_r_loss
            logging.info(
                now_time() + 'context ppl {:4.4f} | text ppl {:4.4f} | rating loss {:4.4f} | valid loss {:4.4f} on validation'.format(
                    math.exp(val_c_loss), math.exp(val_t_loss), val_r_loss, val_loss))
            # Save the model if the validation loss is the best we've seen so far.
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                with open(self.model_path, 'wb') as f:
                    torch.save(self.model, f)
            else:
                endure_count += 1
                logging.info(now_time() + 'Endured {} time(s)'.format(endure_count))
                if endure_count == self.endure_times:
                    logging.info(now_time() + 'Cannot endure it anymore | Exiting from early stop')
                    break
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                self.scheduler.step()
                logging.info(now_time() + 'Learning rate set to {:2.8f}'.format(self.scheduler.get_last_lr()[0]))

    def test(self):
        # Load the best saved model.
        with open(self.model_path, 'rb') as f:
            self.model = torch.load(f).to(self.device)

        # Run on test data.
        test_c_loss, test_t_loss, test_r_loss = self.evaluate(self.test_data)
        logging.info('=' * 89)
        logging.info(
            now_time() + 'context ppl {:4.4f} | text ppl {:4.4f} | rating loss {:4.4f} on test | End of training'.format(
                math.exp(test_c_loss), math.exp(test_t_loss), test_r_loss))

        logging.info(now_time() + 'Generating text')
        text_o = self.generate(self.test_data)
        if self.gen_flg:
            with open(self.prediction_path, 'w', encoding='utf-8') as f:
                f.write(text_o)
            logging.info(now_time() + 'Generated text saved to ({})'.format(self.prediction_path))
