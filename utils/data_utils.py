import os
import math
import torch
import heapq
import random
import pickle
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .constants import *
from .rouge import rouge
from .bleu import compute_bleu


def init_seed(seed=RNG_SEED, reproducibility=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def rouge_score(references, generated):
    """both are a list of strings"""
    score = rouge(generated, references)
    rouge_s = {k: (v * 100) for (k, v) in score.items()}
    '''
    "rouge_1/f_score": rouge_1_f,
    "rouge_1/r_score": rouge_1_r,
    "rouge_1/p_score": rouge_1_p,
    "rouge_2/f_score": rouge_2_f,
    "rouge_2/r_score": rouge_2_r,
    "rouge_2/p_score": rouge_2_p,
    "rouge_l/f_score": rouge_l_f,
    "rouge_l/r_score": rouge_l_r,
    "rouge_l/p_score": rouge_l_p,
    '''
    return rouge_s


def bleu_score(references, generated, n_gram=4, smooth=False):
    """a list of lists of tokens"""
    formatted_ref = [[ref] for ref in references]
    bleu_s, _, _, _, _, _ = compute_bleu(formatted_ref, generated, n_gram, smooth)
    return bleu_s * 100


def two_seq_same(sa, sb):
    if len(sa) != len(sb):
        return False
    for (wa, wb) in zip(sa, sb):
        if wa != wb:
            return False
    return True


def unique_sentence_percent(sequence_batch):
    unique_seq = []
    for seq in sequence_batch:
        count = 0
        for uni_seq in unique_seq:
            if two_seq_same(seq, uni_seq):
                count += 1
                break
        if count == 0:
            unique_seq.append(seq)

    return len(unique_seq) / len(sequence_batch), len(unique_seq)


def feature_detect(seq_batch, feature_set):
    feature_batch = []
    for ids in seq_batch:
        feature_list = []
        for i in ids:
            if i in feature_set:
                feature_list.append(i)
        feature_batch.append(set(feature_list))

    return feature_batch


def feature_matching_ratio(feature_batch, test_feature, ignore=None):
    count = 0
    norm = sum([f != ignore for f in test_feature])
    for (fea_set, fea) in zip(feature_batch, test_feature):
        if fea != ignore and fea in fea_set:
            count += 1

    return count / norm  # len(feature_batch)


def feature_coverage_ratio(feature_batch, feature_set):
    features = set()
    for fb in feature_batch:
        features = features | fb

    return len(features) / len(feature_set)


def feature_diversity(feature_batch):
    list_len = len(feature_batch)

    total_count = 0
    for i, x in enumerate(feature_batch):
        for j in range(i + 1, list_len):
            y = feature_batch[j]
            total_count += len(x & y)

    denominator = list_len * (list_len - 1) / 2
    return total_count / denominator


def mean_absolute_error(predicted, max_r, min_r, mae=True):
    total = 0
    for (r, p) in predicted:
        if p > max_r:
            p = max_r
        if p < min_r:
            p = min_r

        sub = p - r
        if mae:
            total += abs(sub)
        else:
            total += sub ** 2

    return total / len(predicted)


def root_mean_square_error(predicted, max_r, min_r):
    mse = mean_absolute_error(predicted, max_r, min_r, False)
    return math.sqrt(mse)


class WordDictionary:
    def __init__(self, initial_tokens=None):
        self.idx2word = []
        if initial_tokens is not None:
            self.idx2word += initial_tokens
        self.__predefine_num = len(self.idx2word)
        self.word2idx = {w: i for i, w in enumerate(self.idx2word)}
        self.__word2count = {}

    def add_sentence(self, sentence):
        for w in sentence.split():
            self.add_word(w)

    def add_word(self, w):
        if w not in self.word2idx:
            self.word2idx[w] = len(self.idx2word)
            self.idx2word.append(w)
            self.__word2count[w] = 1
        else:
            self.__word2count[w] += 1

    def __len__(self):
        return len(self.idx2word)

    def keep_most_frequent(self, max_vocab_size=20000):
        if len(self.__word2count) > max_vocab_size:
            frequent_words = heapq.nlargest(max_vocab_size, self.__word2count, key=self.__word2count.get)
            self.idx2word = self.idx2word[:self.__predefine_num] + frequent_words
            self.word2idx = {w: i for i, w in enumerate(self.idx2word)}


class EntityDictionary:
    def __init__(self, initial_tokens=None):
        self.idx2entity = []
        if initial_tokens is not None:
            self.idx2entity += initial_tokens
        self.entity2idx = {e: i for i, e in enumerate(self.idx2entity)}

    def add_entity(self, e):
        if e not in self.entity2idx:
            self.entity2idx[e] = len(self.idx2entity)
            self.idx2entity.append(e)

    def __len__(self):
        return len(self.idx2entity)


class DataLoader:
    def __init__(self, dataset, fold, vocab_size, seq_mode=0, tokenizer=None, test_flag=False):
        index_dir = os.path.join(DATA_PATHS[dataset], DATA_MODE, str(fold))
        data_path = os.path.join(DATA_PATHS[dataset], DATA_MODE, 'reviews_new.pickle')

        initial_tokens = [BOS_TOK, EOS_TOK, PAD_TOK, UNK_TOK, SEP_TOK]
        self.word_dict = WordDictionary(initial_tokens)
        self.user_dict = EntityDictionary()
        self.item_dict = EntityDictionary(initial_tokens)
        self.max_rating = float('-inf')
        self.min_rating = float('inf')
        self.initialize(data_path)
        self.word_dict.keep_most_frequent(vocab_size)
        self.__unk = self.word_dict.word2idx[UNK_TOK]
        self.feature_set = set()
        self.tokenizer = tokenizer
        self.train, self.valid, self.test = self.load_data(data_path, index_dir, seq_mode, test_flag)

    def initialize(self, data_path):
        assert os.path.exists(data_path)
        reviews = pickle.load(open(data_path, 'rb'))
        # reviews.sort_values(by=[U_COL, TIME_COL], inplace=True)
        for review in reviews:
            self.user_dict.add_entity(review[U_COL])
            self.item_dict.add_entity(review[I_COL])
            self.word_dict.add_sentence(review[REV_COL])
            # # NOTE: I've added the next line of code so that all words are in dictionary. Comment it out if necessary
            if review[FEAT_COL] != '':
                self.word_dict.add_word(review[FEAT_COL])
            rating = review[RAT_COL]
            if self.max_rating < rating:
                self.max_rating = rating
            if self.min_rating > rating:
                self.min_rating = rating

    def load_data(self, data_path, index_dir, seq_mode=0, test_flag=False):
        data = []
        merge_tok = [self.word_dict.word2idx[SEP_TOK]]
        reviews = pickle.load(open(data_path, 'rb'))
        for review in reviews:
            # (fea, adj, tem, sco) = review['template']
            data.append({U_COL: self.user_dict.entity2idx[review[U_COL]],
                         I_COL: self.item_dict.entity2idx[review[I_COL]],
                         RAT_COL: review[RAT_COL],
                         REV_COL: self.seq2ids(review[REV_COL], max_len=TXT_LEN),
                         # NOTE: This line is different in NRT/Att2Seq and PETER
                         FEAT_COL: self.word_dict.word2idx.get(review[FEAT_COL], self.__unk)})
            if seq_mode >= HIST_I_MODE:
                data[-1][HIST_I_COL] = [self.item_dict.entity2idx[i] for i in review[HIST_I_COL][-HIST_LEN:]]
                data[-1][HIST_RAT_COL] = review[HIST_RAT_COL][-HIST_LEN:]
                if seq_mode >= HIST_REV_MODE:
                    # TODO: Add segment ids to let the model know to which item in the sequence each review belongs to
                    data[-1][HIST_REV_COL] = sum([self.seq2ids(r, max_len=TXT_LEN) + merge_tok for r in review[HIST_REV_COL][-HIST_LEN:]], [])[:-1]
            # NOTE: This if-statement is not present in NRT/Att2Seq code. PETER adds it with the UNK token.
            # TODO: Check how many features are actually added to the feature set
            if review[FEAT_COL] in self.word_dict.word2idx:
                self.feature_set.add(review[FEAT_COL])
            # else:
            #     self.feature_set.add(UNK_TOK)

        train_index, valid_index, test_index = self.load_index(index_dir)
        if test_flag:
            train_index = train_index[:1000]
        train, valid, test = [], [], []
        for idx in train_index:
            train.append(data[idx])
        for idx in valid_index:
            valid.append(data[idx])
        for idx in test_index:
            test.append(data[idx])
        return train, valid, test

    def seq2ids(self, seq, max_len=TXT_LEN):
        return [self.word_dict.word2idx.get(w, self.__unk) for w in seq.split()[:max_len]]

    def load_index(self, index_dir):
        assert os.path.exists(index_dir)
        train_index = np.load(os.path.join(index_dir, 'train.npy')).tolist()
        valid_index = np.load(os.path.join(index_dir, 'validation.npy')).tolist()
        test_index = np.load(os.path.join(index_dir, 'test.npy')).tolist()
        return train_index, valid_index, test_index


def sentence_format(sentence, pad, bos, eos, sep, prefix=None):
    max_len = TXT_LEN
    if prefix is not None:
        max_len += (TXT_LEN + 1) * HIST_LEN
        if prefix:
            sentence = prefix + [sep] + sentence

    length = len(sentence)
    sentence = sentence[-max_len:]

    # Beginning and end of text are added at the end and do not count towards max length
    return [bos] + sentence + [eos] + [pad] * (max_len - length)


def get_review_index(reviews, eos):
    ixs = [0]
    for w in reviews[1:]:
        if w == eos:
            ixs.append(ixs[-1] + 1)
        else:
            ixs.append(ixs[-1])


class Batchify:
    def __init__(self, data, word2idx, batch_size=128, seq_mode=0, shuffle=False):
        bos = word2idx[BOS_TOK]
        eos = word2idx[EOS_TOK]
        pad = word2idx[PAD_TOK]
        unk = word2idx[UNK_TOK]
        sep = word2idx[SEP_TOK]

        u, i, r, t, tix, f = [], [], [], [], [], []
        for x in data:
            u.append(x[U_COL])
            if seq_mode >= HIST_I_MODE:
                i.append([unk] * (HIST_LEN - len(x[HIST_I_COL])) + x[HIST_I_COL] + [x[I_COL]])
                if seq_mode == HIST_I_MODE + 1:
                    r.append(x[RAT_COL])
                else:
                    r.append([pad] * (HIST_LEN - len(x[HIST_I_COL])) + x[HIST_RAT_COL] + [x[RAT_COL]])
            else:
                i.append(x[I_COL])
                r.append(x[RAT_COL])
            if seq_mode >= HIST_REV_MODE:
                t.append(sentence_format(x[REV_COL], pad, bos, eos, sep, x[HIST_REV_COL]))
            else:
                t.append(sentence_format(x[REV_COL], pad, bos, eos, sep))
            f.append([x[FEAT_COL]])

        self.user = torch.tensor(u, dtype=torch.int64).contiguous()
        self.item = torch.tensor(i, dtype=torch.int64).contiguous()
        self.rating = torch.tensor(r, dtype=torch.float).contiguous()
        self.seq = torch.tensor(t, dtype=torch.int64).contiguous()
        self.feature = torch.tensor(f, dtype=torch.int64).contiguous()
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.sample_num = len(data)
        self.index_list = list(range(self.sample_num))
        self.total_step = int(math.ceil(self.sample_num / self.batch_size))
        self.step = 0

    def next_batch(self):
        if self.step == self.total_step:
            self.step = 0
            if self.shuffle:
                # Random seed was not fixed in the original code
                random.shuffle(self.index_list)

        start = self.step * self.batch_size
        offset = min(start + self.batch_size, self.sample_num)
        self.step += 1
        index = self.index_list[start:offset]
        user = self.user[index]  # (batch_size,)
        item = self.item[index]  # (batch_size, seq_len)
        rating = self.rating[index]
        seq = self.seq[index]  # (batch_size, seq_len)
        feature = self.feature[index]  # (batch_size, 1)
        return user, item, rating, seq, feature


def now_time():
    return '[' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f') + ']: '


def ids2tokens(ids, word2idx, idx2word):
    # TODO: Check the fact that this function changes in PEPLER and uses post-processing
    eos = word2idx[EOS_TOK]
    tokens = []
    for i in ids:
        if i == eos:
            break
        tokens.append(idx2word[i])
    return tokens


def plot_mask(mask):
    plt.imshow(mask, cmap='Greys', interpolation='nearest')
    plt.xticks(np.arange(0, mask.shape[1], 1))
    plt.yticks(np.arange(0, mask.shape[0], 1))
    plt.grid()
    plt.show()


def save_results(curr_res):
    filename = f'{DATA_MODE}_results.csv'
    is_new_file = not os.path.isfile(os.path.join(RES_PATH, filename))
    if not is_new_file:
        results = pd.read_csv(os.path.join(RES_PATH, filename))
    else:
        results = pd.DataFrame(columns=curr_res.keys())
    missing_cols = set(curr_res.keys()).difference(results.columns)
    for c in missing_cols:
        results[c] = np.nan
    missing_cols = set(results.columns).difference(curr_res.keys())
    for c in missing_cols:
        curr_res[c] = np.nan
    # results = results.append(pd.DataFrame().from_records([curr_res]))
    results = pd.concat((results, pd.DataFrame().from_records([curr_res])))
    results.to_csv(os.path.join(RES_PATH, filename), index=False)
