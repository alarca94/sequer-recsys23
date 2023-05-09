import os

BASE_PATH = os.getcwd()
DATA_MODE = 'review'
DATA_PATHS = {
    'amazon-sports': os.path.join(BASE_PATH, 'data/Amazon/SportsAndOutdoors'),
    'amazon-toys': os.path.join(BASE_PATH, 'data/Amazon/ToysAndGames'),
    'amazon-beauty': os.path.join(BASE_PATH, 'data/Amazon/Beauty'),
    'yelp': os.path.join(BASE_PATH, 'data/Yelp')
}
RES_PATH = os.path.join(BASE_PATH, 'results')
CKPT_PATH = os.path.join(BASE_PATH, 'checkpoints')
LOG_PATH = os.path.join(BASE_PATH, 'logs')
CONFIG_PATH = 'config'

PRED_F = 'generated.txt'

UNK_TOK = '<unk>'
PAD_TOK = '<pad>'
BOS_TOK = '<bos>'
EOS_TOK = '<eos>'
SEP_TOK = '<sep>'

FEAT_COL = 'feature'
ADJ_COL = 'adj'
SCO_COL = 'sco'
REV_COL = 'text'
U_COL = 'user'
I_COL = 'item'
RAT_COL = 'rating'
TIME_COL = 'timestamp'
HIST_I_COL = 'hist_' + I_COL
HIST_FEAT_COL = 'hist_' + FEAT_COL
HIST_ADJ_COL = 'hist_' + ADJ_COL
HIST_REV_COL = 'hist_' + REV_COL
HIST_RAT_COL = 'hist_' + RAT_COL
HIST_REVID_COL = 'hist_rev_id'
SEG_I_COL = I_COL + '_segment_ids'
SEG_REV_COL = REV_COL + '_segment_ids'
CONTEXT_COL = 'context'

RNG_SEED = 1111
ALL_SEEDS = [1111, 24, 53, 126, 675]

HIST_I_MODE = 1
HIST_REV_MODE = 3

HIST_LEN = 10
TXT_LEN = 15
VOCAB_SIZE = 5000

METRICS = ['RMSE', 'MAE', 'DIV', 'FCR', 'FMR', 'USR', 'USN', 'BLEU-1', 'BLEU-4', 'rouge_1/f_score', 'rouge_1/p_score',
           'rouge_1/r_score', 'rouge_2/f_score', 'rouge_2/p_score', 'rouge_2/r_score', 'rouge_l/f_score',
           'rouge_l/p_score', 'rouge_l/r_score']

MIN_METRICS = ['RMSE', 'MAE', 'DIV']
