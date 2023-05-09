import os
import time
import pandas as pd
import numpy as np

from tqdm import tqdm
from utils.constants import *


# Download Amazon datasets: wget https://jmcauley.ucsd.edu/data/amazon_v2/categoryFiles/{Category}.json.gz
# Download Amazon metadata: https://jmcauley.ucsd.edu/data/amazon_v2/metaFiles2/meta_Movies_and_TV.json.gz
# Download P5 datasets: https://drive.google.com/file/d/1qGxgmx7G_WB7JE4Cn_bEcZ_o_NAJLE3G/view?usp=sharing
def add_hist_feats(data, max_seq_len=20):
    print('### ADDING HISTORY FEATURES TO PETER DATA ###')
    start_time = time.time()
    hist_df = []
    data.sort_values(TIME_COL, inplace=True)
    for uid, udata in tqdm(data.groupby(U_COL)):
        for t in range(udata.shape[0]):
            record = udata.iloc[t].values.tolist()
            record += udata.iloc[max(0, t - max_seq_len):t][[I_COL, FEAT_COL, ADJ_COL, REV_COL, RAT_COL]].values.T.tolist()
            hist_df.append(record)
    data = pd.DataFrame(hist_df, columns=data.columns.tolist() +
                                               [HIST_I_COL, HIST_FEAT_COL, HIST_ADJ_COL, HIST_REV_COL, HIST_RAT_COL])
    print(f'Ellapsed time: {time.time() - start_time:.2f}s')
    print(f'Existing NaNs: {data.isna().sum().sum()}')
    print(f'Total records: {data.shape[0]}')
    print(f'Columns: {data.columns.tolist()}')
    return data


def iterative_kcore_filter(df, kcore, verbose=1):
    def kcore_filter(d, col):
        return d[d[col].map(df[col].value_counts()).gt(kcore-1)]
    copy_df = df.copy()
    prev_sz = -1
    if verbose == 1:
        print(f'Starting Iterative K-Core item and user filtering with K = {kcore}...')
        print(f'Initial number of interactions: {df.shape[0]}')
        print(f'Initial number of users: {df[U_COL].nunique()}')
        print(f'Initial number of items: {df[I_COL].nunique()}')
    while prev_sz != copy_df.shape[0]:
        # Filter by user profile size
        prev_sz = copy_df.shape[0]
        copy_df = kcore_filter(copy_df, U_COL)
        copy_df = kcore_filter(copy_df, I_COL)

    if verbose == 1:
        print(f'Final number of interactions: {copy_df.shape[0]}')
        print(f'Final number of users: {copy_df[U_COL].nunique()}')
        print(f'Final number of items: {copy_df[I_COL].nunique()}')
    return copy_df


def split_data(datasets, test_ratio=0.1):
    val_ratio = test_ratio

    for dataset in datasets:
        print(f'Processing dataset {dataset}')
        data = pd.read_pickle(os.path.join(DATA_PATHS[dataset], 'reviews_new.pickle'))
        data = pd.DataFrame.from_records(data)

        data = data.sort_values(by=[U_COL, TIME_COL]).reset_index(drop=True)
        pd.to_pickle(data.to_dict(orient='records'), os.path.join(DATA_PATHS[dataset], 'reviews_new.pickle'))
        ucounts = data.groupby(U_COL).size().values  # data[U_COL].value_counts().values
        print(f'Min. user count: {ucounts.min()}. Max. user count: {ucounts.max()}')
        uoffsets = ucounts.cumsum()
        split_ixs = np.zeros((data.shape[0], ), dtype=int)
        if isinstance(test_ratio, float):
            assert isinstance(val_ratio, float)
            assert test_ratio < 1.0
            tst_start_ixs = uoffsets - np.maximum(ucounts * test_ratio, 1).astype(int)
            val_start_ixs = tst_start_ixs - np.maximum(ucounts * val_ratio, 1).astype(int)
            # tst_start_ixs = uoffsets - (ucounts * test_ratio).astype(int)
            # val_start_ixs = tst_start_ixs - (ucounts * val_ratio).astype(int)
        elif isinstance(test_ratio, int):
            assert isinstance(val_ratio, int)
            assert all(ucounts > (test_ratio + val_ratio))
            tst_start_ixs = uoffsets - test_ratio
            val_start_ixs = tst_start_ixs - val_ratio
        else:
            raise TypeError('test_ratio is neither int nor float')
        for vix, tix, offset in zip(val_start_ixs, tst_start_ixs, uoffsets):
            split_ixs[tix:offset] = 2
            split_ixs[vix:tix] = 1

        if not os.path.exists(os.path.join(DATA_PATHS[dataset], '0')):
            os.mkdir(os.path.join(DATA_PATHS[dataset], '0'))
        np.save(os.path.join(DATA_PATHS[dataset], DATA_MODE, '0', 'train'),
                np.argwhere(split_ixs == 0).squeeze())

        np.save(os.path.join(DATA_PATHS[dataset], DATA_MODE, '0', 'validation'),
                np.argwhere(split_ixs == 1).squeeze())

        np.save(os.path.join(DATA_PATHS[dataset], DATA_MODE, '0', 'test'),
                np.argwhere(split_ixs == 2).squeeze())
        trn = data.iloc[np.argwhere(split_ixs == 0).squeeze()]
        val = data.iloc[np.argwhere(split_ixs == 1).squeeze()]
        tst = data.iloc[np.argwhere(split_ixs == 2).squeeze()]
        trn_u = set(trn.user.unique())
        val_u = set(val.user.unique())
        tst_u = set(tst.user.unique())
        assert len(trn_u.difference(val_u)) == 0
        assert len(trn_u.difference(tst_u)) == 0
        trn_i = trn.item.unique()
        val_i = val.item.unique()
        tst_i = tst.item.unique()
        val_i_diff = set(val_i).difference(trn_i)
        tst_i_diff = set(tst_i).difference(trn_i)
        print(f"Number of items in valid but not in train: {len(val_i_diff)}")
        print(f"Number of items in test but not in train: {len(tst_i_diff)}")
        print(f"Percentage of missing item samples (Val): {sum(val.item.isin(val_i_diff)) / val.shape[0]:.4f}")
        print(f"Percentage of missing item samples (Test): {sum(tst.item.isin(tst_i_diff)) / tst.shape[0]:.4f}")
        print(f"Ratio of non-empty feature in training: {sum(trn[FEAT_COL] != '') / trn.shape[0]:.4f}")
        print(f"Ratio of non-empty feature in valid: {sum(val[FEAT_COL] != '') / val.shape[0]:.4f}")
        print(f"Ratio of non-empty feature in test: {sum(tst[FEAT_COL] != '') / tst.shape[0]:.4f}")
        print(f'Final ratios for dataset {dataset} are: {np.unique(split_ixs, return_counts=True)[1] / len(split_ixs)}')

    print('Finished!')


def p5_data_to_peter():
    import pickle5 as pickle

    datasets = ['amazon', 'yelp']
    sub_datasets = {'amazon': ['sports', 'beauty', 'toys']}
    col_map = {'reviewerID': U_COL, 'asin': I_COL, 'reviewText': REV_COL, 'unixReviewTime': TIME_COL,
               'overall': RAT_COL, 'feature': FEAT_COL}
    for dataset in datasets:
        for folder in sub_datasets.get(dataset, [dataset]):
            print(f'Processing {folder}...')
            with open(os.path.join(BASE_PATH, 'data', folder, 'review_splits.pkl'), 'rb') as f:
                data = pickle.load(f)

            data = data['train'] + data['val'] + data['test']
            print(f'Orig. number of records: {len(data)}')

            data = pd.DataFrame.from_records(data)
            data.rename(col_map, axis=1, inplace=True)
            data = data[col_map.values()]
            data[FEAT_COL] = data[FEAT_COL].fillna('')
            data[ADJ_COL] = ''

            # Don't filter P5 datasets with k-core as it is already filtered
            # data = iterative_kcore_filter(data, kcore=5, verbose=1)
            # data = data[~(data[FEAT_COL] == '')]

            # Add sequential fields
            data = add_hist_feats(data, max_seq_len=20)
            if dataset == 'amazon':
                folder = 'amazon-' + folder
            if not os.path.exists(DATA_PATHS[folder]):
                os.mkdir(DATA_PATHS[folder])
                if not os.path.exists(os.path.join(DATA_PATHS[dataset], DATA_MODE)):
                    os.mkdir(os.path.join(DATA_PATHS[folder], DATA_MODE))
            pd.to_pickle(data.to_dict(orient='records'), os.path.join(DATA_PATHS[folder], DATA_MODE, 'reviews_new.pickle'))

    print('Finished')


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('max_colwidth', None)
    pd.set_option('expand_frame_repr', False)

    p5_data_to_peter()
    split_data(['amazon-sports', 'amazon-toys', 'amazon-beauty', 'yelp'])
