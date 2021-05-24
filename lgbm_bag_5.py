import pandas as pd
import os
import optuna
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import numpy as np
import sys

def main(argv):
    train_df = pd.read_csv(f'{argv[1]}/final_log_df_train.csv')
    dev_df = pd.read_csv(f'{argv[1]}/final_log_df_dev.csv')

    train_column_rename = {}
    for col in train_df.columns:
        new_column_name = col.replace('train', '')
        train_column_rename[col] = new_column_name
    train_df.rename(columns=train_column_rename, inplace=True)
    train_df['tag'] = train_df['tag'].map({'T': 1, 'F':0})
    dev_df['tag'] = dev_df['tag'].map({'T': 1, 'F':0})
    dev_column_rename = {}
    for col in dev_df.columns:
        new_column_name = col.replace('dev', '')
        dev_column_rename[col] = new_column_name
    dev_df.rename(columns=dev_column_rename, inplace=True)
    y_train = train_df.tag
    train_df.drop(columns='tag', inplace=True)

    y_dev = dev_df.tag
    dev_df.drop(columns='tag', inplace=True)
    classifiers = []

    n_splits = 5
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, valid_idx in cv.split(train_df, y_train):
        params={'num_leaves': 560, 'min_data_in_leaf': 189, 'min_child_weight': 0.0024699349965224245, \
            'max_depth': 26, 'bagging_fraction': 0.9753023991207269, \
            'feature_fraction': 0.7137858794613544, 'lambda_l1': 0.5330703499296066, \
            'lambda_l2': 0.12028486914560135, \
           'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt', \
    'boost_from_average': True, 'num_threads': 4, 'random_state': 42} # minimize
        x_train_train = train_df.iloc[train_idx]
        y_train_train = y_train.iloc[train_idx]
        x_train_valid = train_df.iloc[valid_idx]
        y_train_valid = y_train.iloc[valid_idx]

        lgb_train = lgb.Dataset(data=x_train_train.astype('float32'), label=y_train_train.astype('float32'))
        lgb_valid = lgb.Dataset(data=x_train_valid.astype('float32'), label=y_train_valid.astype('float32'))

        estimator = lgb.train(params, lgb_train, 10000, valid_sets=lgb_valid,\
                          early_stopping_rounds=25, verbose_eval=0)
        classifiers.append(estimator)

    preds = []
    for clf in classifiers:
        y_part = estimator.predict(dev_df, num_iteration=estimator.best_iteration)
        preds.append(y_part)
    preds = np.mean(preds, axis=0)

    check = (preds > 0.5).astype(int)
    
    print(accuracy_score(y_dev, check))

if __name__ == '__main__':
    main(sys.argv)
    
