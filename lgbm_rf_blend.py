import pandas as pd
import os
import optuna
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import numpy as np
import sys
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from sklearn import linear_model
from sklearn import ensemble

def main(argv):
    np.random.seed(7)
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
    rf_pars = {'criterion': 'gini', 'n_estimators': 342, 'max_depth': 17, 'min_samples_split': 6, 'min_samples_leaf': 1}
    
    estimator = ensemble.RandomForestClassifier(**rf_pars)
    estimator.fit(train_df, y_train)

    rf_preds = estimator.predict(dev_df)
    rf_preds_proba = estimator.predict_proba(dev_df)
    #print(accuracy_score(y_dev, rf_preds))
    
    params={'num_leaves': 560, 'min_data_in_leaf': 189, 'min_child_weight': 0.0024699349965224245, 'max_depth': 26, 'bagging_fraction': 0.9753023991207269, \
'feature_fraction': 0.7137858794613544,   'lambda_l1': 0.5330703499296066, 'lambda_l2': 0.12028486914560135}

    lgb_train = lgb.Dataset(data=train_df.astype('float32'), label=y_train.astype('float32'))
    lgb_valid = lgb.Dataset(data=dev_df.astype('float32'), label=y_dev.astype('float32'))

    estimator = lgb.train(params, lgb_train, 800, verbose_eval=0)

    lgbm_probs = estimator.predict(dev_df)
    print(accuracy_score(y_dev,(((lgbm_probs * 0.822 + rf_preds_proba[:, 1] * 0.8125) \
                       / (0.822 + 0.8125)) > 0.5).astype(int)))

if __name__ == '__main__':
    main(sys.argv)
    
