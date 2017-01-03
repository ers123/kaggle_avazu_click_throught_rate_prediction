'''
    This file is the initial Avazu Click Through Rate Prediction pipeline script that serves as a starter.
    It has all the components necessary for a ml pipeline(data, model, eval(trn,vld), submission) with no tuning.
    Cross-validation scheme uses 9:1 stratified shuffle split x 1 time

    This script has result as below:
        TRN logloss : 0.4175489
        VLD logloss : 0.4175405
        Public  LB  :
        Private LB  : 0.4400804

    Data : 'leak'

    Model :

    top finisher's code : https://www.kaggle.com/c/avazu-ctr-prediction/forums/t/12471/what-was-your-best-single-model
    fast_solution (hash with SGD) : https://www.kaggle.com/c/avazu-ctr-prediction/forums/t/10745/beating-the-benchmark
    Winner's solutions : https://www.kaggle.com/c/avazu-ctr-prediction/forums/t/12460/congrats-to-the-winners

'''

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import time
import os

np.random.seed(777)


def main():

    print('=' * 50)
    print('# Begin : Avazu Click Through Rate Prediction _ bare_minimum')

    ##################################################################################################################
    ### Loading data
    ##################################################################################################################

    print('# Loading data..')
    # load train
    f = open('../root_input/train', 'r')
    cols = f.readline().strip().split(',')

    # pick number of rows for trn data
    lines = 1000000

    trn = []
    count = 0
    while True:
        count += 1
        if count == lines:
            break

        line = f.readline().strip()
        arr = line.split(',')

        if line == '':
            break

        trn.append(arr)

        if count % 10000 == 0:
            print('# {} lines..'.format(count))

    trn = pd.DataFrame(trn, columns=cols)

    tst = pd.read_csv('../root_input/test')

    ##################################################################################################################
    ### Feature Engineering
    ##################################################################################################################

    print('# Feature Engineering')
    features = []

    # add 3 features
    features.append('site_id')
    features.append('C14')
    features.append('device_type')

    # change to integer
    lb = LabelEncoder()
    lb.fit(pd.concat([trn['site_id'], tst['site_id']], axis=0))
    trn['site_id'] = lb.transform(trn['site_id'])
    tst['site_id'] = lb.transform(tst['site_id'])

    trn['C14'] = trn['C14'].astype(int)
    trn['device_type'] = trn['device_type'].astype(int)


    # prepare data
    x = trn.as_matrix(columns=features)
    y = trn['click'].values.astype(int)

    print('# x {} y {}'.format(x.shape, y.shape))
    del trn

    ##################################################################################################################
    ### Cross Validation
    ##################################################################################################################

    print('# Cross validation..')
    # 9:1 split for cross-validation
    model = RandomForestClassifier(max_depth=3, n_jobs=-1, random_state=777)

    trn_scores = []
    vld_scores = []
    n_splits = 1
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.1, random_state=777)
    for i, (t_ind, v_ind) in enumerate(sss.split(x, y)):
        print('# Iter {} / {}'.format(i+1, n_splits))
        x_trn = np.asarray(x)[t_ind]
        x_vld = np.asarray(x)[v_ind]
        y_trn = np.asarray(y)[t_ind]
        y_vld = np.asarray(y)[v_ind]

        model.fit(x_trn, y_trn)

        score = log_loss(y_trn, model.predict_proba(x_trn))
        trn_scores.append(score)

        score = log_loss(y_vld, model.predict_proba(x_vld))
        vld_scores.append(score)

    print('# TRN logloss: {}'.format(np.mean(trn_scores)))
    print('# VLD logloss: {}'.format(np.mean(vld_scores)))

    ##################################################################################################################
    ### Model Fit
    ##################################################################################################################

    print('# ReFit model to all data..')
    model.fit(x, y)

    ##################################################################################################################
    ### Prediction
    ##################################################################################################################

    print('# Making predictions on test..')
    test_prediction = model.predict_proba(tst.as_matrix(columns=features))

    ##################################################################################################################
    ### Submission
    ##################################################################################################################

    print('# Generating a submission..')
    result = pd.DataFrame(test_prediction[:, 1], columns=['click'])
    result['id'] = tst['id']

    now = datetime.now()
    if not os.path.exists('./output'):
        os.mkdir('./output')
    sub_file = './output/submission_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    result.to_csv(sub_file, index=False)

    print('# DONE!')


if __name__ == '__main__':
    start = time.time()
    main()
    print('finished ({:.2f} sec elapsed)'.format(time.time() - start))
    # 97 sec

