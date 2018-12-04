import numpy as np
import pandas as pd
import second_statge.evaluate as evaluate
import lightgbm as lgb
import re
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                                  GradientBoostingClassifier, ExtraTreesClassifier)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import xgboost
import datetime
import operator

import xgboost as xgb
def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()


def xgb_select(X,y):
#    train = pd.read_csv("../input/train.csv")
 #   cat_sel = [n for n in train.columns if n.startswith('cat')]  # 类别特征数值化
 #   for column in cat_sel:
  #      train[column] = pd.factorize(train[column].values, sort=True)[0] + 1

    params = {
        'min_child_weight': 100,
        'eta': 0.02,
        'colsample_bytree': 0.7,
        'max_depth': 12,
        'subsample': 0.7,
        'alpha': 1,
        'gamma': 1,
        'silent': 1,
        'verbose_eval': True,
        'seed': 12
    }
    rounds = 10
    #y = train['loss']
    #X = train.drop(['loss', 'id'], 1)

    xgtrain = xgb.DMatrix(X, label=y)
    bst = xgb.train(params, xgtrain, num_boost_round=rounds)

    features = [x for x in X.columns if x not in ['id', 'loss']]
    ceate_feature_map(features)

    importance = bst.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=operator.itemgetter(1))
    #feature_num=30
    #importance = importance[:feature_num]
    feature_list=np.array(importance)[:,0]
    delete_list=list(set(X.columns).difference(set(feature_list)))
    X.drop(delete_list, axis=1,inplace=True)

    return X,delete_list

def tpr_weight_funtion(y_true,y_predict):
    d = pd.DataFrame()
    d['prob'] = list(y_predict)
    d['y'] = list(y_true)
    d = d.sort_values(['prob'], ascending=[0])
    y = d.y
    PosAll = pd.Series(y).value_counts()[1]
    NegAll = pd.Series(y).value_counts()[0]
    pCumsum = d['y'].cumsum()
    nCumsum = np.arange(len(y)) - pCumsum + 1
    pCumsumPer = pCumsum / PosAll
    nCumsumPer = nCumsum / NegAll
    TR1 = pCumsumPer[abs(nCumsumPer-0.001).idxmin()]
    TR2 = pCumsumPer[abs(nCumsumPer-0.005).idxmin()]
    TR3 = pCumsumPer[abs(nCumsumPer-0.01).idxmin()]
    return 'TC_AUC',0.4 * TR1 + 0.3 * TR2 + 0.3 * TR3,True


train = pd.read_csv('train.csv')
#train = train.sample(frac=1)
y = pd.DataFrame()
y['UID'] = train['UID']
y['Tag'] = train['Tag']
test = pd.read_csv('test.csv')
label = np.array(y['Tag'])

#train, delete_list = xgb_select(train.drop(['UID', 'Tag'], axis=1), label)
#test.drop(delete_list, axis=1, inplace=True)
train = np.array(train.drop(['UID', 'Tag'], axis=1))

test_id = test['UID']
test = np.array(test.drop(['UID', 'Tag'], axis=1))

train_x, train_y, test_x = train, label, test

x_train, y_train, x_test=train_x, train_y, test_x


from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
def lgx_best():
    bst = XGBClassifier(max_depth=2, learning_rate=0.1, silent=True, objective='binary:logistic')
    param_test = {  # 弱分类器的数目以及范围
        'n_estimators': list(range(20, 81, 10))
    }
    clf = GridSearchCV(estimator=bst, param_grid=param_test, n_jobs=4,scoring='roc_auc', cv=5)
    clf.fit(train_x, train_y)
    xbg_model = clf.best_estimator_
    return xbg_model



def rf_best():
    rf_model = RandomForestClassifier()
    grid_rf_model = GridSearchCV(rf_model, param_grid={'n_estimators': range(10, 100, 10)}, n_jobs=4, scoring='roc_auc',
                                 cv=5)
    grid_rf = grid_rf_model.fit(train_x, train_y)
    rf_model = grid_rf.best_estimator_
    return rf_model


def gbdt_best():
    grid_gdbt_model = GridSearchCV(
        GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500, min_samples_leaf=50,
                                   max_depth=8, max_features='sqrt', subsample=0.8, random_state=10),
        param_grid={'n_estimators': range(20, 81, 10)}, n_jobs=4, scoring='roc_auc', cv=5)
    grid_gdbt = grid_gdbt_model.fit(train_x, train_y)
    gdbt_model = grid_gdbt.best_estimator_
    return gdbt_model


import lightgbm
def lgb_best():
    grid_lgb_model = GridSearchCV(
        lightgbm.LGBMClassifier(boosting_type='gbdt', objective='binary', metric='auc', is_unbalance='True',
                                lambda_l1=0.1, verbose=1),
        param_grid={
            'max_depth': range(3, 8, 2), 'num_leaves': range(50, 170, 30)
        }, n_jobs=4, scoring='roc_auc', cv=5)

    grid_lgb = grid_lgb_model.fit(train_x, train_y)
    lgb_model = grid_lgb.best_estimator_
    return lgb_model




#xbg_model = lgx_best()
lgb_mode=lgb_best()
single_model=lgb_mode


n_folds=5
kf = KFold(n_splits=n_folds)

train_num, test_num = x_train.shape[0], x_test.shape[0]
second_level_train_set = np.zeros((train_num,))
second_level_test_set = np.zeros((test_num,))
test_nfolds_sets = np.zeros((test_num, n_folds))


oof_preds = np.zeros(train.shape[0])
oof_preds1=np.zeros(train.shape[0])
print('now is train: ' + str(single_model.__class__.__name__))
for i,(train_index, test_index) in enumerate(kf.split(x_train)):
    x_tra, y_tra = x_train[train_index], y_train[train_index]
    x_tst, y_tst =  x_train[test_index], y_train[test_index]

    single_model.fit(x_tra, y_tra)
    oof_preds[test_index] = single_model.predict_proba(x_tst)[:, 1]
    oof_preds1[test_index] = single_model.predict(x_tst)

    second_level_train_set[test_index] = single_model.predict_proba(x_tst)[:,1]
    print("eval:%f" % accuracy_score(y_tst, single_model.predict(x_tst)))
    print("eval_score:%f" % evaluate.tpr_weight_funtion(y_tst, single_model.predict_proba(x_tst)[:, 1]))
    test_nfolds_sets[:, i] = single_model.predict_proba(x_test)[:,1]


print("eval_all:%f" % accuracy_score(train_y, oof_preds1))
print("eval_score_all:%f" % evaluate.tpr_weight_funtion(train_y,oof_preds))
pd.DataFrame(oof_preds).to_csv('result_singe_TRAIN.csv',index=False)

second_level_test_set[:] = test_nfolds_sets.mean(axis=1)

sub = pd.read_csv('..\sub.csv')
sub['Tag'] = second_level_test_set
sub.to_csv('result_singe.csv',index=False)