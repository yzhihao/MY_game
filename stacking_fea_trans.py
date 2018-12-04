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
import datetime
import xgboost as xgb
import operator
adb_model = AdaBoostClassifier()
#gdbt_model = GradientBoostingClassifier()
et_model = ExtraTreesClassifier()
svc_model = SVC()
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
# ===================================处理交易详情=====================================#

operation_trn = pd.read_csv('..\operation_TRAIN_new.csv')
operation_test = pd.read_csv('..\operation_round1_new.csv')
transaction_trn = pd.read_csv('..\\transaction_TRAIN_new.csv')
transaction_test = pd.read_csv('..\\transaction_round1_new.csv')
tag_trn = pd.read_csv('..\\tag_TRAIN_new.csv')

geo_dict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
            '9': 9, 'b': 10, 'c':11, 'd':12, 'e':13, 'f':14, 'g':15, 'h':16,  'j':17,
            'k':18, 'm':19, 'n':20, 'p':21, 'q':22, 'r':23, 's':24, 't':25, 'u':26,
            'v':27, 'w':28, 'x':29, 'y':30, 'z':31,}

def split_geo(g, n):
    if pd.isna(g):
        return np.nan
    return geo_dict[g[n-1]]

transaction_trn = pd.merge(transaction_trn, tag_trn, how='left', on='UID')
transaction_test['Tag'] = -1
df = pd.concat([transaction_trn, transaction_test])

label_feature = ['channel', 'amt_src1', 'merchant', 'code1', 'code2', 'trans_type1', 'acc_id1',
                 'device_code1', 'device_code2', 'device_code3', 'device1', 'device2', 'mac1', 'ip1',
                 'amt_src2', 'acc_id2', 'acc_id3', 'trans_type2', 'market_code', 'ip1_sub']
for each in label_feature:
    df[each] = pd.factorize(df[each])[0]
df['hour'] = df['time'].apply(lambda x: datetime.datetime.strptime(x, '%H:%M:%S').hour)
del df['time']
for i in range(1, 5):
    df['geo_' + str(i)] = df['geo_code'].apply(lambda g: split_geo(g, i))
del df['geo_code']


def mis_impute(data):
    for i in data.columns:
        if data[i].dtype == "object":
            data[i] = data[i].fillna("other")
        elif (data[i].dtype == "int64" or data[i].dtype == "float64"):
            data[i] = data[i].fillna(data[i].mode()[0])  # 还有众数.mode()；中位数.median()
        else:
            pass
    return data
df = mis_impute(df)

X = np.array(df[df.Tag != -1].drop(['Tag', 'UID'], axis=1))
y = np.array(df[df.Tag != -1]['Tag'])
test = np.array(df[df.Tag == -1].drop(['Tag', 'UID'], axis=1))
train_x, train_y, test_x=X,y,test



def stad(X):
    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)
    rng = maxs - mins
    X = 1 - ((maxs - X) / rng)
    return X

#train_x=stad(train_x)
#test_x=stad(test_x)

from xgboost import XGBClassifier
import lightgbm

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

def lgx_best():
    bst = XGBClassifier(max_depth=2, learning_rate=0.1, silent=True, objective='binary:logistic')
    param_test = {  # 弱分类器的数目以及范围
        'n_estimators': list(range(20, 81, 10))
    }
    clf = GridSearchCV(estimator=bst, param_grid=param_test, n_jobs=4,scoring='roc_auc', cv=5)
    clf.fit(train_x, train_y)
    xbg_model = clf.best_estimator_
    return xbg_model


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



def get_stage1( x_train, y_train, x_test, n_folds=5):
    train_sets = []
    test_sets = []

    lgb_model=lgb_best()

    gdbt_model =gbdt_best()
    xbg_model=lgx_best()

    rf_model =rf_best()

    RL_model = LogisticRegression()

    for clf in [lgb_model, gdbt_model, rf_model,RL_model,xbg_model]:
        print('now is train: ' + str(clf.__class__.__name__))
        #train_set, test_set = get_stage1(clf, train_x, train_y, test_x)
        """
        这个函数是stacking的核心，使用交叉验证的方法得到次级训练集
        x_train, y_train, x_test 的值应该为numpy里面的数组类型 numpy.ndarray .
        如果输入为pandas的DataFrame类型则会把报错"""
        train_num, test_num = x_train.shape[0], x_test.shape[0]
        second_level_train_set = np.zeros((train_num,))
        second_level_test_set = np.zeros((test_num,))
        test_nfolds_sets = np.zeros((test_num, n_folds))
        kf = KFold(n_splits=n_folds)

        for i,(train_index, test_index) in enumerate(kf.split(x_train)):
            x_tra, y_tra = x_train[train_index], y_train[train_index]
            x_tst, y_tst =  x_train[test_index], y_train[test_index]
            clf.fit(x_tra, y_tra)
            second_level_train_set[test_index] = clf.predict_proba(x_tst)[:,1]
            print("eval_stage1:%f" % accuracy_score(y_tst, clf.predict(x_tst)))
            test_nfolds_sets[:, i] = clf.predict_proba(x_test)[:,1]

        #second_level_test_set[:] = np.array(pd.DataFrame(test_nfolds_sets).mode(axis=1).values).T
        second_level_test_set[:] = test_nfolds_sets.mean(axis=1)
        train_set, test_set =second_level_train_set, second_level_test_set
        train_sets.append(train_set)
        test_sets.append(test_set)
    return train_sets, test_sets


train_sets,test_sets=get_stage1(train_x, train_y, test_x)
meta_train = np.concatenate([result_set.reshape(-1, 1) for result_set in train_sets], axis=1)
meta_test = np.concatenate([y_test_set.reshape(-1, 1) for y_test_set in test_sets], axis=1)


train=pd.DataFrame(meta_train)
test=pd.DataFrame(meta_test)

test['UID']=transaction_test['UID']
train['UID']=transaction_trn['UID']

train.to_csv('stacking_fea_trans_pre.csv',index=False)
test.to_csv('stacking_fea_trans_test_pre.csv',index=False)

#pd.DataFrame(meta_train).to_csv('stacking_fea_trans_pre.csv',index=False)
#pd.DataFrame(meta_test).to_csv('stacking_fea_trans_test_pre.csv',index=False)

#df_predict=get_stage2(stage2_model,train_sets,train_y,test_sets)


