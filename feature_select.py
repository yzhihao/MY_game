from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
import numpy as np
import pandas as pd
import lightgbm as lgb
import datetime
import xgboost as xgb
import matplotlib.pyplot as plt
import operator

#-------------------------xbgoost筛选特征--------------
def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()

def xgb_select(X,y):
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

    xgtrain = xgb.DMatrix(X, label=y)
    bst = xgb.train(params, xgtrain, num_boost_round=rounds)

    features = [x for x in X.columns if x not in ['id', 'loss']]
    ceate_feature_map(features)

    importance = bst.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=operator.itemgetter(1))

    #可视化特征重要性
    '''
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    plt.figure()
    df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
    plt.title('XGBoost Feature Importance')
    plt.xlabel('relative importance')
    plt.show()
    '''


    #feature_num=30
    #importance = importance[:feature_num]
    feature_list=np.array(importance)[:,0]
    delete_list=list(set(X.columns).difference(set(feature_list)))
    #X.drop(delete_list, axis=1,inplace=True)

    return feature_list

#-------------------------lightgbm筛选特征--------------
import lightgbm
from sklearn.model_selection import GridSearchCV
def lgb_select(X,y):
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        # 'max_depth': 3,
        'metric': 'auc',
        'num_leaves': 31,
        'learning_rate': 0.02,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 1,
        'is_unbalance': True,
        'lambda_l1': 0.1
    }
    train_in = int((X.shape[0] / 5) * 4)
    X_train, X_test, y_train, y_test = X[:train_in], X[train_in:], y[:train_in], y[train_in:]

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=2000,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=30,
                    verbose_eval=100,
                    )


    feature_names = [x for x in X.columns if x not in ['id', 'loss']]

    print(pd.DataFrame({
        'column': feature_names,
        'importance': gbm.feature_importance(),
    }).sort_values(by='importance',ascending=False))
    return   pd.DataFrame({
        'column': feature_names,
        'importance': gbm.feature_importance(),
    }).sort_values(by='importance',ascending=False)


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
#train = train.sample(frac=1)
y = pd.DataFrame()
y['UID'] = train['UID']
y['Tag'] = train['Tag']
label = np.array(y['Tag'])

train_se = train.drop(['UID', 'Tag'], axis=1)
feature_list = xgb_select(train_se, label)
len_feature_list=len(feature_list)

feature_importance=lgb_select(train_se,label)
feature_importance=feature_importance[:len_feature_list]
get_feature=set(list(feature_importance.column)+list(feature_list))
delete_list=list(set(train_se.columns).difference(get_feature))
test.drop(delete_list, axis=1, inplace=True)
train.drop(delete_list, axis=1, inplace=True)


train.to_csv('train.csv',index=False)
test.to_csv('test.csv',index=False)