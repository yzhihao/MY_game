import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest  # 导入SelectKBest库
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import f_regression
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import lightgbm as lgb
import datetime
import xgboost as xgb
import matplotlib.pyplot as plt
import operator

#data = pd.read_csv('..\data\\transform_data\\transform_train_4.csv',index_col=None,parse_dates = True) #pd.read_csv默认生成DataFrame对象
iris = load_iris()
print(iris.data.shape)#查看数据
train_x,train_y=iris.data, iris.target
feature_names=iris.feature_names

'''
data1 = data0.iloc[:-227,-1].values[:,np.newaxis]
data2 =data0.iloc[:,2:8].values  #取第2-9列
data = np.concatenate((data1,data2),axis=1)

feature_names=data.columns.values.tolist()
del feature_names[8],feature_names[0]
train_x1=data.values[:1000,1:8]
train_x2=data.values[:1000,9:]
train_x = np.concatenate((train_x1,train_x2),axis=1)
train_y=data.values[:1000,8]
'''

#--------随机森林特征选择
def rf_select():
    from sklearn.model_selection import cross_val_score, ShuffleSplit
    from sklearn.ensemble import RandomForestRegressor

    names = feature_names

    rf = RandomForestRegressor(n_estimators=100, max_depth=4)
    scores = []
    for i in range(train_x.shape[1]):
        print(str(i)+' '+str(train_x.shape[1]))
        score = cross_val_score(rf, train_x[:, i:i+1], train_y, scoring="r2",
                              cv=ShuffleSplit(len(train_x), 3, .3))
        scores.append((round(np.mean(score), 3),names[i]))
    print(sorted(scores, reverse=True))


#--------------皮尔逊系数
def Pearson_select():
    import numpy as np
    from scipy.stats import pearsonr
    np.random.seed(0)
    size = 300
    x = np.random.normal(0, 1, size)
    print
    "Lower noise", pearsonr(x, x + np.random.normal(0, 1, size))
    print
    "Higher noise", pearsonr(x, x + np.random.normal(0, 10, size))

#----------------------F值回归特征选择
def f_regression_select():
    names = feature_names
    test = SelectKBest(score_func=f_regression, k=2)
    fit = test.fit(train_x, train_y)
    print(fit.scores_)
    scores=[]
    for i in range(len(fit.scores_)):
        scores.append((fit.scores_[i], names[i]))
    print(sorted(scores, reverse=True))
    #features = fit.transform(X)  # 返回选择特征后的数据

#-----------------------互信息回归特征选择
def mutual_info_regression_select():
    names = feature_names
    test = SelectKBest(score_func=mutual_info_regression, k=2)
    fit = test.fit(train_x, train_y)
    print(fit.scores_)
    scores=[]
    for i in range(len(fit.scores_)):
        scores.append((fit.scores_[i], names[i]))
    print(sorted(scores, reverse=True))
    #features = fit.transform(X)  # 返回选择特征后的数据



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
    X.drop(delete_list, axis=1,inplace=True)

    return np.array(X),delete_list

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
    }).sort_values(by='importance'))


if __name__ == '__main__':
    rf_select()
