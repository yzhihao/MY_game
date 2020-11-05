import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.metrics import accuracy_score

import xgboost
class XGBClassifier():
    def __init__(self):
        """set parameters"""
        self.num_rounds = 1000
        self.xgb=xgboost
        self.early_stopping_rounds = 15
        self.params = {
            'objective': 'multi:softmax',
            'num_class':3,
            'eta': 0.1,
            'max_depth': 8,
            'eval_metric': 'mlogloss',
            'seed': 0,
            'silent': 0
        }

    def fit(self, x_train, y_train,x_val, y_val):
        print('train with xgb model')
        xgbtrain = self.xgb.DMatrix(x_train, y_train)
        xgbval = self.xgb.DMatrix(x_val, y_val)
        watchlist = [(xgbtrain, 'train'), (xgbval, 'val')]
        model=self.xgb.train(self.params,
                          xgbtrain,
                          self.num_rounds,
                           watchlist,
                            early_stopping_rounds = self.early_stopping_rounds)
        return model

    def predict(self,model, x_test):
        print('test with xgb model')
        xgbtest = self.xgb.DMatrix(x_test)
        return model.predict(xgbtest)


import lightgbm
class LGBClassifier():
    def __init__(self):
        self.num_boost_round = 2000
        self.lgb=lightgbm
        self.early_stopping_rounds = 15
        self.params = {
            'task': 'train',
            'boosting_type': 'dart',
            'objective': 'multiclass',
           # 'application':'num_class',
            'metric': 'multi_logloss',
            'num_leaves': 80,
            'learning_rate': 0.05,
            'num_class':3,
            # 'scale_pos_weight': 1.5,
            'feature_fraction': 0.5,
            'bagging_fraction': 1,
            'bagging_freq': 5,
            'max_bin': 300,
            'is_unbalance': True,
            'lambda_l2': 5.0,
            'verbose': -1
        }

    def fit(self, x_train, y_train,x_val, y_val):
        print('train with lgb model')
        lgbtrain = self.lgb.Dataset(x_train, y_train)
        lgbval = self.lgb.Dataset(x_val, y_val)
        model=self.lgb.train(self.params,
                          lgbtrain,
                          valid_sets=lgbval,
                          verbose_eval=self.num_boost_round,
                          num_boost_round=self.num_boost_round,
                         early_stopping_rounds = self.early_stopping_rounds)
        return model
    def predict(self,model, x_test):
        print('test with lgb model')
        return np.argmax(model.predict(x_test, num_iteration=model.best_iteration))


def get_stage1(clf, x_train, y_train, x_test, n_folds=5):
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
        if clf.__class__.__name__=='XGBClassifier' or clf.__class__.__name__=='LGBClassifier' :
            model=clf.fit(x_tra, y_tra,x_tst, y_tst)
            second_level_train_set[test_index] = clf.predict(model,x_tst)
            test_nfolds_sets[:, i] = clf.predict(model,x_test)
        else:
            clf.fit(x_tra, y_tra)
            second_level_train_set[test_index] = clf.predict(x_tst)
            print("eval_stage1:%f" % accuracy_score(y_tst, clf.predict(x_tst)))
            test_nfolds_sets[:, i] = clf.predict(x_test)

    second_level_test_set[:] = test_nfolds_sets.mean(axis=1)
    return second_level_train_set, second_level_test_set


def get_stage2(stage2_model,train_sets,train_y,test_sets,n_folds=5 ):
    xx_submit = []
    meta_train = np.concatenate([result_set.reshape(-1, 1) for result_set in train_sets], axis=1)
    meta_test = np.concatenate([y_test_set.reshape(-1, 1) for y_test_set in test_sets], axis=1)
    kf = KFold(n_splits=n_folds)

    for i,(train_index, test_index) in enumerate(kf.split(meta_train)):
        x_tra, y_tra = meta_train[train_index], train_y[train_index]
        x_tst, y_tst =  meta_train[test_index], train_y[test_index]
        if stage2_model.__class__.__name__=='XGBClassifier' or stage2_model.__class__.__name__=='LGBClassifier' :
            model = stage2_model.fit(x_tra, y_tra, x_tst, y_tst)
            xx_submit.append(stage2_model.predict(model,meta_test))
            #print("eval:%f" % evaluate.tpr_weight_funtion(y_tst, stage2_model.predict(model,x_tst)))
        else:
            #stage2_model=stage2_model
            stage2_model.fit(x_tra, y_tra)
            xx_submit.append(stage2_model.predict(meta_test))
            #print("eval:%f" % evaluate.tpr_weight_funtion(y_tst, stage2_model.predict(x_tst)))

    result=pd.DataFrame(xx_submit).mode(axis=0).values.T


    '''
    s = 0
    for each in xx_submit:
        s += each
    result=list(s / n_folds)
    '''
    return result

    '''
    # 使用决策树作为我们的次级分类器
    #from sklearn.tree import DecisionTreeClassifier
    dt_model = stage2_model
    dt_model.fit(meta_train, train_y)
    df_predict = dt_model.predict(meta_test)
    return df_predict
    '''

if __name__=='__main__':
    #我们这里使用5个分类算法，为了体现stacking的思想，就不加参数了
    from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                                  GradientBoostingClassifier, ExtraTreesClassifier)
    from sklearn.svm import SVC
    xgb=XGBClassifier()
    lgb=LGBClassifier()
    #rf_model = RandomForestClassifier()
    #adb_model = AdaBoostClassifier()
    gdbc_model = GradientBoostingClassifier()
    et_model = ExtraTreesClassifier()
    svc_model = SVC()
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    iris = load_iris()
    train_x, test_x, train_y, test_y = train_test_split(iris.data, iris.target, test_size=0.2)
    train_sets = []
    test_sets = []

    for clf in [ gdbc_model, et_model, svc_model]:
        print('now is train: ' + str(clf.__class__.__name__))
        train_set, test_set = get_stage1(clf, train_x, train_y, test_x)
        train_sets.append(train_set)
        test_sets.append(test_set)

    stage2_model=DecisionTreeClassifier()
    df_predict=get_stage2(stage2_model,train_sets,train_y,test_sets)
    #在这里我们使用train_test_split来人为的制造一些数据

    print(accuracy_score(test_y, df_predict))