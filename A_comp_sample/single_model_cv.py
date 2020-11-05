from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                                  GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
def lgx_best(train_x,train_y):
    bst = XGBClassifier(max_depth=2, learning_rate=0.1, silent=True, objective='binary:logistic')
    param_test = {  # 弱分类器的数目以及范围
        'n_estimators': list(range(20, 81, 10))
    }
    clf = GridSearchCV(estimator=bst, param_grid=param_test, n_jobs=4,scoring='roc_auc', cv=5)
    clf.fit(train_x, train_y)
    xbg_model = clf.best_estimator_
    return xbg_model



def rf_best(train_x,train_y):
    rf_model = RandomForestClassifier()
    grid_rf_model = GridSearchCV(rf_model, param_grid={'n_estimators': range(10, 100, 10)}, n_jobs=4, scoring='roc_auc',
                                 cv=5)
    grid_rf = grid_rf_model.fit(train_x, train_y)
    rf_model = grid_rf.best_estimator_
    return rf_model


def gbdt_best(train_x,train_y):
    grid_gdbt_model = GridSearchCV(
        GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500, min_samples_leaf=50,
                                   max_depth=8, max_features='sqrt', subsample=0.8, random_state=10),
        param_grid={'n_estimators': range(20, 81, 10)}, n_jobs=4, scoring='roc_auc', cv=5)
    grid_gdbt = grid_gdbt_model.fit(train_x, train_y)
    gdbt_model = grid_gdbt.best_estimator_
    return gdbt_model


import lightgbm
def lgb_best(train_x,train_y):
    grid_lgb_model = GridSearchCV(
        lightgbm.LGBMClassifier(boosting_type='gbdt', objective='binary', metric='auc', is_unbalance='True',
                                lambda_l1=0.1, verbose=1),
        param_grid={
            'max_depth': range(3, 8, 2), 'num_leaves': range(50, 170, 30)
        }, n_jobs=4, scoring='roc_auc', cv=5)

    grid_lgb = grid_lgb_model.fit(train_x, train_y)
    lgb_model = grid_lgb.best_estimator_
    return lgb_model

