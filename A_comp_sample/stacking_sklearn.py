from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
import numpy as np
#对于分类问题可以使用 ClassifierMixin


class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    # 我们将原来的模型clone出来，并且进行实现fit功能
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        #对于每个模型，使用交叉验证的方法来训练初级学习器，并且得到次级训练集
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                #instance=model
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # 使用次级训练集来训练次级学习器
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    #在上面的fit方法当中，我们已经将我们训练出来的初级学习器和次级学习器保存下来了
    #predict的时候只需要用这些学习器构造我们的次级预测数据集并且进行预测就可以了
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)


if __name__ == '__main__':
    from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                                  GradientBoostingClassifier, ExtraTreesClassifier)
    from sklearn.svm import SVC
    rf_model = RandomForestClassifier()
    adb_model = AdaBoostClassifier()
    gdbc_model = GradientBoostingClassifier()
    et_model = ExtraTreesClassifier()
    svc_model = SVC()
    base_models=[rf_model, adb_model, gdbc_model, et_model, svc_model]
    from sklearn.tree import DecisionTreeClassifier
    meta_model = DecisionTreeClassifier()
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    iris = load_iris()
    train_x, test_x, train_y, test_y = train_test_split(iris.data, iris.target, test_size=0.2)
    StackingModel=StackingAveragedModels(base_models,meta_model)
    StackingModel.fit(train_x,train_y)
    df_predict=StackingModel.predict(test_x)
    from sklearn.metrics import accuracy_score
    print("eval stage1 :%f" % accuracy_score(test_y, df_predict))