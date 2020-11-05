from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
iris = load_iris()
print(iris.data.shape)#查看数据
train_x,train_y=iris.data, iris.target
feature_names=iris.feature_names
import seaborn as sns
import pandas as pd


train_y=pd.Series(train_y)
train_x=pd.DataFrame(train_x)
train_x.columns=feature_names
B, M,C= train_y.value_counts()

y=train_y
x=train_x



#画出特征与label的关系，注意x为DataFrame，y为series
def violinplot_feature(x,y):
    data = x
    data['label']=y
    data_n_2 = (data - data.mean()) / (data.std())              # standardization
    data = pd.melt(data,
                        id_vars='label',
                        var_name="features",
                        value_name='value')
    plt.figure(figsize=(10,10))
    sns.violinplot(x="features", y="value",hue='label', data=data, inner="quart")
    plt.xticks(rotation=90)
    plt.show()
    return


#画出特征与label的关系，注意x为DataFrame，y为series，更简单好看
def swarmplot_feature(x,y):
    sns.set(style="whitegrid", palette="muted")
    data = x
    data = (data - data.mean()) / (data.std())
    data['label'] = y  # standardization
    data = pd.melt(data,
                   id_vars='label',
                   var_name="features",
                   value_name='value')
    plt.figure(figsize=(10,10))
    sns.swarmplot(x="features", y="value", hue="label", data=data)
    plt.xticks(rotation=90)
    plt.show()

#画出特征图，便于观察离群点，
def outliers_boxplot(x,y):
    data = x
    data = (data - data.mean()) / (data.std())
    data['label']=y        # standardization
    data = pd.melt(data,
                        id_vars='label',
                        var_name="features",
                        value_name='value')
    plt.figure(figsize=(10,10))
    sns.boxplot(x="features", y="value", hue="label", data=data)
    plt.xticks(rotation=90)
    plt.show()



#观察特征之间的相关度
def feature_corr():
    data = x
    data = (data - data.mean()) / (data.std())
    #data['label']=y        # standardization
    # correlation map
    f, ax = plt.subplots(figsize=(18, 18))
    sns.heatmap(x.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)
    plt.show()



def select_feature_RF_vis(x,y):
    data = x
    data = (data - data.mean()) / (data.std())
    from sklearn.feature_selection import RFECV
    from sklearn.ensemble import RandomForestClassifier

    # The "accuracy" scoring is proportional to the number of correct classifications
    clf_rf_4 = RandomForestClassifier()
    rfecv = RFECV(estimator=clf_rf_4, step=1, cv=5,scoring='accuracy')   #5-fold cross-validation
    rfecv = rfecv.fit(data, y)

    print('Optimal number of features :', rfecv.n_features_)
    print('Best features :', data.columns[rfecv.support_])

    # Plot number of features VS. cross-validation scores

    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score of number of selected features")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()


def PCA_vis(x,y):
    from sklearn.model_selection import train_test_split
    # split data train 70 % and test 30 %
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    #normalization
    x_train_N = (x_train-x_train.mean())/(x_train.max()-x_train.min())
    x_test_N = (x_test-x_test.mean())/(x_test.max()-x_test.min())

    from sklearn.decomposition import PCA
    pca = PCA()
    pca.fit(x_train_N)

    plt.figure(1, figsize=(14, 13))
    plt.clf()
    plt.axes([.2, .2, .7, .7])
    plt.plot(pca.explained_variance_ratio_, linewidth=2)
    plt.axis('tight')
    plt.xlabel('n_components')
    plt.ylabel('explained_variance_ratio_')
    plt.show()




