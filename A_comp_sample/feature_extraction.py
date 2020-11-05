from sklearn.datasets import load_iris
import pandas as pd
import numpy as  np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
iris = load_iris()
print(iris.data.shape)#查看数据
train_x,train_y=iris.data, iris.target
feature_names=iris.feature_names
#print(pd.DataFrame(train_x).describe()) #查看数值数据情况
#print(pd.DataFrame(train_x).info()) #查看数据情况

#-------------对类别特征编号,并应用独热编码
def feature_factorize(label_feature,df):
    df = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6, 3, 2], "raw_grade": ['a', 'b', 'b', 'a', 'a', 'e', 'c', 'a']})
    label_feature=df.columns
    for each in label_feature:
        df[each] = pd.factorize(df[each])[0]
    enc =OneHotEncoder()
    enc.fit(df)
    print(enc.transform([[0,0]]).toarray())

#-----------------------------------特征离散化
def feature_discretize(label_feature,df):
    x=[1,1,5,5,5,5,8,8,10,10,10,10,14,14,14,14,15,15,15,15,15,15,18,18,18,18,18,18,18,18,18,20,2,20,20,20,20,20,20,21,21,21,25,25,25,25,25,28,28,30,30,30]
    x=pd.Series(x)
    s=pd.cut(x,bins=[0,10,20,30])
    d=pd.get_dummies(s)
    print()

#--------------------PCA降维----
def PCA_process():
    data = np.array([[ 1.  ,  1.  ],
           [ 0.9 ,  0.95],
           [ 1.01,  1.03],
           [ 2.  ,  2.  ],
           [ 2.03,  2.06],
           [ 1.98,  1.89],
           [ 3.  ,  3.  ],
           [ 3.03,  3.05],
           [ 2.89,  3.1 ],
           [ 4.  ,  4.  ],
           [ 4.06,  4.02],
           [ 3.97,  4.01]])
    from sklearn.decomposition import PCA
    pca=PCA(n_components=1) #n_components是指最终维数，若设置为mle的时候，表示自动又模型评估
    newData=pca.fit_transform(data)
    print(newData)

#==========================归一化===========================#
def z_score_standardizing():
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    X = np.array([[ 1., -1.,  2.],
                  [ 2.,  0.,  0.],
                  [ 0.,  1., -1.]])
    scaler = StandardScaler().fit(X)
    print(scaler)
    #StandardScaler(copy=True, with_mean=True, with_std=True)
    print(scaler.mean_)
    #[ 1.          0.          0.33333333]
    print (scaler.var_)
    #[0.66666667 0.66666667 1.55555556]
    print (scaler.transform(X).std(axis=0))
    #[1. 1. 1.]
    print (scaler.transform(X).mean(axis=0))
    #[0. 0. 0.]

def min_max_standardizing():
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    X_train = np.array([[ 1., -1.,  2.],
                        [ 2.,  0.,  0.],
                        [ 0.,  1., -1.]])
    min_max_scaler = MinMaxScaler()
    print (min_max_scaler)
    #MinMaxScaler(copy=True, feature_range=(0, 1))
    X_train_minmax = min_max_scaler.fit_transform(X_train)
    print(X_train_minmax)
    #[[ 0.5         0.          1.        ]
    # [ 1.          0.5         0.33333333]
    # [ 0.          1.          0.        ]]

    #将相同的缩放应用到测试集数据中
    X_test = np.array([-3., -1., 4.])
    X_test_minmax = min_max_scaler.transform(X_test)
    print (X_test_minmax)
    #[-1.5         0.          1.66666667]
    #缩放因子等属性
    print (min_max_scaler.scale_)
    #[ 0.5         0.5         0.33333333]
    print (min_max_scaler.min_)
    #[ 0.          0.5         0.33333333]

