from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
import numpy as np
from sklearn.feature_selection import f_classif
import pandas as pd
import lightgbm as lgb
import re
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import datetime

def getdata():
    transaction_trn = pd.read_csv('..\\transaction_TRAIN_new.csv')
    transaction_test = pd.read_csv('..\\transaction_round1_new.csv')
    tag_trn = pd.read_csv('..\\tag_TRAIN_new.csv')

    # ===================================处理操作详情=====================================#

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

    df['hour'] = df['time'].apply(lambda x: datetime.datetime.strptime(x, '%H:%M:%S').hour)
    del df['time']

    df['time_hour']=df['day']*24+df['hour']


    df.info()
    print(df.isnull().any())
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None) #打印所有信息，不省略
    print(df.describe())

    df['bal_trans']=df['trans_amt']*10+df['bal']
    x=df['bal_trans']
    print()



    #--------------------------所以数值特征可视化

    label_feature_vis = ['channel', 'trans_amt', 'bal','day', 'trans_type2', 'market_type', 'Tag']

    df_vis=pd.DataFrame()
    for each in label_feature_vis:
        df_vis[each] = df[each]

    X_vis = df_vis[df_vis.Tag != -1].drop(['Tag'], axis=1)[:10000]
    y_vis = df_vis[df_vis.Tag != -1]['Tag'][:10000]
    return X_vis,y_vis

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

#-----------------------------两个特征可视化

'''
label_feature_vis = ['bal','trans_amt','Tag']

df_vis=pd.DataFrame()
for each in label_feature_vis:
    df_vis[each] = df[each]

X = df_vis[df_vis.Tag != -1].drop(['Tag'], axis=1)[:1000000]
y = df_vis[df_vis.Tag != -1]['Tag'][:1000000]

feature_names=df[df.Tag != -1].drop(['Tag', 'UID'], axis=1).columns

'''


import seaborn as sns
import matplotlib.pyplot as plt

def vis_two_feature1(x,y):
    sns.set(style="whitegrid", palette="muted")
    data=pd.DataFrame()
    data['UID'] = x
    #data = (data - data.mean()) / (data.std())
    data['label'] = y  # standardization
    #data = pd.melt(data,id_vars='label',var_name="features",value_name='value')
    plt.figure(figsize=(10, 10))
    sns.lmplot(x="UID", y="label",hue="label",  data=data,fit_reg=False)#,分图
    #sns.lmplot(x="bal", y="trans_amt", hue="label", data=data, fit_reg=False)  #不分图
    plt.xticks(rotation=90)
    plt.show()



x1=pd.read_csv('..\\tag_train_new.csv')
#train=train.sample(frac=1)
#x1 = pd.DataFrame()
#x1['UID']=train['UID']
#x1['Tag']=train['Tag']

#x1.sample(frac = 1)

a=x1.shape[0]
index=int((x1.shape[0])/3)
print(x1['Tag'][:index].value_counts())
print(x1['Tag'][index:index*2].value_counts())
print(x1['Tag'][index*2:index*3].value_counts())
#print(x1['Tag'][index*3:index*4].value_counts())
#print(x1['Tag'][index*4:index*5].value_counts())
#vis_two_feature(X,y)

y = pd.read_csv('..\\tag_TRAIN_new.csv')

test=pd.read_csv('test.csv')

def vis_two_feature():


    y1 = pd.read_csv('..\\tag_TRAIN_new.csv')
    #x = y.drop('Tag', axis=1)
    y = y1['Tag']

    #feature_names = df[df.Tag != -1].drop(['Tag', 'UID'], axis=1).columns



    sns.set(style="whitegrid", palette="muted")
    data = y1[:1000]
    #data = (data - data.mean()) / (data.std())
    #data['label'] = y  # standardization
    #data = pd.melt(data,id_vars='label',var_name="features",value_name='value')
    plt.figure(figsize=(10, 10))
    #sns.lmplot(x="device_cat", y="os", col="label", data=data,fit_reg=False)#,分图
    sns.swarmplot(x="UID", y="Tag", data=data)  #不分图
    plt.xticks(rotation=90)
    plt.show()

#vis_two_feature()



