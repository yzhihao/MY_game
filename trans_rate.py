import numpy as np
import pandas as pd
op_train = pd.read_csv('..\operation_TRAIN_new.csv')
op_test = pd.read_csv('..\operation_round1_new.csv')
trans_train = pd.read_csv('..\\transaction_TRAIN_new.csv')
trans_test = pd.read_csv('..\\transaction_round1_new.csv')
y = pd.read_csv('..\\tag_TRAIN_new.csv')
sub = pd.read_csv('..\sub.csv')



def mis_impute(data):
    for i in data.columns:
        if data[i].dtype == "object":
            data[i] = data[i].fillna("other")
        elif (data[i].dtype == "int64" or data[i].dtype == "float64"):
            data[i] = data[i].fillna(data[i].mean())#还有众数.mode()；中位数.median()
        else:
            pass
    return data

op_train=mis_impute(op_train)
op_test=mis_impute(op_test)
trans_train=mis_impute(trans_train)
trans_test=mis_impute(trans_test)

op_train = pd.merge(op_train, y, how='left', on='UID')
op_train1=op_train[op_train['Tag']==1]

trans_train = pd.merge(trans_train, y, how='left', on='UID')
trans_train1=trans_train[trans_train['Tag']==1]

op_train.info()
trans_train.info()

print(12)
# 统计转化率，某个IP在羊毛党总数的概率也就是某个IP是否代表羊毛党ip的概率
'''
for feature in op_train.columns[2:]:
    if op_train[feature].dtype == 'object' and feature!='Tag':
        x=op_train1[feature].value_counts()/op_train1.shape[0]
        op_train[feature + str('trans')] = x[op_train[feature]].reset_index(drop=True)
        op_test[feature + str('trans')]= x[op_test[feature]].reset_index(drop=True)

for feature in trans_train.columns[2:]:
    if trans_train[feature].dtype == 'object' and feature != 'Tag':
        x = trans_train1[feature].value_counts() / trans_train1.shape[0]
        trans_train[feature + str('trans')] = x[trans_train[feature]].reset_index(drop=True)
        trans_test[feature + str('trans')] = x[trans_test[feature]].reset_index(drop=True)
'''

# 统计转化率，某个IP在某个阈值下代表是羊毛党

for feature in op_train.columns[2:]:
    if op_train[feature].dtype == 'object' and feature!='Tag':
        x=op_train1[feature].value_counts()/op_train1.shape[0]
        num = op_train.shape[0]
        op_train[feature + str('trans')] = np.zeros((num,))
        op_train.loc[x[op_train[feature]].reset_index(drop=True).values >0.1,feature + str('trans')] = 1

        num = op_test.shape[0]
        op_test[feature + str('trans')] = np.zeros((num,))
        op_test.loc[x[op_test[feature]].reset_index(drop=True).values > 0.1, feature + str('trans')] = 1

for feature in trans_train.columns[2:]:
    if trans_train[feature].dtype == 'object' and feature != 'Tag':
        x = trans_train1[feature].value_counts() / trans_train1.shape[0]

        num = trans_train.shape[0]
        trans_train[feature + str('trans')] = np.zeros((num,))
        trans_train.loc[x[trans_train[feature]].reset_index(drop=True).values > 0.1, feature + str('trans')] = 1


        num = trans_test.shape[0]
        trans_test[feature + str('trans')] = np.zeros((num,))
        trans_test.loc[x[trans_test[feature]].reset_index(drop=True).values > 0.1, feature + str('trans')] = 1




op_train.info()
trans_train.info()
op_test.info()
trans_test.info()
op_train=op_train.drop('Tag',axis = 1).fillna(0)
op_test=op_test.drop('Tag',axis = 1).fillna(0)
trans_train=trans_train.drop('Tag',axis = 1).fillna(0)
trans_test=trans_test.drop('Tag',axis = 1).fillna(0)


print(12)
print()

#