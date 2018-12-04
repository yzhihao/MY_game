
import pandas as pd
import  numpy as np
#----------stacking  阈值特征
stacking_train_feature_trans_pre = pd.read_csv('stacking_fea_trans_pre.csv')
stacking_test_feature_trans_pre  = pd.read_csv('stacking_fea_trans_test_pre.csv')
stacking_train_feature_op_pre = pd.read_csv('stacking_fea_op_pre.csv')
stacking_test_feature_op_pre = pd.read_csv('stacking_fea_op_test_pre.csv')
stacking_train_feature_trans = pd.DataFrame()
stacking_test_feature_trans  =  pd.DataFrame()
stacking_train_feature_op = pd.DataFrame()
stacking_test_feature_op =  pd.DataFrame()

for feature in stacking_train_feature_op_pre.columns:
    if feature!='UID'and feature!='2' and feature!=2 :#
            num = stacking_train_feature_op_pre.shape[0]
            stacking_train_feature_op[str('stackingop') + feature] = stacking_train_feature_op_pre[feature]

            #stacking_train_feature_op[str('stackingop')+feature ] = np.zeros((num,))
            #stacking_train_feature_op.loc[stacking_train_feature_op_pre[feature].reset_index(drop=True).values >=0.991, str('stackingop')+feature] = 1.0

            stacking_test_feature_op[str('stackingop') + feature] = stacking_test_feature_op_pre[feature]

            num = stacking_test_feature_op_pre.shape[0]
            #stacking_test_feature_op[str('stackingop') + feature] = np.zeros((num,))
            #stacking_test_feature_op.loc[stacking_test_feature_op_pre[feature].reset_index(drop=True).values >=0.991, str('stackingop') + feature] = 1.0

for feature in stacking_train_feature_trans_pre.columns:
    if feature!='UID'and feature!='2' and feature!=2:
            num = stacking_train_feature_trans_pre.shape[0]
            stacking_train_feature_trans[str('stacking') + feature] = stacking_train_feature_trans_pre[feature]

            #stacking_train_feature_trans[str('stacking')+feature ] = np.zeros((num,))
            #stacking_train_feature_trans.loc[stacking_train_feature_trans_pre[feature].reset_index(drop=True).values >=0.991, str('stacking')+feature] = 1.0

            stacking_test_feature_trans[str('stacking') + feature] = stacking_test_feature_trans_pre[feature]
            num = stacking_test_feature_trans_pre.shape[0]
            #stacking_test_feature_trans[str('stacking') + feature] = np.zeros((num,))
            #stacking_test_feature_trans.loc[stacking_test_feature_trans_pre[feature].reset_index(drop=True).values >=0.991, str('stacking')+feature] = 1.0

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None) #打印所有信息，不省略
print('====================stacking_train_feature_op==============') #查看数值数据情况
print(pd.DataFrame(stacking_train_feature_op).describe()) #查看数值数据情况
print('====================stacking_train_feature_trans==============') #查看数值数据情况
print(pd.DataFrame(stacking_train_feature_trans).describe()) #查看数值数据情况
print('===================stacking_test_feature_op===============') #查看数值数据情况
print(pd.DataFrame(stacking_test_feature_op).describe()) #查看数值数据情况
print('===================stacking_test_feature_trans===============') #查看数值数据情况
print(pd.DataFrame(stacking_test_feature_trans).describe()) #查看数值数据情况