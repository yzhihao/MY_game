
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



#------------------神经网络特征+排序特征，加0/1是否空值特征
def get_feature_nn(nn_feature_op,trans_nn_feature,stacking_test_feature_op,stacking_test_feature, label):
    '''
    for feature in nn_feature_op.columns[1:]:
        label = label.merge(nn_feature_op.groupby(['UID'])[feature].count().reset_index(), on='UID', how='left')
        label = label.merge(nn_feature_op.groupby(['UID'])[feature].nunique().reset_index(), on='UID', how='left',suffixes=['_x', '_y'])

        label = label.merge(nn_feature_op.groupby(['UID'])[feature].max().reset_index(), on='UID', how='left')
        label = label.merge(nn_feature_op.groupby(['UID'])[feature].min().reset_index(), on='UID', how='left',suffixes=['_x1', '_y1'])
        label = label.merge(nn_feature_op.groupby(['UID'])[feature].sum().reset_index(), on='UID', how='left')
        label = label.merge(nn_feature_op.groupby(['UID'])[feature].mean().reset_index(), on='UID', how='left',suffixes=['_x2', '_y2'])
        label = label.merge(nn_feature_op.groupby(['UID'])[feature].std().reset_index(), on='UID', how='left')
        label = label.merge(nn_feature_op.groupby(['UID'])[feature].median().reset_index(), on='UID', how='left',suffixes=['_x3', '_y3'])

        # ---------交叉----差值特征
        label[feature + str('cha_cn')] = label[feature + str('_x')] - label[feature + str('_y')]
        label[feature + str('rate_cn')] = label[feature + str('_x')] / label[feature + str('_y')]
        label[feature + str('rate_mm')] = label[feature + str('_x1')] - label[feature + str('_y1')]

        label[feature + str('_rxifn')] = label[feature + str('_x')].isnull().astype('int')
        label[feature + str('_rx')] = label[feature + str('_x')].rank()
        label[feature + str('_ry')] = label[feature + str('_y')].rank()
        label[feature + str('_rx1')] = label[feature + str('_x1')].rank()
        label[feature + str('_ry1')] = label[feature + str('_y1')].rank()
        label[feature + str('_rx2')] = label[feature + str('_x2')].rank()
        label[feature + str('_ry2')] = label[feature + str('_y2')].rank()
        label[feature + str('_rx3')] = label[feature + str('_x3')].rank()
        label[feature + str('_ry3')] = label[feature + str('_y3')].rank()

    for feature in trans_nn_feature.columns[1:]:
        label = label.merge(trans_nn_feature.groupby(['UID'])[feature].count().reset_index(), on='UID', how='left')
        label = label.merge(trans_nn_feature.groupby(['UID'])[feature].nunique().reset_index(), on='UID', how='left',suffixes=['_x', '_y'])
        label = label.merge(trans_nn_feature.groupby(['UID'])[feature].max().reset_index(), on='UID', how='left')
        label = label.merge(trans_nn_feature.groupby(['UID'])[feature].min().reset_index(), on='UID', how='left',suffixes=['_x1', '_y1'])
        label = label.merge(trans_nn_feature.groupby(['UID'])[feature].sum().reset_index(), on='UID', how='left')
        label = label.merge(trans_nn_feature.groupby(['UID'])[feature].mean().reset_index(), on='UID', how='left',suffixes=['_x2', '_y2'])
        label = label.merge(trans_nn_feature.groupby(['UID'])[feature].std().reset_index(), on='UID', how='left')
        label = label.merge(trans_nn_feature.groupby(['UID'])[feature].median().reset_index(), on='UID', how='left',suffixes=['_x3', '_y3'])

        # ---------交叉----差值特征
        label[feature + str('cha_cn')] = label[feature + str('_x')] - label[feature + str('_y')]
        label[feature + str('rate_cn')] = label[feature + str('_x')] / label[feature + str('_y')]
        label[feature + str('rate_mm')] = label[feature + str('_x1')] - label[feature + str('_y1')]

        label[feature + str('_rxifn')] = label[feature + str('_x')].isnull().astype('int')
        label[feature + str('_rx')] = label[feature + str('_x')].rank()
        label[feature + str('_ry')] = label[feature + str('_y')].rank()
        label[feature + str('_rx1')] = label[feature + str('_x1')].rank()
        label[feature + str('_ry1')] = label[feature + str('_y1')].rank()
        label[feature + str('_rx2')] = label[feature + str('_x2')].rank()
        label[feature + str('_ry2')] = label[feature + str('_y2')].rank()
        label[feature + str('_rx3')] = label[feature + str('_x3')].rank()
        label[feature + str('_ry3')] = label[feature + str('_y3')].rank()
        '''

    for feature in stacking_test_feature.columns:
        if feature!= 'UID':
            label = label.merge(stacking_test_feature.groupby(['UID'])[feature].count().reset_index(), on='UID', how='left')
            label = label.merge(stacking_test_feature.groupby(['UID'])[feature].nunique().reset_index(), on='UID', how='left',suffixes=['_x', '_y'])
            label = label.merge(stacking_test_feature.groupby(['UID'])[feature].max().reset_index(), on='UID', how='left')
            label = label.merge(stacking_test_feature.groupby(['UID'])[feature].min().reset_index(), on='UID', how='left',suffixes=['_x1', '_y1'])
            label = label.merge(stacking_test_feature.groupby(['UID'])[feature].sum().reset_index(), on='UID', how='left')
            label = label.merge(stacking_test_feature.groupby(['UID'])[feature].mean().reset_index(), on='UID', how='left',suffixes=['_x2', '_y2'])
            label = label.merge(stacking_test_feature.groupby(['UID'])[feature].std().reset_index(), on='UID', how='left')
            label = label.merge(stacking_test_feature.groupby(['UID'])[feature].median().reset_index(), on='UID', how='left',suffixes=['_x3', '_y3'])
            '''
            # ---------交叉----差值特征
            label[feature + str('cha_cn')] = label[feature + str('_x')] - label[feature + str('_y')]
            label[feature + str('rate_cn')] = label[feature + str('_x')] / label[feature + str('_y')]
            label[feature + str('rate_mm')] = label[feature + str('_x1')] - label[feature + str('_y1')]

            label[feature + str('_rxifn')] = label[feature + str('_x')].isnull().astype('int')
            label[feature + str('_rx')] = label[feature + str('_x')].rank()
            label[feature + str('_ry')] = label[feature + str('_y')].rank()
            label[feature + str('_rx1')] = label[feature + str('_x1')].rank()
            label[feature + str('_ry1')] = label[feature + str('_y1')].rank()
            label[feature + str('_rx2')] = label[feature + str('_x2')].rank()
            label[feature + str('_ry2')] = label[feature + str('_y2')].rank()
            label[feature + str('_rx3')] = label[feature + str('_x3')].rank()
            label[feature + str('_ry3')] = label[feature + str('_y3')].rank()
            '''
    for feature in stacking_test_feature_op.columns:
        if feature!= 'UID':
            label = label.merge(stacking_test_feature_op.groupby(['UID'])[feature].count().reset_index(), on='UID', how='left')
            label = label.merge(stacking_test_feature_op.groupby(['UID'])[feature].nunique().reset_index(), on='UID', how='left',suffixes=['_x', '_y'])
            label = label.merge(stacking_test_feature_op.groupby(['UID'])[feature].max().reset_index(), on='UID', how='left')
            label = label.merge(stacking_test_feature_op.groupby(['UID'])[feature].min().reset_index(), on='UID', how='left',suffixes=['_x1', '_y1'])
            label = label.merge(stacking_test_feature_op.groupby(['UID'])[feature].sum().reset_index(), on='UID', how='left')
            label = label.merge(stacking_test_feature_op.groupby(['UID'])[feature].mean().reset_index(), on='UID', how='left',suffixes=['_x2', '_y2'])
            label = label.merge(stacking_test_feature_op.groupby(['UID'])[feature].std().reset_index(), on='UID', how='left')
            label = label.merge(stacking_test_feature_op.groupby(['UID'])[feature].median().reset_index(), on='UID', how='left',suffixes=['_x3', '_y3'])

    return label

nn_train_feature_op = (pd.read_csv('nn_faeture_op.csv')).drop(['cnn_feature_pre','nn_feature_pre'],axis = 1).fillna(0)
test_feature_op = (pd.read_csv('nn_faeture_test_op.csv')).drop(['cnn_feature_pre','nn_feature_pre'],axis = 1).fillna(0)
nn_faeture = (pd.read_csv('nn_faeture.csv')).drop(['cnn_feature_pre','nn_feature_pre'],axis = 1).fillna(0)
nn_faeture_test = (pd.read_csv('nn_faeture_test.csv')).drop(['cnn_feature_pre','nn_feature_pre'],axis = 1).fillna(0)


#stacking_train_feature_trans = pd.read_csv('stacking_fea_trans.csv')
#stacking_test_feature_trans  = pd.read_csv('stacking_fea_trans_test.csv')
#stacking_train_feature_op = pd.read_csv('stacking_fea_op.csv')
#stacking_test_feature_op  = pd.read_csv('stacking_fea_op_test.csv')


train = get_feature_nn(nn_train_feature_op, nn_faeture,stacking_train_feature_op,stacking_train_feature_trans, train).fillna(-1)
test = get_feature_nn(test_feature_op, nn_faeture_test,stacking_test_feature_op, stacking_test_feature_trans,test).fillna(-1)