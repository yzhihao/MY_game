import numpy as np
import pandas as pd
import datetime

op_train = pd.read_csv('..\operation_TRAIN_new.csv')
op_test = pd.read_csv('..\operation_round1_new.csv')
trans_train = pd.read_csv('..\\transaction_TRAIN_new.csv')
trans_test = pd.read_csv('..\\transaction_round1_new.csv')
y = pd.read_csv('..\\tag_TRAIN_new.csv')
sub = pd.read_csv('..\sub.csv')


#------------数据预处理---------------end#
def mis_impute(data):
    for i in data.columns:
        if data[i].dtype == "object":
            data[i] = data[i].fillna("other")
        elif (data[i].dtype == "int64" or data[i].dtype == "float64"):
            data[i] = data[i].fillna(data[i].mean())#还有众数.mode()；中位数.median()
        else:
            pass
    return data

def split_version(v, n):
    if pd.isna(v) or v=='other':
        return np.nan
    return int(v.split('.')[n-1])

geo_dict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
            '9': 9, 'b': 10, 'c':11, 'd':12, 'e':13, 'f':14, 'g':15, 'h':16,  'j':17,
            'k':18, 'm':19, 'n':20, 'p':21, 'q':22, 'r':23, 's':24, 't':25, 'u':26,
            'v':27, 'w':28, 'x':29, 'y':30, 'z':31,}
def split_geo(g, n):
    if pd.isna(g) or g=='other' :
        return np.nan
    return geo_dict[g[n-1]]

def add_feature_all(df):

    df['hour'] = df['time'].apply(lambda x: datetime.datetime.strptime(x, '%H:%M:%S').hour)
    del df['time']

    for i in range(1, 5):
        df['geo_' + str(i)] = df['geo_code'].apply(lambda g: split_geo(g, i))
    del df['geo_code']

    df['time_hour'] = df['day'] * 24 + df['hour']
    return df


def add_feature_op(df):
    df['version_1'] = df['version'].apply(lambda v: split_version(v, 1))
    df['version_2'] = df['version'].apply(lambda v: split_version(v, 2))
    df['version_3'] = df['version'].apply(lambda v: split_version(v, 3))
    del df['version']

#    df['device_os_cat'] = df['os'] * 10 + df['device_cat']
    df.drop(['wifi'], axis=1, inplace=True)

    return df

def add_feature_trans(df):
    device_num = df.shape[0]

    df['make_is_null'] = np.zeros((device_num,))
    df.loc[df['market_type'].isnull() == True, 'make_is_null'] = 1
    x = df['make_is_null']

    df.drop(['market_code', 'code1', 'code2'], axis=1, inplace=True)
    df['bal_trans'] = df['trans_amt'] + df['bal']
    return df


'''
op_train=add_feature_all(op_train)
op_test=add_feature_all(op_test)
trans_train=add_feature_all(trans_train)
trans_test=add_feature_all(trans_test)

trans_train=add_feature_trans(trans_train)
trans_test=add_feature_trans(trans_test)

op_train=add_feature_op(op_train)
op_test=add_feature_op(op_test)
'''

#------------处理空缺值--------------
op_train=mis_impute(op_train)
op_test=mis_impute(op_test)
trans_train=mis_impute(trans_train)
trans_test=mis_impute(trans_test)


#----------转化率特征&时间转化率
op_train = pd.merge(op_train, y, how='left', on='UID')
op_train1=op_train[op_train['Tag']==1]

trans_train = pd.merge(trans_train, y, how='left', on='UID')
trans_train1=trans_train[trans_train['Tag']==1]

'''
#----------时间转化率特征
trans_train=trans_train.reset_index()
trans_test=trans_test.reset_index()

op_train_copy=op_train.copy()
op_test_copy=op_test.copy()
trans_train_copy=trans_train.copy()
trans_test_copy=trans_test.copy()

for feature in trans_train.columns[2:]:
    if trans_train_copy[feature].dtype == 'object' and feature != 'Tag':
    #if  feature in ['acc_id1','acc_id2','acc_id3','device_code1','device_code2','device_code3']:#op_train[feature].dtype == 'object' and
        trans_train = trans_train.merge(trans_train_copy.groupby([feature])['time_hour'].rank().reset_index(), on='index', how='left',suffixes=[feature+'_x',feature+'_y'])

    #if  feature in ['acc_id1','acc_id2','acc_id3','device_code1','device_code2','device_code3']:#op_train[feature].dtype == 'object' and
        trans_test = trans_test.merge(trans_test_copy.groupby([feature])['time_hour'].rank().reset_index(), on='index', how='left',suffixes=[feature+'_x',feature+'_y'])

trans_train=trans_train.drop('index',axis=1).fillna(0)
trans_test=trans_test.drop('index',axis=1).fillna(0)


op_test=op_test.reset_index()
op_train=op_train.reset_index()
for feature in op_train.columns[2:]:
    if op_train_copy[feature].dtype == 'object' and feature != 'Tag':
    #if  feature in ['device_code1','device_code2','device_code3']:#op_train[feature].dtype == 'object' and
        op_train = op_train.merge(op_train_copy.groupby([feature])['time_hour'].rank().reset_index(), on='index', how='left',suffixes=[feature+'_x',feature+'_y'])

    #if  feature in ['device_code1','device_code2','device_code3']:#op_train[feature].dtype == 'object' and
        op_test = op_test.merge(op_test_copy.groupby([feature])['time_hour'].rank().reset_index(), on='index', how='left',suffixes=[feature+'_x',feature+'_y'])


op_train=op_train.drop('index',axis=1).fillna(0)
op_test=op_test.drop('index',axis=1).fillna(0)

'''

#----------原始转化率特征
for feature in op_train.columns[2:]:
    if op_train[feature].dtype == 'object' and feature!='Tag':
        x=op_train1[feature].value_counts()/op_train1.shape[0]
        if feature + str('trans') in ['modetrans', #'versiontrans', 'device1trans',
                                      'device_code3trans', 'mac1trans',
                                      'ip2trans', 'ip2_subtrans']:

            num = op_train.shape[0]
            op_train[feature + str('trans')] = np.zeros((num,))
            op_train.loc[x[op_train[feature]].reset_index(drop=True).values > 0.2, feature + str('trans')] = 1.0
            z = x[op_train[feature]].reset_index(drop=True)
            num = op_test.shape[0]
            op_test[feature + str('trans')] = np.zeros((num,))
            op_test.loc[x[op_test[feature]].reset_index(drop=True).values > 0.2, feature + str('trans')] = 1.0
        else:
            num = op_train.shape[0]
            op_train[feature + str('trans')] = np.zeros((num,))
            op_train.loc[x[op_train[feature]].reset_index(drop=True).values >0.3,feature + str('trans')] = 1.0

            num = op_test.shape[0]
            op_test[feature + str('trans')] = np.zeros((num,))
            op_test.loc[x[op_test[feature]].reset_index(drop=True).values > 0.03, feature + str('trans')] = 1.0

for feature in trans_train.columns[2:]:
    if trans_train[feature].dtype == 'object' and feature != 'Tag':
        x = trans_train1[feature].value_counts() / trans_train1.shape[0]
        if feature + str('trans') in ['amt_src1trans', 'code1trans', 'code2trans', 'trans_type1trans',
                                      'device_code3trans', 'device1trans',
                                      'amt_src2trans', 'acc_id2trans', 'acc_id3trans', 'market_codetrans']:

            num = trans_train.shape[0]
            trans_train[feature + str('trans')] = np.zeros((num,))
            z=x[trans_train[feature]].reset_index(drop=True).values
            trans_train.loc[x[trans_train[feature]].reset_index(drop=True).values > 0.2, feature + str('trans')] = 1.0

            num = trans_test.shape[0]
            trans_test[feature + str('trans')] = np.zeros((num,))
            trans_test.loc[x[trans_test[feature]].reset_index(drop=True).values > 0.2, feature + str('trans')] = 1.0
        else:
            num = trans_train.shape[0]
            trans_train[feature + str('trans')] = np.zeros((num,))
            trans_train.loc[x[trans_train[feature]].reset_index(drop=True).values > 0.03, feature + str('trans')] = 1.0

            num = trans_test.shape[0]
            trans_test[feature + str('trans')] = np.zeros((num,))
            trans_test.loc[x[trans_test[feature]].reset_index(drop=True).values > 0.03, feature + str('trans')] = 1.0

op_train=op_train.drop('Tag',axis = 1).fillna(0)
op_test=op_test.fillna(0)
trans_train=trans_train.drop('Tag',axis = 1).fillna(0)
trans_test=trans_test.fillna(0)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None) #打印所有信息，不省略
print(pd.DataFrame(op_train).describe()) #查看数值数据情况
print(pd.DataFrame(trans_train).describe()) #查看数值数据情况


print(1)

#-----------------基础统计特征+排序特征，加0/1是否空值特征
def get_feature(op, trans, label):
    label['trans_op']= op.groupby(['UID'])['UID'].count() / trans.groupby(['UID'])['UID'].count()

    for feature in op.columns[2:]:
        label = label.merge(op.groupby(['UID'])[feature].count().reset_index(), on='UID', how='left')
        label = label.merge(op.groupby(['UID'])[feature].nunique().reset_index(), on='UID', how='left')
        '''
        if op[feature].dtype == 'object':
            label = label.merge(op.groupby(['UID'])[feature].count().reset_index(),on='UID',how='left')
            label =label.merge(op.groupby(['UID'])[feature].nunique().reset_index(),on='UID',how='left',suffixes=['_opx','_opy'])
            
            #---------交叉----差值特征
            label[feature + str('cha_cn')] = label[feature + str('_opx')]-label[feature + str('_opy')]
            label[feature + str('rate_cn')] = label[feature + str('_opx')]/label[feature + str('_opy')]

            label[feature + str('_oprxifn')]=label[feature + str('_opx')].isnull().astype('int')
            label[feature + str('_oprx')] = label[feature + str('_opx')].rank()
            label[feature + str('_opry')] = label[feature + str('_opy')].rank()
            
        else:
            label = label.merge(op.groupby(['UID'])[feature].count().reset_index(), on='UID', how='left')
            label = label.merge(op.groupby(['UID'])[feature].nunique().reset_index(), on='UID', how='left',suffixes=['_opx','_opy'])
            label = label.merge(op.groupby(['UID'])[feature].max().reset_index(), on='UID', how='left')
            label = label.merge(op.groupby(['UID'])[feature].min().reset_index(), on='UID', how='left',suffixes=['_opx1','_opy1'])
            label = label.merge(op.groupby(['UID'])[feature].sum().reset_index(), on='UID', how='left')
            label = label.merge(op.groupby(['UID'])[feature].mean().reset_index(), on='UID', how='left',suffixes=['_opx2','_opy2'])
            label = label.merge(op.groupby(['UID'])[feature].std().reset_index(), on='UID', how='left')
            label = label.merge(op.groupby(['UID'])[feature].median().reset_index(), on='UID', how='left',suffixes=['_opx3','_opy3'])
             ---------交叉----差值特征
            label[feature + str('cha_cn')] = label[feature + str('_opx')] - label[feature + str('_opy')]
            label[feature + str('rate_cn')] = label[feature + str('_opx')] / label[feature + str('_opy')]
            label[feature + str('rate_mm')] = label[feature + str('_opx1')] -label[feature + str('_opy1')]



            label[feature + str('_oprxifn')] = label[feature + str('_opx')].isnull().astype('int')
            label[feature + str('_oprx')] = label[feature + str('_opx')].rank()
            label[feature + str('_opry')] = label[feature + str('_opy')].rank()
            label[feature + str('_oprx1')] = label[feature + str('_opx1')].rank()
            label[feature + str('_opry1')] = label[feature + str('_opy1')].rank()
            label[feature + str('_oprx2')] = label[feature + str('_opx2')].rank()
            label[feature + str('_opry2')] = label[feature + str('_opy2')].rank()
            label[feature + str('_oprx3')] = label[feature + str('_opx3')].rank()
            label[feature + str('_opry3')] = label[feature + str('_opy3')].rank()
            '''

    for feature in trans.columns[2:]:
        if trans[feature].dtype == 'object':
            label =label.merge(trans.groupby(['UID'])[feature].count().reset_index(),on='UID',how='left')
            label =label.merge(trans.groupby(['UID'])[feature].nunique().reset_index(),on='UID',how='left')
        else:
            label =label.merge(trans.groupby(['UID'])[feature].count().reset_index(),on='UID',how='left')
            label =label.merge(trans.groupby(['UID'])[feature].nunique().reset_index(),on='UID',how='left')
            label =label.merge(trans.groupby(['UID'])[feature].max().reset_index(),on='UID',how='left')
            label =label.merge(trans.groupby(['UID'])[feature].min().reset_index(),on='UID',how='left')
            label =label.merge(trans.groupby(['UID'])[feature].sum().reset_index(),on='UID',how='left')
            label =label.merge(trans.groupby(['UID'])[feature].mean().reset_index(),on='UID',how='left')
            label =label.merge(trans.groupby(['UID'])[feature].std().reset_index(),on='UID',how='left')
    '''
    for feature in trans.columns[2:]:
        if trans[feature].dtype == 'object':
            label =label.merge(trans.groupby(['UID'])[feature].count().reset_index(),on='UID',how='left')
            label =label.merge(trans.groupby(['UID'])[feature].nunique().reset_index(),on='UID',how='left',suffixes=['_x','_y'])
            
            # ---------交叉----差值特征
            label[feature + str('cha_cn')] = label[feature + str('_x')] - label[feature + str('_y')]
            label[feature + str('rate_cn')] = label[feature + str('_x')] / label[feature + str('_y')]

            label[feature + str('_rxifn')] = label[feature + str('_x')].isnull().astype('int')
            label[feature + str('_rx')] = label[feature + str('_x')].rank()
            label[feature + str('_ry')] = label[feature + str('_y')].rank()
            
        else:
            label =label.merge(trans.groupby(['UID'])[feature].count().reset_index(),on='UID',how='left')
            label =label.merge(trans.groupby(['UID'])[feature].nunique().reset_index(),on='UID',how='left',suffixes=['_x','_y'])
            label =label.merge(trans.groupby(['UID'])[feature].max().reset_index(),on='UID',how='left')
            label =label.merge(trans.groupby(['UID'])[feature].min().reset_index(),on='UID',how='left',suffixes=['_x1','_y1'])
            label =label.merge(trans.groupby(['UID'])[feature].sum().reset_index(),on='UID',how='left')
            label =label.merge(trans.groupby(['UID'])[feature].mean().reset_index(),on='UID',how='left',suffixes=['_x2','_y2'])
            label =label.merge(trans.groupby(['UID'])[feature].std().reset_index(),on='UID',how='left')
            label = label.merge(trans.groupby(['UID'])[feature].median().reset_index(), on='UID', how='left',suffixes=['_x3','_y3'])
            
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
    return label


train = get_feature(op_train, trans_train, y).fillna(-1)
test = get_feature(op_test, trans_test, sub).fillna(-1)


train.to_csv('train.csv',index=False)
test.to_csv('test.csv',index=False)