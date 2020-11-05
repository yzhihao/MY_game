from sklearn.datasets import load_iris
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as  np

iris = load_iris()
print(iris.data.shape)#查看数据
train_x,train_y=iris.data, iris.target
feature_names=iris.feature_names

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None) #打印所有信息，不省略
print(pd.DataFrame(train_x).describe()) #查看数值数据情况
print(pd.DataFrame(train_x).info()) #查看数据情况


#print(pd.DataFrame(train_x).describe()) #查看数值数据情况
#print(pd.DataFrame(train_x).info()) #查看数据情况
#xgboost
#=================================处理缺失值============================================#
#df.isnull().any() 用来判断某列是否有缺失值
#df.isnull().all() 用来判断某列是否全部为空值

#--------------平均值，众数，中位数填充---------
def mis_impute(data):
    for i in data.columns:
        if data[i].dtype == "object":
            data[i] = data[i].fillna("other")
        elif (data[i].dtype == "int64" or data[i].dtype == "float64"):
            data[i] = data[i].fillna(data[i].mean())#还有众数.mode()；中位数.median()
        else:
            pass
    return data


'''
# 用前一个数据代替NaN：method='pad'
data_train.fillna(method='pad')
# 与pad相反，bfill表示用后一个数据代替NaN
data_train.fillna(method='bfill') 
'''

#data = data.dropna(axis=0) 删除存在nan的行数据

#--------------用随机森林预测缺失值---------
def set_missing_ages(df):

    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    # y即目标年龄
    y = known_age[:, 0]

    # X即特征属性值
    X = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1:])
#     print predictedAges
    # 用得到的预测结果填补原缺失数据
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges

    return df, rfr


# ------------------存在大量数据缺失，可以改变特征格式,（可能需要把变量映射到高维空间）
# 比如性别，有男、女、缺失三种情况，则映射成3个变量：是否男、是否女、是否缺失
# 定义工资改变特征缺失值处理函数，将有变化设为Yes，缺失设为No，该数值为类别
def set_salary_change(df):
    df.loc[(df.salary_change.notnull()), 'salary_change']="YES"
    df.loc[(df.salary_change.notnull()), 'salary_change']="NO"
    return df

#df = mis_impute(df)


#=================================异常值检测============================================#
# 对于每一个特征，找到值异常高或者是异常低的数据点
def IQR_check(log_data):
    for feature in feature_names:

        # TODO：计算给定特征的Q1（数据的25th分位点）
        Q1 = np.percentile(log_data[feature], 25)

        # TODO：计算给定特征的Q3（数据的75th分位点）
        Q3 = np.percentile(log_data[feature], 75)

        # TODO：使用四分位范围计算异常阶（k=1.5倍的四分位距）当k=3时是极度异常。
        step = 1.5 *(Q3 - Q1 )  # 中度异常
        min_Q = Q1 - step #下界
        max_Q = Q3 + step #上界
        print(Q1, Q3, step, min_Q, max_Q)


#-------------------统计学检测离群值
def statistics_check(total_data):
    # 标准差上下三倍绝对中位差之间属于正常点
    import numpy as np
    median = np.median(total_data)
    b = 1.4826  # 这个值应该是看需求加的，有点类似加大波动范围之类的
    mad = b * np.median(np.abs(total_data - median))
    lower_limit = median - (3 * mad)
    upper_limit = median + (3 * mad)
    # 平均值上下三倍标准差之间属于正常点
    std = np.std(total_data)
    mean = np.mean(total_data)
    b = 3
    lower_limit = mean - b * std
    upper_limit = mean + b * std