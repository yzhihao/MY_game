import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
np.random.seed(sum(map(ord, "categorical")))
titanic = sns.load_dataset("titanic")
tips = sns.load_dataset("tips")
iris = sns.load_dataset("iris")
# 查看特征
#sns.stripplot(x="day", y="total_bill", data=tips, jitter=True)
# 两个特征
sns.swarmplot(x="day", y="total_bill", hue="sex", data=tips)
plt.show()