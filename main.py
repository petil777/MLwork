#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
from xgboost import plot_importance
import putil
from scipy.stats import skew


df = pd.read_csv('./data/train.csv')
#%%
# 1. Data Null Handling and Drop Column or data Row
# (Having a lot of null values column or not versatile value and not important columns)
# df.info()

# Seperate categorical and numerical data
df_num = df.select_dtypes(exclude=['object'])
df_obj = df.select_dtypes(include=['object'])

isnull_series = df.isnull().sum()
isnull_series = isnull_series[isnull_series>0].sort_values(ascending=False)

# Remove columns having a lot of nulls
df.drop(columns=isnull_series[:5].index.values, inplace=True)
df.drop(columns=['Id'], inplace=True)

# Remove row data
# df = df[df[col].notnull()] # For Data Drop
'''
isnull_series
PoolQC          1453
MiscFeature     1406
Alley           1369
Fence           1179
FireplaceQu      690=====
LotFrontage      259
'''

# Watch distribution of each columns for numerical data and object data
# fig, ax = plt.subplots(1, 1)  # To reproduce every image, make this with every for loop
# sns.distplot(df[col], ax=ax)  # Data Distribution histogram
# df[col].value_counts()        # distinct values check of columns 
# sns.regplot(x='feature_name', y='target_feature_name, data = df, ax = ax) # 1:1 pairplot with reg linear
# sns.pairplot(data=df, vars=['feature1', 'feature2'], ax = ax)             # pair scatter plot
# sns.heatmap(df.corr())        # Pearson correlation
#%%
# 2. Filling, Encoding, Scaling, Outlier (Come back after test some model)
# Filling
# df.describe()
# numerical aggregation is only applied to numerical columns
df.fillna(df.mean(), inplace=True)
#df[col].fillna(df[col].value_counts().index[0], inplace=True) # For most_frequent

# Outlier Detection (before scaling. watch among important features....by ridge, lasso, linearReg, etc.)
# sns.regplot(x='GrLivArea', y='SalePrice', data = df)
cond1 = df['GrLivArea'] > 4000
cond2 = df['SalePrice'] < 500000
outlier_index = df[cond1 & cond2].index
df.drop(outlier_index, axis=0, inplace=True)

# Scaling (numerical)
# For skewed data, log scaling, For large scale data and wide range data, robustScaler
feature_index = df.dtypes[df.dtypes!='object'].index
skew_features = df[feature_index].apply(lambda x : skew(x))
skew_features_top = skew_features[skew_features > 1].sort_values(ascending=False)
df[skew_features_top.index] = np.log1p(df[skew_features_top.index])

# Encoding (string object)
# Onehot, OrderValueEncode(0,1,2,3), Merge Columns, Re-Categorize(including data value compression)
# with pd.get_dummies, change all str data to onehot(not touch numerical)
df = pd.get_dummies(df)
#%%
# 3. Train with checking learning_curve, validation curve, cross_val_score(r2, acc, rmse, etc...) by GridSearching...
x_train, x_test, y_train, y_test = train_test_split(df.drop(columns=['SalePrice'], axis=1), df['SalePrice'], train_size=0.7 )

#%%

lr_reg = LinearRegression()
lr_reg.fit(x_train, y_train)
ridge_reg = Ridge(alpha=12)
ridge_reg.fit(x_train, y_train)
lasso_reg = Lasso()
lasso_reg.fit(x_train, y_train)
models = [lr_reg, ridge_reg]
for model in models:
    print(putil.get_rmse(model, x_test, y_test))
    coef = pd.Series(model.coef_, index=x_train.columns)
    codef = coef.sort_values(ascending=False)
    coef_high = codef.head(20)
    coef_high[::-1].plot.bar

#%%
reg = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, tree_method='gpu_hist', colsample_bytree=0.5,subsample=0.8)


reg.fit(x_train, y_train)

#%%
print('train error : ', r2_score(y_train, reg.predict(x_train)))
print('validation error : ', r2_score(y_test, reg.predict(x_test)))
plot_importance(reg, max_num_features=20, height=0.4)
'''
dic = reg.get_booster().get_score()
dd = pd.DataFrame.from_dict(dic, orient='index')
dd[0].sort_values(ascending=False)
'''

