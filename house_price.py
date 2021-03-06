#%%
# 0. Loading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
import xgboost as xgb
import lightgbm as lgbm
import putil
from scipy.stats import skew
import pickle

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

#[Watch distribution] of each columns for numerical data and object data
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

# Outlier Detection (Before scaling! watch among important features....by ridge, lasso, linearReg, etc.)
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
# 3. Model Define
# Train with checking learning_curve, validation curve
# If using cross_val_score(r2, acc, rmse, etc...) with GridSearchCv, don't have to use this func

# Model Define (If classifier, RandomForest == bagging+decisionClassifier, stackingClassifer, 
# May use pipeline
lr_reg = LinearRegression()
lr_params = {}
ridge_reg = Ridge()
ridge_params = [{'alpha' : [0.05, 0.1, 1, 5, 8, 10, 12, 15, 20]}]
lasso_reg = Lasso()
lasso_params = {'alpha' : [0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]}
models = [lr_reg, ridge_reg, lasso_reg]
params = [lr_params, ridge_params, lasso_params]
best_models = []
#%%
# 4. Make dataset for training and GridSearch for parameter tuning
# classifier : (confusion matrix, roc-auc)
# regressor : (adjusted r2, rmse, F statistic)
# Unsupervised : (elbow, silhouette, just see....)
x_train, x_test, y_train, y_test = train_test_split(df.drop(columns=['SalePrice'], axis=1), df['SalePrice'], train_size=0.7 )#radom_state = 

for model, param in zip(models, params):
    # inner fold(train + validation fold). nested cross val. useful for large data
    # In the end, train all folds 
    gs = GridSearchCV(model, param_grid=param, scoring='neg_mean_squared_error', cv=4)
    gs.fit(x_train, y_train)
    print('model : {}, best_params : {}'.format(model.__class__.__name__, gs.best_params_))
    # outer fold(inner + test fold) -> KFold. If classifier, stratifiedKFold automoatically
    # If use KFold directly, we can drop or select that fold. 
    print('새로운 데이터에 대한 정확도 : ', r2_score(y_test, gs.best_estimator_.predict(x_test)))
    best_models.append(gs.best_estimator_)


# Tuning for one model with nested cross validation (useful for big data)
# scores = []
# best_models = []
# cv = KFold(n_split=5) # Outer Fold (inner Fold + test set). Same with cross_val_score
# for tidx, vidx in cv.split(df):
#     x_train = df.drop(columns=['SalePrice'], axis=1).iloc[tidx]
#     y_train = df['SalePrice'].iloc[tidx]
#     # Inner Fold (train set + validation set. At Final step, this train all inner folds)
#     gs = GridSearchCV(model, param_grid={}, scroing='neg_mean_squared_error', cv=2)
#     gs.fit(x_train, y_train)

#     x_test = df.drop(columns=['SalePrice'], axis=1).iloc[vidx]
#     y_test = df['SalePrice'].iloc[vidx]
#     score = r2_score(y_test, x_test)
#     scores.append(score)
#     best_models.append(gs.best_estimator_)

#%%
# 5. Find some important feature and go back to 2.
fig, ax = plt.subplots(1, len(best_models))
fig.set_figwidth(15)
for idx, model in enumerate(best_models):
    print(putil.get_rmse(model, x_test, y_test))
    coef = pd.Series(model.coef_, index=x_train.columns)
    codef = coef.sort_values(ascending=False)
    coef_high = codef.head(10)
    coef_low = codef.tail(10)
    coefs = pd.concat([coef_high, coef_low])
    # coef_high[::-1].plot.barh(ax =ax[idx])
    sns.barplot(x=coefs.values, y=coefs.index, ax=ax[idx])

# 6. Final parameter tuning one by one with fixed (with validation curve)
#%%
# Another Model (Step in 3)
xgb_reg = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, tree_method='gpu_hist', colsample_bytree=0.5,subsample=0.8)
xgb_reg.fit(x_train, y_train)
print('train error : ', r2_score(y_train, xgb_reg.predict(x_train)))
print('validation error : ', r2_score(y_test, xgb_reg.predict(x_test)))
xgb.plot_importance(xgb_reg, max_num_features=20, height=0.4)

y_pred = xgb_reg.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
print('xgb_reg rmse : ', np.sqrt(mse))

lgbm_reg = lgbm.LGBMRegressor(n_estimators=1000, learning_rate=0.05, num_leaves=4,\
subsample=0.6, colsample_bytree=0.4, reg_lambda=10)
lgbm_reg.fit(x_train, y_train)
print('train error : ', r2_score(y_train, lgbm_reg.predict(x_train)))
print('validation error : ', r2_score(y_test, lgbm_reg.predict(x_test)))
lgbm.plot_importance(lgbm_reg, max_num_features=20, height=0.4)

y_pred = lgbm_reg.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
print('lgbm_reg rmse : ', np.sqrt(mse))

#%%
# 7. Ensemble all good models
estimators = []
for b in best_models:
    estimators.append((b.__class__.__name__, b))
estimators.append(('xgb', xgb_reg))
estimators.append(('lgbm', lgbm_reg))
final_reg = StackingRegressor(estimators=estimators, final_estimator=RandomForestRegressor(n_estimators=10))
final_reg.fit(x_train, y_train)
mse = mean_squared_error(y_test, final_reg.predict(x_test))
rmse = np.sqrt(mse)
print('final reg rmse : ', rmse)
#pickle.dump(final_reg, open('final.plk', 'wb'))
#final_reg = pickle.load(open('final.plk', 'rb'))


# Real value : np.expm1(final_reg.predict(x_test))
'''
# When trying to plot directly, dd[0] will be feature series with importance
dic = reg.get_booster().get_score()
dd = pd.DataFrame.from_dict(dic, orient='index')
dd[0].sort_values(ascending=False)
'''

'''
GridsearchCv(model, ...)
model should inherit BaseEstimator and some mixin
BaseEstimator (for get, set params)
ClassifierMixin, RegressorMixin (and should implt fit(return self), predict)
transformermixin(if define fit, transform, automatically fit_transform)

kerasClassifier(Regressor) with build_fn can be good
https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
'''

# %%

# %%
