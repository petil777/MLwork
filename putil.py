import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import validation_curve

def show_distrib(df, st=0, ed=0, col_names=[]):
    #Use muliple subplots to see in different plots
    for col in col_names:
        fig,ax=plt.subplots(1, 1)
        sns.distplot(df[col], ax = ax)
    return

def show_values(df, st=0, ed=0, col_names=[]):
    for col in col_names:
        print('=======================')
        print(df[col].value_counts())
    return

def null_zero_else_one(df, st=0, ed=0, col_names=[]):
    for col in col_names:
        df[col][~df[col].isnull()] = 1
        df[col][df[col].isnull()] = 0
        if df[col].dtype=='float64':
            df[col] = df[col].astype(str).astype(float)
        else: #Object or already int
            df[col] = df[col].astype(str).astype(int)

#Lean towards oneside
def log_scale_change(df, col_names=[]):
    for col in col_names:
        df[col] = np.log1p(df[col])

#Min is 0, max is 1
def minmax_scale_change(df, col_names=[]):
    mms = MinMaxScaler()
    for col in col_names:
        df[col] = mms.fit_transform(df[col].values.reshape(-1, 1))
        df[col] = df[col].astype(str).astype(float)
#median 0, IQR(interquartile range) 1
def robust_scale_change(df, col_names=[]):
    rob = RobustScaler()
    for col in col_names:
        df[col] = rob.fit_transform(df[col].values.reshape(-1, 1))
        df[col] = df[col].astype(str).astype(float)
#mean 0, standrad 1
def standard_scale_change(df, col_names=[]):
    sc = StandardScaler()
    for col in col_names:
        df[col] = sc.fit_transform(df[col].values.reshape(-1, 1))
        df[col] = df[col].astype(str).astype(float)

def get_rmse(model, x_test, y_test):
    pred = model.predict(x_test)
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    return rmse


#learning curve

#validation curve (for one hyper parameter tuning)

#silhouette graph (for unsupervised clustering)