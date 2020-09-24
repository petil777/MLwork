#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
import putil
from scipy.stats import skew
import tensorflow as tf
from tensorflow.keras import Model
import os
from tensorflow.keras import backend as K

from sklearn.base import BaseEstimator
df = pd.read_csv('./data/train.csv')

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

# 2. Filling, Encoding, Scaling, Outlier (Come back after test some model)
# Filling
df.fillna(df.mean(), inplace=True)
#df[col].fillna(df[col].value_counts().index[0], inplace=True) # For most_frequent

# Outlier Detection (before scaling. watch among important features....by ridge, lasso, linearReg, etc.)
# sns.regplot(x='GrLivArea', y='SalePrice', data = df)
cond1 = df['GrLivArea'] > 4000
cond2 = df['SalePrice'] < 500000
outlier_index = df[cond1 & cond2].index
df.drop(outlier_index, axis=0, inplace=True)

# Scaling (numerical)
feature_index = df.dtypes[df.dtypes!='object'].index
skew_features = df[feature_index].apply(lambda x : skew(x))
skew_features_top = skew_features[skew_features > 1].sort_values(ascending=False)
df[skew_features_top.index] = np.log1p(df[skew_features_top.index])

# Encoding (string object)
df = pd.get_dummies(df)


# x_train = df.drop(columns=['SalePrice'], axis=1)
# y_train = df['SalePrice']
x_train, x_test, y_train, y_test = train_test_split(df.drop(columns=['SalePrice'], axis=1), df['SalePrice'], train_size=0.7 )#radom_state = 

# %%
# Function API version (useful for quick test. But not compatible with StackRegressor for get_params absent...)
def create_model(NN1, NN2, NN3, drop_rate, drop_rate2):
    K.clear_session()
    inp = tf.keras.Input(shape=(df.columns.shape[0]-1,))#without batch!
    out1 = tf.keras.layers.Dense(NN1, activation=tf.nn.tanh)(inp)
    drop1 = tf.keras.layers.Dropout(drop_rate)(out1)
    out2 = tf.keras.layers.Dense(NN2, activation=tf.nn.leaky_relu)(drop1)
    drop2 = tf.keras.layers.Dropout(drop_rate2)(out2)
    out3 = tf.keras.layers.Dense(NN3)(drop2)
    out = tf.keras.layers.Dense(1, activation='linear')(out3)

    model = tf.keras.Model(inputs=inp, outputs=out)
    model.summary()
    model.compile(optimizer = tf.keras.optimizers.Adam(), loss=tf.keras.losses.msle)
    return model
Rmodel = tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn=create_model, batch_size=10, epochs=30)

#%%
# SubClass version
class MyModel(tf.keras.Model, BaseEstimator):
    def __init__(self, NN1, NN2, NN3, drop_rate, drop_rate2):
        super(MyModel, self).__init__()
        self.out1 = tf.keras.layers.Dense(NN1, activation=tf.nn.tanh)
        self.dropout = tf.keras.layers.Dropout(drop_rate)
        self.out2 = tf.keras.layers.Dense(NN2, activation=tf.nn.leaky_relu)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate2)
        self.out3 = tf.keras.layers.Dense(NN3)
        self.out = tf.keras.layers.Dense(1, activation='linear')
    # If set training = True, dropout will be used also in predict. 
    # fit automatically apply training=true to dropout, predict(evaluate) automatically ignore "dropout layer" and training=False    
    def __call__(self, inputs, training):
        o1 = self.out1(inputs)
        d1 = self.dropout(o1)
        o2 = self.out2(d1)
        d2 = self.dropout2(o2)
        o3 = self.out3(d2)
        return self.out(o3)

def create_model_instance(NN1, NN2, NN3, drop_rate, drop_rate2):
    K.clear_session()
    mymodel = MyModel(NN1, NN2, NN3, drop_rate, drop_rate2)
    mymodel.compile(optimizer = tf.keras.optimizers.Adam(), loss=tf.keras.losses.msle)
    return mymodel

Rmodel = tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn=create_model_instance, batch_size=10, epochs=50)
# %%
checkpoint_filepath = './checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='auto',
    save_best_only=True)
#%%
# gs = GridSearchCV(estimator=Rmodel, param_grid={'epochs' : [15, 30, 50, 100, 200], 'batch_size' : [5, 10]}, cv=2, scoring='neg_mean_squared_error')
gs = RandomizedSearchCV(estimator=Rmodel, \
    param_distributions={'drop_rate' : [0.2, 0.5, 0.8], 'drop_rate2':[0, 0.2, 0.5, 0.8],\
        'NN1':[300, 500, 700], 'NN2' : [100, 300, 500, 700, 900], 'NN3': [20, 50, 100]},\
            cv=3, scoring='neg_mean_squared_error', n_iter=1)

# {'drop_rate2': 0.2, 'drop_rate': 0.2, 'NN3': 50, 'NN2': 700, 'NN1': 300} -0.06169853871182165
# rmse :  0.38203848699753357

# %%
history = gs.fit(x_train, y_train, validation_data=(x_test, y_test) ,callbacks=[model_checkpoint_callback])

mse = mean_squared_error(y_test, gs.predict(x_test))
print('rmse : ', np.sqrt(mse))
# %%

# 
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
import pickle

final_reg = pickle.load(open('final.plk', 'rb'))

gs.best_estimator_._estimator_type="regressor" # https://github.com/keras-team/keras/issues/13669 => critical bug of sklearn
ee = []
ee.append(('final', final_reg))
ee.append(('dnn', gs.best_estimator_))
fff = StackingRegressor(estimators=ee, final_estimator=RandomForestRegressor(n_estimators=5))
fff.fit(x_train, y_train)