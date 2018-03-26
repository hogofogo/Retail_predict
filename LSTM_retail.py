#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 10:24:24 2018

@author: vlad
"""

import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
import gc
from tqdm import tqdm_notebook
from itertools import product
import sklearn
import scipy.sparse 
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from keras.models import Model, Input
from keras.layers import Dense
from keras.layers import LSTM
from pandas import Series, DataFrame, concat
from keras.models import Sequential
from sklearn import preprocessing
from numpy import inf
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split



def get_data():  
    DATA_FOLDER = '~/.kaggle/competitions/competitive-data-science-final-project/'

    transactions    = pd.read_csv(os.path.join(DATA_FOLDER, 'sales_train.csv'))
    items           = pd.read_csv(os.path.join(DATA_FOLDER, 'items.csv'))
    item_categories = pd.read_csv(os.path.join(DATA_FOLDER, 'item_categories.csv'))
    shops           = pd.read_csv(os.path.join(DATA_FOLDER, 'shops.csv'))
    test            = pd.read_csv(os.path.join(DATA_FOLDER, 'test.csv'))


  
    #REPLACE DATE STRING WITH DATETIME
    transactions['date'] = pd.Series([datetime.strptime(d, '%d.%m.%Y') for d in transactions['date']])
    #transactions['day'] = pd.Series([t.day for t in transactions['date']])
    transactions['month'] = pd.Series([t.month for t in transactions['date']])
    #transactions['year'] = pd.Series([t.year for t in transactions['date']])
    transactions['revenue'] = transactions.item_price * transactions.item_cnt_day
    transactions = pd.merge(transactions, items, how='left', on='item_id')
    transactions = pd.merge(transactions, shops, how='left', on='shop_id')
    transactions.drop(labels = ['item_name', 'shop_name'], axis =1, inplace = True)

    
    #clip sales price outliers
    lowerbound, upperbound = np.percentile(transactions['item_price'], [1,99])
    transactions['item_price'] = np.clip(transactions['item_price'], lowerbound, upperbound)
    #transactions['item_price'].hist(bins = 30)

    return transactions, items



def downcast_dtypes(df):
    '''
        Changes column types in the dataframe: 
                
                `float64` type to `float32`
                `int64`   type to `int32`
    '''
    
    # Select columns to downcast
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols =   [c for c in df if df[c].dtype == "int64"]
    
    # Downcast
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols]   = df[int_cols].astype(np.int32)
    
    return df

#BULID DATA SET
#transactions includes all stores from the data set
transactions, items = get_data()

sales = transactions[transactions['shop_id'].isin([24,58,15,26,7])]

# Create "grid" with columns
index_cols = ['shop_id', 'item_id', 'date_block_num']

# For every month we create a grid from all shops/items combinations from that month
grid = [] 
for block_num in sales['date_block_num'].unique():
    cur_shops = sales.loc[sales['date_block_num'] == block_num, 'shop_id'].unique()
    cur_items = sales.loc[sales['date_block_num'] == block_num, 'item_id'].unique()
    grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])),dtype='int32'))

# Turn the grid into a dataframe
grid = pd.DataFrame(np.vstack(grid), columns = index_cols,dtype=np.int32)

# Groupby data to get shop-item-month aggregates
gb = sales.groupby(index_cols,as_index=False).agg({'item_cnt_day':{'target':'sum'}})
# Fix column names
gb.columns = [col[0] if col[-1]=='' else col[-1] for col in gb.columns.values] 
# Join it to the grid
all_data = pd.merge(grid, gb, how='left', on=index_cols).fillna(0)

#!!! Same as above but with price
gb = sales.groupby(['item_id','shop_id','date_block_num'],as_index = False).agg({'item_price':{'target_price':'mean'}})
gb.columns = [col[0] if col[-1] == '' else col[-1] for col in gb.columns.values]
all_data = pd.merge(all_data, gb, how='left', on=['item_id', 'shop_id','date_block_num']).fillna(0)


#!!! include price and price change
cols_to_rename = list(all_data.columns.difference(index_cols)) 
shift_range = [1]
for month_shift in tqdm_notebook(shift_range):
    train_shift = all_data[index_cols + cols_to_rename].copy()
    
    train_shift['date_block_num'] = train_shift['date_block_num'] + month_shift
    
    foo = lambda x: '{}_lag_{}'.format(x, month_shift) if x in cols_to_rename else x
    train_shift = train_shift.rename(columns=foo)

    all_data = pd.merge(all_data, train_shift, on=index_cols, how='left').fillna(0)

del train_shift
del(cur_items, cur_shops)


all_data['price_diff'] = all_data['target_price'] / all_data['target_price_lag_1']


#now drop the lagged columns that were only necessary at this point
#to create price diff feature
all_data.drop(['target_lag_1', 'target_price_lag_1'], axis = 1, inplace = True)


# Downcast dtypes from 64 to 32 bit to save memory
all_data = downcast_dtypes(all_data)
del grid, gb 
gc.collect();




#now create 12m time series for individual columns



# List of columns that we will use to create lags
cols_to_rename = list(all_data.columns.difference(index_cols)) 

cols_to_rename = ['target']
shift_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

#for earch column create a separate data frame with time series
#!!!quick, dirty and lazy = make a loop from the below
for month_shift in tqdm_notebook(shift_range):
    train_shift = all_data[index_cols + cols_to_rename].copy()
    
    train_shift['date_block_num'] = train_shift['date_block_num'] + month_shift
    
    foo = lambda x: '{}_lag_{}'.format(x, month_shift) if x in cols_to_rename else x
    train_shift = train_shift.rename(columns=foo)

    all_data = pd.merge(all_data, train_shift, on=index_cols, how='left').fillna(0)

cols_to_rename = ['target_price']
shift_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

#for earch column create a separate data frame with time series

for month_shift in tqdm_notebook(shift_range):
    train_shift = all_data[index_cols + cols_to_rename].copy()
    
    train_shift['date_block_num'] = train_shift['date_block_num'] + month_shift
    
    foo = lambda x: '{}_lag_{}'.format(x, month_shift) if x in cols_to_rename else x
    train_shift = train_shift.rename(columns=foo)

    all_data = pd.merge(all_data, train_shift, on=index_cols, how='left').fillna(0)


cols_to_rename = ['price_diff']
shift_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

#for earch column create a separate data frame with time series

for month_shift in tqdm_notebook(shift_range):
    train_shift = all_data[index_cols + cols_to_rename].copy()
    
    train_shift['date_block_num'] = train_shift['date_block_num'] + month_shift
    
    foo = lambda x: '{}_lag_{}'.format(x, month_shift) if x in cols_to_rename else x
    train_shift = train_shift.rename(columns=foo)

    all_data = pd.merge(all_data, train_shift, on=index_cols, how='left').fillna(0)


# Don't use old data from year 2013
all_data = all_data[all_data['date_block_num'] >= 12] 

#add column replicas to all_data
temp = all_data['shop_id']
for i in range((12-1)):
    temp = pd.concat([temp, all_data['shop_id']], axis =1)

all_data = pd.concat([all_data, temp], axis = 1)
del temp

temp = all_data['item_id']
for i in range((12-1)):
    temp = pd.concat([temp, all_data['item_id']], axis =1)

all_data = pd.concat([all_data, temp], axis = 1)
del temp


df = np.array(all_data)
df = df[:,6:]
#replace inf with 0
df[df == -inf] = 0
df[df == inf] = 0
#replace nan with 0
df = np.array(df)
df = np.nan_to_num(df)
df = df.astype(dtype = 'float32')

# scale data to [-1, 1]
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(df)
df_scaled = scaler.transform(df)

y = all_data['target']

#shuffle X and y
X, y = shuffle(df_scaled, y, random_state=0)

#transform to shape required by lstm
X = X.reshape(len(X),5,12)
X = np.swapaxes(X, 1,2)

#split X and y into testing and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

batch_size = 1
nb_epoch = 5
neurons = 50

#BUILD AN LSTM MODEL

#def fit_lstm(train, batch_size, nb_epoch, neurons):
#	X, y = train[:, 0:-1], train[:, -1]
#	X = X.reshape(X.shape[0], 1, X.shape[1])
model = Sequential()
model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
for i in range(nb_epoch):
		model.fit(X_train, y_train, epochs=1, batch_size=batch_size, shuffle=False)
		model.reset_states()
#return model


pred_lstm_test = model.predict(X_test, batch_size=batch_size)
pred_lstm_train = model.predict(X_train, batch_size=batch_size)

print('Train R-squared for lstm is %f' % r2_score(y_train, pred_lstm_train))
print('Test R-squared for lstm is %f' % r2_score(y_test, pred_lstm_test))






lr = LinearRegression()
lr.fit(X_train.values, y_train)
pred_lr_test = lr.predict(X_test.values)
pred_lr_train = lr.predict(X_train.values)

print('Train R-squared for linreg is %f' % r2_score(y_train, pred_lr_train))
print('Test R-squared for linreg is %f' % r2_score(y_test, pred_lr_test))








