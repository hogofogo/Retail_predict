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
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error



def get_data():  
    DATA_FOLDER = '~/.kaggle/competitions/competitive-data-science-final-project/'

    transactions    = pd.read_csv(os.path.join(DATA_FOLDER, 'sales_train.csv'))
    items           = pd.read_csv(os.path.join(DATA_FOLDER, 'items.csv'))
    item_categories = pd.read_csv(os.path.join(DATA_FOLDER, 'item_categories.csv'))
    shops           = pd.read_csv(os.path.join(DATA_FOLDER, 'shops.csv'))
    test            = pd.read_csv(os.path.join(DATA_FOLDER, 'test.csv'))


  
    #REPLACE DATE STRING WITH DATETIME
    transactions['date'] = pd.Series([datetime.strptime(d, '%d.%m.%Y') for d 
                in transactions['date']])
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

    return transactions, items, test



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



def get_all_data():

    #This is also where all the features get added that need to be compared month
    #to month
    # Create "grid" with columns

    # For every month create a grid from all shops/items combinations from that month

    grid_left = sales[['date_block_num', 'shop_id', 'item_id']]
    grid_right = test[['shop_id', 'item_id']]
    #this is required to replicate the content of submission. putting all permutations
    #under 33 allows to build the time series for the combinations of interest
    grid_right['date_block_num'] = 34
    grid_construct = pd.merge(grid_left, grid_right, how = 'outer', on=['shop_id', 'item_id', 
                    'date_block_num']).drop_duplicates()
    
    grid = [] 
    for block_num in grid_construct['date_block_num'].unique():
        cur_shops = grid_construct.loc[grid_construct['date_block_num'] == 
                                       block_num, 'shop_id'].unique()
        cur_items = grid_construct.loc[grid_construct['date_block_num'] == 
                                       block_num, 'item_id'].unique()
        grid.append(np.array(list(product(*[cur_shops, cur_items, 
                                            [block_num]])),dtype='int32'))

    # Turn the grid into a dataframe
    grid = pd.DataFrame(np.vstack(grid), columns = index_cols,dtype=np.int32)

    # Groupby data to get shop-item-month aggregates
    gb = sales.groupby(index_cols,as_index=False).agg({'item_cnt_day':{'target':'sum'}})

    # Fix column names
    gb.columns = [col[0] if col[-1]=='' else col[-1] for col in gb.columns.values] 

    # Join it to the grid
    all_data = pd.merge(grid, gb, how='left', on=index_cols).fillna(0)

    # Add shop-month aggregates
    gb = sales.groupby(['shop_id', 'date_block_num'],as_index=False).agg({'item_cnt_day':{'target_shop':'sum'}})
    gb.columns = [col[0] if col[-1]=='' else col[-1] for col in gb.columns.values]
    all_data = pd.merge(all_data, gb, how='left', on=['shop_id', 'date_block_num']).fillna(0)

    # Add item-month aggregates
    gb = sales.groupby(['item_id', 'date_block_num'],as_index=False).agg({'item_cnt_day':{'target_item':'sum'}})
    gb.columns = [col[0] if col[-1] == '' else col[-1] for col in gb.columns.values]
    all_data = pd.merge(all_data, gb, how='left', on=['item_id', 'date_block_num']).fillna(0)

    #Add price
    gb = sales.groupby(['item_id','shop_id','date_block_num'],as_index = False).agg({'item_price':{'target_price':'mean'}})
    gb.columns = [col[0] if col[-1] == '' else col[-1] for col in gb.columns.values]
    all_data = pd.merge(all_data, gb, how='left', on=['item_id', 'shop_id','date_block_num']).fillna(0)
    #price change would also be interesting - will be setup automatically below

    # Downcast dtypes from 64 to 32 bit to save memory
    all_data = downcast_dtypes(all_data)

    return all_data



def get_lags(all_data, shift_range):

    #this makes sure that the features in iput data frame get their lag counterparts
    # List of columns that we will use to create lags
    cols_to_rename = list(all_data.columns.difference(index_cols)) 


    for month_shift in tqdm_notebook(shift_range):
        train_shift = all_data[index_cols + cols_to_rename].copy()
        
        train_shift['date_block_num'] = train_shift['date_block_num'] + month_shift
        
        foo = lambda x: '{}_lag_{}'.format(x, month_shift) if x in cols_to_rename else x
        train_shift = train_shift.rename(columns=foo)

        all_data = pd.merge(all_data, train_shift, on=index_cols, how='left').fillna(0)

    del train_shift

    return all_data



def clean_price(all_data):
    #this was done because it appeared easier to clean up unnecessary price-related
    #columns after lags than as a separate function
    temp = list(all_data.columns)
    price_col_list = []
    for i in range(len(temp)):
        if 'target_price' in temp[i]:
            price_col_list.append(temp[i])
               
    for i in range(len(price_col_list)-1):
        gc = all_data[price_col_list[i]] / all_data[price_col_list[i+1]]
        gc = gc.rename('price_diff_' + str(i))
        gc = gc.replace([np.inf, -np.inf], np.nan)
        all_data = pd.merge(all_data, gc.to_frame(), how='left',  
                            left_index = True, right_index = True).fillna(0)


    #drop undesired time and time_diff columns
    #time_data_cols = list(all_data.columns)
    cols_to_kill = ['target_price_lag_1', 'target_price_lag_2', 
                    'target_price_lag_3','target_price_lag_4', 
                    'target_price_lag_5', 'target_price_lag_12', 
                    'price_diff_1', 'price_diff_2', 'price_diff_3', 
                    'price_diff_4', 'price_diff_5']

    all_data.drop(cols_to_kill, axis = 1, inplace = True)
    
    all_data = downcast_dtypes(all_data)

    return all_data



def add_solitary_features(all_data):
    #THIS IS WHERE YOU ADD ALL THE ADDITIONAL FEATURES AFTER ALL
    #THE LAGGED FEATURES HAVE BEEN ADDED!!! add shop city
    gb = sales.groupby(['shop_id'],as_index=False).agg({'shop_city':{'target_city':'min'}})
    gb.columns = [col[0] if col[-1] == '' else col[-1] for col in gb.columns.values]
    all_data = pd.merge(all_data, gb, how='left', on=['shop_id']).fillna(0)


    #weight of per shop revenue in total revenue
    gb = sales.groupby(['shop_id'], as_index = False).agg({'revenue':{'rev_weight':'sum'}})
    gb['revenue'] = gb['revenue']/gb.sum()[1]
    all_data = pd.merge(all_data, gb, how='left', on=['shop_id']).fillna(0)

    '''
    #frequency of transactions per category id
    gb = sales.groupby(['item_category_id'], as_index = False).agg({'item_cnt_day':{'cat_freq':'sum'}})
    gb['item_cnt_day'] = gb['item_cnt_day']/gb.sum()[1]
    all_data = pd.merge(all_data, gb, how='left', on=['item_category_id']).fillna(0)

    #frequency of transactions per item id
    gb = sales.groupby(['item_id'], as_index = False).agg({'item_cnt_day':{'item_freq':'sum'}})
    gb['item_cnt_day'] = gb['item_cnt_day']/gb.sum()[1]
    all_data = pd.merge(all_data, gb, how='left', on=['item_id']).fillna(0)
    '''

    #add mean encodings for target
    all_data['item_target_enc'] = all_data.groupby('item_id')['target'].transform('mean')
    all_data['item_target_enc'].fillna(0.3343, inplace=True) 


    all_data['item_cat_enc'] = all_data.groupby('item_category_id')['target'].transform('mean')
    
    
    all_data['city_enc'] = all_data.groupby('target_city')['target'].transform('mean')

    return all_data




def prepare_tt_sets(all_data, dates, last_block):


    y_train = all_data.loc[dates <  (last_block), 'target'].values
    y_test =  all_data.loc[dates == (last_block), 'target'].values
    

    #y_test12 =  all_data.loc[dates == last_block]

    #scale features
    all_data_scaled = preprocessing.scale(all_data)
    all_data_scaled = pd.DataFrame(all_data_scaled)
    replace_col_names = list(all_data)
    all_data_scaled.columns = replace_col_names

    X_train = all_data_scaled.loc[dates <  (last_block)].drop(to_drop_cols, axis=1)
    X_test =  all_data_scaled.loc[dates == (last_block)].drop(to_drop_cols, axis=1)
    #X_to_predict = all_data_scaled.loc[dates == last_block].drop(to_drop_cols, axis=1)

    #y_test12 = all_data12['target'].values
    #X_test12 = all_data12.drop(to_drop_cols, axis=1)

    return X_train,X_test,y_train,y_test



#BULID DATA SET
#transactions includes all stores from the data set
transactions, items, test = get_data()

#select desired stores from the options below or another combination:
#this excludes stores that tend to ubalance the set
#in this case, build a full data set that can be updated with data prediction segments as needed
#sales = transactions[~transactions['shop_id'].isin([9])] #shop 9 is not in the test set for submissiona and is bad => drop
sales = transactions
#sales = transactions[transactions['shop_id'].isin([24,58,15,26,7,38,19,21,43,56,16,29,53,14,30,41,37,59,52,2,45,4,5,44,3,17,48,51,49,10,39,34,0,20,13,33,32,23,40,1,8,11,36,6,18,25, 27, 28,31,35,42,46,47,50,54,57])]
#sales = transactions[~transactions['shop_id'].isin([12,22,9,55])]
#this subset gives high predictability
#sales = transactions[transactions['shop_id'].isin([31,25,28,42,54,27,57,6,18,50,47,46,35])]
#this is the cluster around shop12 suggested by tSNE
#sales = transactions[transactions['shop_id'].isin([7,12,15,16,18,19])]

#!!!all_data[all_data['date_block_num'] == 33].count()

#define index columns
index_cols = ['shop_id', 'item_id', 'date_block_num']

#build data file
all_data = get_all_data()

#set shift for number of months to lag
shift_range = [1, 2, 3, 4, 5, 12]

#create lags for the given data set
all_data = get_lags(all_data, shift_range)

#remove unnecessary price columns
all_data = clean_price(all_data)

# Drop old data from 2013
all_data = all_data[all_data['date_block_num'] >= 12] 

#clip count outliers at 60
lowerbound, upperbound = np.percentile(all_data['target'], [0,99.95])
all_data['target'] = np.clip(all_data['target'], lowerbound, upperbound)
all_data['target_lag_1'] = np.clip(all_data['target_lag_1'], lowerbound, upperbound)

# List of all lagged features
fit_cols = [col for col in all_data.columns if col[-1] in [str(item) for item in shift_range]] 
# We will drop these at fitting stage
to_drop_cols = list(set(list(all_data.columns)) - (set(fit_cols)|set(index_cols))) + ['date_block_num'] 

# Category for each item
item_category_mapping = items[['item_id','item_category_id']].drop_duplicates()

all_data = pd.merge(all_data, item_category_mapping, how='left', on='item_id')
all_data = downcast_dtypes(all_data)
del(col, item)
gc.collect();

#add solitary features after lagged features
all_data = add_solitary_features(all_data)

# Save `date_block_num`, as we can't use them as features, but will need them to split the dataset into parts 
dates = all_data['date_block_num']

last_block = dates.max()
print('Test `date_block_num` is %d' % last_block)


X_train,X_test,y_train,y_test = prepare_tt_sets(all_data, dates, last_block)

#!!!this is where insert is made this is a catch-all prediction based on the full data set
#load X and y
#X_train_segm, y_train_segm, X_to_predict = prepare_final_set(all_data)

#RUN LINEAR REGRESSION
lr = LinearRegression()
lr.fit(X_train.values, y_train)
pred_lr_test = lr.predict(X_test.values)
pred_lr_train = lr.predict(X_train.values)

print('Train R-squared for linreg is %f' % r2_score(y_train, pred_lr_train))
#print('Test R-squared for linreg is %f' % r2_score(y_test, pred_lr_test))
#mean_squared_error(y_test, pred_lr_test)



lgb_params = {
               'feature_fraction': 0.3,
               'metric': 'rmse',
               'nthread':1, 
               'min_data_in_leaf': 1, 
               'bagging_fraction': 0.3, 
               'learning_rate': 0.03, 
               'objective': 'mse', 
               'bagging_seed': 2**7, 
               'num_leaves': 2**7,
               'bagging_freq':1,
               'verbose':0,
               'num_iterations':130,
               'max_depth':11
              }

#fine-tuned predictions for data segments will be done below
model = lgb.train(lgb_params, lgb.Dataset(X_train, label=y_train), 100)
pred_lgb_train = model.predict(X_train)
pred_lgb_test = model.predict(X_test)

print('Train R-squared for LightGBM is %f' % r2_score(y_train, pred_lgb_train))
#print('Test R-squared for LightGBM is %f' % r2_score(y_test, pred_lgb_test))
#mean_squared_error(y_test, pred_lgb_test)


#concatenate test predictions for meta featrues testing
X_test_level2 = np.c_[pred_lr_test, pred_lgb_test]

#get predictins from models for months 27-32 using same parameters.
dates_train = dates[dates <  (last_block)]
dates_test  = dates[dates == (last_block)]
dates_train_level2 = dates_train[dates_train.isin([28, 29, 30, 31, 32, 33])]

# That is how we get target for the 2nd level dataset
y_train_level2 = y_train[dates_train.isin([28, 29, 30, 31, 32, 33])]


# Prepare 2nd level feeature matrix, init it with zeros first
X_train_level2 = np.zeros([y_train_level2.shape[0], 2])


# fill `X_train_level2` with metafeatures
for cur_block_num in [28, 29, 30, 31, 32, 33]:
    
    print(cur_block_num)
    
    '''
        Split `X_train` into parts
        Fit linear regression 
       3. Fit LightGBM and put predictions          
       4. Store predictions from 2. and 3. in the right place of `X_train_level2`. 
     '''      
    
    X_train_block = X_train[dates_train < cur_block_num]
    y_train_block = y_train[dates_train < cur_block_num]
    X_test_block = X_train[dates_train == cur_block_num]
    y_test_block = y_train[dates_train == cur_block_num]

    lr.fit(X_train_block.values, y_train_block)
    pred_lr = lr.predict(X_test_block.values)
    print('Test R-squared for linreg is %f' % r2_score(y_test_block, pred_lr))

    model = lgb.train(lgb_params, lgb.Dataset(X_train_block.values, label=y_train_block), 100)
    pred_lgb = model.predict(X_test_block.values)

    print('Test R-squared for LightGBM is %f' % r2_score(y_test_block, pred_lgb))

    X_train_level2[dates_train_level2==cur_block_num] = np.c_[pred_lr, pred_lgb]


print(X_train_level2.mean(axis=0))




#linear model mix
alphas_to_try = np.linspace(0, 1, 1001)

best_alpha = float()
r2_train_simple_mix = float()

for alpha in alphas_to_try:
    
    mix = alpha * X_train_level2[:,0] + (1 - alpha) * X_train_level2[:,1]
    r2_step = r2_score(y_train_level2, mix)
    
    if r2_step > r2_train_simple_mix:
        best_alpha = alpha
        r2_train_simple_mix = r2_step
        
print('Best alpha: %f; Corresponding r2 score on train: %f' % (best_alpha, r2_train_simple_mix))



#use alpha to build test set prediction
lr.fit(X_train, y_train)
model = lgb.train(lgb_params, lgb.Dataset(X_train, label=y_train), 100)

test_preds = best_alpha * lr.predict(X_test) + (1 - best_alpha) * model.predict(X_test)


#predict y_value
submission = test_preds

submission = pd.DataFrame(submission)
X_submission =  all_data.loc[dates == (last_block)].drop(to_drop_cols, axis=1)

submission.index = X_submission.index
submission = pd.merge(submission, all_data.loc[: ,['shop_id', 'item_id']], 
                      how = 'left', left_index=True, right_index=True)


#save submission; I will need it later for baseline prediction
#it includes all the items that need to be submitted in the correct order
#submission.to_csv('~/Projects/Retail_predict/X_submission_baseline.csv', index = True)
#!!! now when test is merged with submission, first merge it with submission_baseline
# and then update the values with targeted model predictions

submission = pd.merge(test, submission, how = 'left', on=['shop_id', 'item_id']).fillna(999)
submission[submission[0] == 999].count()


#upload results to the correct segment of test

to_submit = submission.loc[:,['ID', 0]]
to_submit.columns = ['ID', 'item_cnt_month']
to_submit.to_csv('~/Projects/Retail_predict/submission.csv', index = False)


#LET'S TRY TO STACK THE MODELS AS WELL. THE SIMPLE WEIGHTED COMBINATION
#DID NOT PROVE USEFUL AS THE MODEL ALLOCATED ZERO TO LR

lr.fit(X_train_level2, y_train_level2)

train_preds = lr.predict(X_train_level2)
r2_train_stacking = r2_score(y_train_level2, train_preds)
print('Train R-squared for stacking is %f' % r2_train_stacking)

test_preds = lr.predict(X_test_level2)

submission = test_preds

submission = pd.DataFrame(submission)
X_submission =  all_data.loc[dates == (last_block)].drop(to_drop_cols, axis=1)
submission.index = X_submission.index
submission = pd.merge(submission, all_data.loc[: ,['shop_id', 'item_id']], 
                      how = 'left', left_index=True, right_index=True)


submission = pd.merge(test, submission, how = 'left', on=['shop_id', 'item_id']).fillna(999)
submission[submission[0] == 999].count()

#upload results to the correct segment of test

to_submit = submission.loc[:,['ID', 0]]
to_submit.columns = ['ID', 'item_cnt_month']
to_submit.to_csv('~/Projects/Retail_predict/submission.csv', index = False)












#IGNORE BELOW - THIS IS TEMPORARY CODE______________________________________
#build multiple linear regressions for item 20949
#first build all_data without clipping values
#This will need an autoregression model

temp1 = all_data[(all_data['item_id'] == 20949) & (all_data['date_block_num'] 
                 == 33)].sort_values(by = 'shop_id')
temp3 = temp1[['shop_id', 'item_id', 'target','target_lag_1','target_lag_2',
               'target_lag_3','target_lag_4','target_lag_5' ]]
y_temp  = temp3[['target_lag_5', 'target_lag_4','target_lag_3','target_lag_2', 'target_lag_1', 'target']]
res = y_temp.mean(axis = 1)
#remove one-off
res.iloc[8] = 0
submission12 = pd.concat([temp3[['shop_id', 'item_id']], res], axis = 1)




