# Store sales
## Overview

Data is from Kaggle:
https://www.kaggle.com/c/competitive-data-science-predict-future-sales

The dataset is collected over 30+ months and includes data from 60 stores and thousands of items. The objective is to predict store/item sale combination for next month. The dataset required deep analysis and segmentation as it contains elements (e.g. online stores) very different from the rest of the data which needed to be identified and addressed.

In addition, the original data was not feature-rich, the models performed poorly initially, and getting better performance called for generation of many advanced new features.


## Architecture

In this case model stacking will be a likely approach as the results appeared promising based on a small sample, and given the disparity of the data set. So far, I have fitted linear regression model, gradient boosted decision tree model and an LSTM. LN and GBDT are built on the same feature set. LSTM has built on a smaller feature set

## Data cleaning

Retail_exploratory.pynb.ipynb or Retail_exploratory.pynb.pdf explains what has been done and why. The examples include: clipping outliers, generating new features including time-lagged features and target encodings, running a tSNE visualization for identifying cross-store cluster similarities, and normalization.


## Training

The script runs linear regression and GBDT. The latter proved to be very productive after the addition of new features, which I continue to work on. There are several outlier stores, in particular store 12 which has a tendency to badly throw off the model when bundled with the rest of the data.

I also built an LSTM script, which I don't expect to be used across the full data set, but it may turn to be productive on its stubborn segments. LSTM is unlikely to be a good fit in this case given how long it takes to train and the size of the data set (~3m million transactions and many more store/item/month combinations. At any rate, I ran it on a small sample/small feature set to test; it showed a steady accuracy improvement on the val data set, but failed to reach the level of linear regression R2. Again, this may be a problem the smaller data set caused, and can't be solved anyway without a larger computational budget. At any rate, GBDT is recognized to work best for similar applications, and it certainly behaves very well here.

## Results

I initially considered segmenting the data (i.e. there are chunks of data that perform radically differently, although clipping the outliers appears to take care of the problem). My intent was to build the 'good' set and then update it with selected predictions from the consistent outliers in the 'bad' set. I was surprised to learn that the results got a lot worse compared to my running the predictions based on the full dataset (I picked a handful of high-volume items which came consistently high in the previous months, built separate predictions for them, and updated the results). I conclude that the test set has these items clipped. This makes the test data likely inconsistent with the train data, but actually simplifies things going forward as I will deal with a full dataset at the same time, and basically focus on the correct outlier cut off and model training and maybe add some new features. 

At any rate, I'm #16 on the leaderboard with 0.976 RMSE, which puts in in the ballpark where I wanted to get. I will continue updating the model if time permits, and basically try to get into the top 10 just to make the point.

Also, the results are currently based on GBDT alone. I tried stacking as simple weighted optimization of two models' predictions and a linear regression of meta features produced by the models. Neither proved to be productive yet, as GBDT does better solo. An addition of another model may change things.


