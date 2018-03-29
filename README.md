# Store sales
## Overview

Data is from Kaggle:
https://www.kaggle.com/c/competitive-data-science-predict-future-sales

The dataset is collected over 30+ months and includes data from 60 stores and thousands of items. The objective is to predict store/item sale combination for next month. The dataset required deep analysis and segmentation as it contains elements (e.g. online stores) very different from the rest of the data which needed to be isolated and treated individually.

In addition, the original data was not feature-rich, the models did not show good results and getting better performance called for generation of many advanced new features.


## Architecture

In this case model stacking will be a likely approach as the results appeared promising based on a small sample, and given the disparity of the data set. So far, I have fitted linear regression model, gradient boosted decision tree model and an LSTM. LN and GBDT are built on the same feature set. LSTM is built on a smaller feature set

## Data cleaning

Retail_exploratory.pynb.ipynb or Retail_exploratory.pynb.pdf explains what has been done and why. The examples include: clipping outliers, generating new features including time-lagged features and target encodings, running a tSNE visualization for identifying cross-store cluster similarities, and normalization.


## Training

The script runs linear regression and GBDT. The latter proved to be very productive after the addition of new features, which I continue to work on. There are several outlier stores, in particular store 12 which has a tendency to badly throw off the model when bundled with the rest of the data. So far, the RMSE has reach ~0.9 on the validation set with improvement headroom left (without stacking yet).

I also built an LSTM script, which I don't expect to be used across the full data set, but it may turn to be productive on its stubborn segments. LSTM is unlikely to be a good fit in this case given how long it takes to train and the size of the data set (~3m million transactions and many more store/item/month combinations. At any rate, I ran it on a small sample/small feature set to test; it showed a steady accuracy improvement on the val data set, but failed to reach the level of linear regression R2. Again, this may be a problem the smaller data set caused, and can't be solved anyway without a larger computational budget. At any rate, GBDT is recognized to work best for similar applications, and it certainly behaves very well here.

## Results

So far I have trained the models on the full data set, without trying to segment it, which would be my next thing to do. I also tried stacking as simple weighted optimization of two models' predictions and a linear regression of meta features produced by the models. Neither proved to be productive yet: the weighting came 100 percent in favor of GBDT, and stacked model gave a worse overall outcome. 

Next (in this order):
*  segmenting training data
*  additional features
*  additional model (maybe)


