import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def numbins(data, feature, plot):
    if plot == 'No':
        IQR = data[features].quantile(0.75) - data[features].quantile(0.25)
        bin_width = 2 * IQR / len(data)**(1/3)
        numbins = int((data[feature].max() - data[feature].min()) / bin_width)
        return numbins
    else:
        IQR = data.iloc[:, feature].quantile(0.75) - data.iloc[:, feature].quantile(0.25)
        bin_width = 2 * IQR / len(data)**(1/3)
        numbins = int((data.iloc[:, feature].max() - data.iloc[:, feature].min()) / bin_width)
        return numbins

def Histogram(data, feature, numberbins, Normalization):
    plt.hist(data.iloc[:, feature], bins = numberbins, edgecolor='black')
    # if Normalization == 'None':
    #     plt.title(f'Histogram of Feature {feature}')
    # elif Normalization == 'Min/Max':
    #     plt.title(f'Min/Max Normalization Histogram of Feature {feature}')
    # elif Normalization == 'Z-Score':
    #     plt.title(f'Z-Score Normalization Histogram of Feature {feature}')
    plt.xlabel(f'Activity Detected')
    plt.ylabel('Frequency')
    plt.show()

data = pd.read_csv('./DataA.csv')
numsamples = 19000

# detecting and filling missing data using median
for features in data.columns:
    if data[features].isnull().any():
        if data[features].isnull().sum() > 0.50*numsamples:
            data = data.drop(columns=[features])
        else:
            median = data[features].median()
            data[features] = data[features].fillna(median)

# detecting and replacing outliers with upper/lower boundary
for features in data.columns:
    IQR = data[features].quantile(0.75) - data[features].quantile(0.25)
    lowerbound = data[features].quantile(0.25) - 1.5*IQR
    upperbound = data[features].quantile(0.75) + 1.5*IQR

    outliers = data[(data[features] < lowerbound) | (data[features] > upperbound)]
    if not outliers.empty:
        numberbins = numbins(data, features, 'No')
        bins = pd.cut(data[features], bins=numberbins, labels=False, include_lowest=True)

        closer_to_upper = (outliers[features] > upperbound) & (bins == bins.max())
        closer_to_lower = (outliers[features] < lowerbound) & (bins == bins.min())

        data.loc[closer_to_upper, features] = upperbound
        data.loc[closer_to_lower, features] = lowerbound

# Min-max normalization and Z-score normalization
minmaxnorm_data = pd.DataFrame()
znorm_data = pd.DataFrame()
for features in data.columns:
    znorm_data[features] = (data[features] - data[features].mean())/data[features].std()
    if data[features].max() - data[features].min() == 0:
        minmaxnorm_data[features] = 0
    else:
        minmaxnorm_data[features] = (data[features] - data[features].min())/(data[features].max()-data[features].min())
    

selectedfeatures = [9, 24]

for features in selectedfeatures:
    numberbins = numbins(data, features, 'Yes')
    Histogram(data, features, numberbins, 'None')
    Histogram(minmaxnorm_data, features, numberbins, 'Min/Max')
    Histogram(znorm_data, features, numberbins, 'Z-Score')


# print(data.isnull().sum())
# print(data.describe())
# print(minmaxnorm_data.describe())
# print(znorm_data.describe())