"""
========================================
Nearest Neighbors time-series regression
========================================

"""

# Author: Ogban-Asuquo Ugot <ogbanugot@gmail.com>

# #############################################################################
import numpy as np
from sklearn import neighbors
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import seaborn as sns

# Importing the training set
dataset = pd.read_csv('dataset.csv')

sysfreq = dataset['sysFreq']
# Narrower bandwidth
sns.kdeplot(sysfreq, shade=True, bw=.05, color="olive")
plt.show()

plt.subplot(211)
plt.plot(dataset['sysFreq'],dataset['load'], marker='o', alpha=0.4)
plt.title("A subplot with 2 lines")
plt.show()

dataset_train = dataset.loc[:3746, :]
dataset_test2 = dataset.loc[3747:, :]
dataset_test = dataset_test2.loc[dataset_test2['month'] != 12]

dataset_valid = dataset_test2.loc[dataset_test2['month'] == 12]
dataset_test_month = dataset_test2.loc[dataset_test2['month'] == 10]
dataset_test_month = dataset_test_month.iloc[:, 6:].values
month_sysfreq = dataset_valid.iloc[:, 6:].values 

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
dataset_train, dataset_test = train_test_split(dataset, test_size = 0.2, random_state = 0)


training_set = dataset_train.iloc[:, 6:].values

from matplotlib import pyplot
from pandas.tools.plotting import autocorrelation_plot
autocorrelation_plot(training_set)
pyplot.show()

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)


# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(30, 3747):
    X_train.append(training_set_scaled[i-30:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))

y_test = dataset_test_month 
#dataset_test.iloc[:, 6:].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['sysFreq'], dataset_test['sysFreq']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 30:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(30, 352):
    X_test.append(inputs[i-30:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))

# #############################################################################
# Fit regression model
n_neighbors = 20
weights = "distance"
knn = neighbors.KNeighborsRegressor(n_neighbors, weights = weights, algorithm='ball_tree', n_jobs =-1)
knn.fit(X_train, y_train)

# Grid search 
# =============================================================================
from sklearn.model_selection import GridSearchCV
parameters = {'n_neighbors':[10,20,30],'algorithm':('ball_tree','kd_tree')}
knn = neighbors.KNeighborsRegressor(weights = 'distance')
reg = GridSearchCV(knn, parameters, scoring = 'neg_mean_squared_error', n_jobs = -1)
reg.fit(X_train, y_train)
best_score = reg.best_score_
best_param = reg.best_params_
knn = reg.best_estimator_ #Can use in Cross_val_score & predict



KNN_prediction = knn.predict(X_test)
KNN_prediction = sc.inverse_transform(KNN_prediction.reshape(1, -1))
KNN_prediction = KNN_prediction.reshape(-1)
y_test = y_test.reshape(-1)


KNN_mse = mean_squared_error(y_test,KNN_prediction)
KNN_var = np.var(KNN_prediction)


c = 'blue'
plt.xticks(np.array([0]), "mse")
plt.boxplot(RNN_mses,notch=True, patch_artist=True,
            boxprops=dict(facecolor=c, color=c),
            capprops=dict(color=c),
            whiskerprops=dict(color=c),
            flierprops=dict(color=c, markeredgecolor=c),
            medianprops=dict(color='red', linewidth=5),)
plt.show()

#Variants of scoring
msle = mean_squared_log_error(y_test, KNN_prediction)
KNN_mse = mean_squared_error(y_test,KNN_prediction)
mae = mean_absolute_error(y_test,KNN_prediction)
r2 = r2_score(y_test,KNN_prediction)
from math import sqrt
rmse = sqrt(mse) #root mean --

plt.plot(y_test, color = 'red', label = 'Truth')
plt.plot(KNN_prediction, color = 'blue', label = 'KNN')
plt.title('Forecasting performance for KNN')
plt.xlabel('Hours')
plt.ylabel('Frequency')
plt.legend()
plt.show()

from sklearn.metrics import accuracy_score, normalized_mutual_info_score, classification_report, cohen_kappa_score
bias  = 63.2
y_test_operation, y_test_sysfreq = Give_operation(y_test, bias)
y_test_kokappa = cohen_kappa_score(y_test_operation,y_test_operation)
y_test_clrp = classification_report(y_test_operation,y_test_operation)
y_test_nmi = normalized_mutual_info_score(y_test_operation,y_test_operation)
from AGC_model import  Give_operation


KNN_prediction_operation, KNN_sysfreq = Give_operation(y_test, bias)
knn_acc = accuracy_score(y_test_operation,KNN_prediction_operation)
knn_nmi = normalized_mutual_info_score(y_test_operation,KNN_prediction_operation)
knn_clrp = classification_report(y_test_operation,KNN_prediction_operation)
knn_kokappa = cohen_kappa_score(y_test_operation,KNN_prediction_operation)