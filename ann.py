# Importing the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from keras.models import model_from_json
import seaborn as sns

# Importing the training set
dataset = pd.read_csv('dataset.csv')

sysfreq = dataset['sysFreq']
# Narrower bandwidth
sns.kdeplot(sysfreq, shade=True, bw=.05, color="olive")
plt.show()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
dataset_train, dataset_test = train_test_split(dataset, test_size = 0.2, random_state = 0)


training_set = dataset_train.iloc[:, 6:].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)


# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 3883):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))

y_test = dataset_test.iloc[:, 6:].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['sysFreq'], dataset_test['sysFreq']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 1031):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))


#Evaluating improving and tuning the ANN 
#Evaluating the ANN
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import Dropout
from keras.models  import Sequential
from keras.layers import Dense
def build_model():
   model = Sequential()
   model.add(Dense(60, activation = 'relu', input_dim = 30))
   model.add(Dropout(rate = 0.2))
   model.add(Dense(units = 1, kernel_initializer = 'uniform', ))
   model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['mse','mae'])
   return model

# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=build_model, epochs=5, batch_size=10)
estimator.fit(X_train, y_train, batch_size = 10, epochs = 100)

ANN_prediction = estimator.predict(X_test)
ANN_prediction = sc.inverse_transform(ANN_prediction.reshape(1, -1))
ANN_prediction = ANN_prediction.reshape(-1)
y_test = y_test.reshape(-1)

ANN_mse = mean_squared_error(y_test,ANN_prediction)
ANN_var = np.var(ANN_prediction)

#Variants of scoring
msle = mean_squared_log_error(y_test, ANN_prediction)
ANN_mse = mean_squared_error(y_test,ANN_prediction)
mae = mean_absolute_error(y_test,ANN_prediction)
r2 = r2_score(y_test,ANN_prediction)
from math import sqrt
rmse = sqrt(mse) #root mean --

#Model visualization

plt.plot(y_test, color = 'red', label = 'Truth')
plt.plot(ANN_prediction, color = 'blue', label = 'ANN')
plt.title('Forecasting performance for ANN')
plt.xlabel('Hours')
plt.ylabel('Frequency')
plt.legend()
plt.show()

from keras.utils import plot_model
plot_model(estimator.model, to_file='ANN_model.png')

# serialize model to JSON
model_json = estimator.model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
estimator.model.save_weights("model.h5")
print("Saved model to disk")

# later...

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model 
loaded_model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['mse','mae'])

from AGC_model import AGC_control, Give_operation
from sklearn.metrics import accuracy_score, normalized_mutual_info_score
bias  = 63.2
y_test_operation = Give_operation(y_test, bias)
ANN_prediction_operation = Give_operation(ANN_prediction, bias)
ann_acc = accuracy_score(y_test_operation,ANN_prediction_operation)
ann_nmi = normalized_mutual_info_score(y_test_operation,ANN_prediction_operation)
