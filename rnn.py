# Recurrent Neural Network

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the training set
dataset = pd.read_csv('dataset2.csv')

dataset_train = dataset.loc[dataset['year'] != 3]
dataset_test = dataset.loc[dataset['year'] == 3]

#dataset_train = dataset.loc[dataset['year'] != 3]
#dataset_test = dataset.loc[dataset['year'] == 3]


dataset_train = dataset.loc[:3746, :]
dataset_test = dataset.loc[3747:, :]

month = dataset_test.loc[dataset_test['month'] == 12]
week_freq = month.loc[month['week'] == 4]
day_freq = week_freq.loc[month['day'] == 7]
month_sysfreq = month.iloc[:, 6:].values 
day = month.loc[month['day']==1]
day = day.loc[day['week']==1]

sns.kdeplot(month['sysFreq'], shade=True, bw=.05, color="olive")
plt.show()

#plt.subplot(211)
sns.set_style("whitegrid", {'axes.grid' : False})
sns.regplot( day['load'], day['month'], fit_reg=False, scatter_kws={"color":"darkred","alpha":0.3,"s":200} )
plt.title("A subplot with 2 lines")
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
for i in range(120, 504):
    X_train.append(training_set_scaled[i-120:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras_layer_normalization import LayerNormalization

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LayerNormalization(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1))))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LayerNormalization(LSTM(units = 50, return_sequences = True)))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LayerNormalization(LSTM(units = 50, return_sequences = True)))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LayerNormalization(LSTM(units = 50)))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
train_history = regressor.fit(X_train, y_train, epochs = 100, batch_size = 64)

# Grid search 
# =============================================================================

# Getting the real stock price of 2017

y_test = day.iloc[:, 6:].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['sysFreq'], dataset_test['sysFreq']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 120:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(120, 504):
    X_test.append(inputs[i-120:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

RNN_prediction = regressor.predict(X_test)
RNN_prediction = sc.inverse_transform(RNN_prediction)

y_test = y_test.reshape(-1)
RNN_prediction = RNN_prediction.reshape(-1)

from sklearn.metrics import mean_squared_error
RNN_mse = mean_squared_error(y_test,RNN_pred)

from math import sqrt
rmse = sqrt(mse) #root mean --

# Visualising results
plt.plot(y_test, color = 'red', label = 'Truth')
plt.plot(RNN_pred, color = 'blue', label = 'RNN')
plt.title('Forecasting performance for RNN in one day')
plt.xlabel('Hours')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Visualising the results
#plt.plot(prior_time, color = 'red', label = 'Truth')
#plt.plot(RNN_prediction, color = 'blue', label = 'RNN')
plt.xticks(np.array([0,1,2,3,4]), prior_time)
plt.plot(RNN_vars, color = 'blue', label = 'RNN')
plt.plot(ANN_vars, color = 'brown', label = 'ANN')
plt.plot(KNN_vars, color = 'red', label = 'KNN')
plt.plot(SVM_vars, color = 'green', label = 'SVM')
plt.title('Variance with increase in prior time step')
plt.xlabel('Prior Time Step')
plt.ylabel('Variance')
plt.legend()
plt.show()


from keras.utils import plot_model
plot_model(regressor.model, to_file='RNN_model.png')

# serialize model to JSON
model_json = regressor.model.to_json()
with open("RNN_model_120.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
regressor.model.save_weights("RNN_model_60.h5")
print("Saved model to disk")

# later...

from keras.models import model_from_json
# load json and create model
json_file = open('RNN_model_60.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
regressor = model_from_json(loaded_model_json)
# load weights into new model
regressor.load_weights("RNN_model.h5")
print("Loaded model from disk")

# evaluate loaded model 
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['mse'])

from AGC_model import Give_operation
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, classification_report, cohen_kappa_score
bias  = 63.2
y_test_operation, y_test_sysfreq = Give_operation(y_test, bias)

RNN_prediction_operation, RNN_sysfreq = Give_operation(RNN_prediction, bias)
rnn_acc = accuracy_score(y_test_operation,RNN_prediction_operation)
rnn_nmi = normalized_mutual_info_score(y_test_operation,RNN_prediction_operation)
rnn_clrp = classification_report(y_test_operation,RNN_prediction_operation)
rnn_kokappa = cohen_kappa_score(y_test_operation,RNN_prediction_operation)
