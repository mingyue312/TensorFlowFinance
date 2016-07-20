from yahoo_finance import Share
from pprint import pprint
import pandas as pd
from pandas_datareader import data
import tensorflow as tf
import numpy as np

# Define some macros
TRAINING_SET_RATIO = 0.8

#apple = Share('AAPL')
#past_price = apple.get_historical('2016-05-01', '2016-05-31')
training_test_data = data.DataReader('AAPL', 'yahoo', '2016-07-01')

training_test_data['Class'] = 0

# reset the original indexing (by date) to 0..n
training_test_data=training_test_data.reset_index(drop=True)

# setting up some useful variables
data_length = len(training_test_data['Open'])

# set the classification data
for i in range(0, len(training_test_data['Open'])):
    if training_test_data['Close'][i] > training_test_data['Open'][i]:
        training_test_data.set_value(i, 'Class', 1)
    else:
        training_test_data.set_value(i, 'Class', 0)

for i in range(0, len(training_test_data['Open'])):
    training_test_data.set_value(i, 'Volume', training_test_data['Volume'][i]/1000000.0)

# Remove the close values
training_test_data = training_test_data.drop('Close', 1)
training_test_data = training_test_data.drop('Adj Close', 1)

# Split the data into x_i and y, in other words, features and results
predictors_tf = training_test_data[training_test_data.columns[:-1]]
classes_tf = training_test_data[training_test_data.columns[-1]]

# divide the training set into training and testing
training_set_size = int(len(training_test_data)*TRAINING_SET_RATIO)
test_set_size = len(training_test_data) - training_set_size

training_predictors_tf = predictors_tf[:training_set_size]
training_classes_tf = classes_tf[:training_set_size]
test_predictors_tf = predictors_tf[training_set_size:]
test_classes_tf = classes_tf[training_set_size:]

# Make tensorFlow session
sess = tf.Session()

num_predictors = len(training_predictors_tf.columns)
num_classes = 1

feature_data = tf.placeholder("float", [None, num_predictors])
actual_classes = tf.placeholder("float", [None, 1])

weights1 = tf.Variable(tf.truncated_normal([4, 20], stddev=0.0001))
biases1 = tf.Variable(tf.ones([20]))
weights2 = tf.Variable(tf.truncated_normal([20, 20], stddev=0.0001))
biases2 = tf.Variable(tf.ones([20]))
weights3 = tf.Variable(tf.truncated_normal([20, 1], stddev=0.0001))
biases3 = tf.Variable(tf.ones([1]))

hidden_layer_1 = tf.nn.relu(tf.matmul(feature_data, weights1) + biases1)
hidden_layer_2 = tf.nn.relu(tf.matmul(hidden_layer_1, weights2) + biases2)
model = tf.nn.softmax(tf.matmul(hidden_layer_2, weights3) + biases3)

cost = -tf.reduce_sum(actual_classes*tf.log(model))

train_op1 = tf.train.AdamOptimizer(learning_rate=0.00010).minimize(cost)

init = tf.initialize_all_variables()
sess.run(init)

correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(actual_classes, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

for i in range(1, 30001):
    sess.run(
        train_op1,
        feed_dict={
            feature_data: training_predictors_tf.values,
            actual_classes: training_classes_tf.values.reshape(len(training_classes_tf.values), 1)
        }
    )
    if i%500 == 0:
        print(i, sess.run(
            accuracy,
            feed_dict={
                feature_data: training_predictors_tf.values,
                actual_classes: training_classes_tf.values.reshape(len(training_classes_tf.values), 1)
            }
        )
        )