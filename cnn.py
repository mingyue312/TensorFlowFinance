from yahoo_finance import Share
from pprint import pprint
import pandas as pd
from pandas_datareader import data
import tensorflow as tf
import numpy as np
import sys

# Define some macros
TRAINING_SET_RATIO = 0.8

#apple = Share('AAPL')
#past_price = apple.get_historical('2016-05-01', '2016-05-31')
training_test_data = data.DataReader('AAPL', 'yahoo', '2014-07-01')

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

# placeholders for feeding data
# feature data size: BATCH_SIZE * num_predictors
feature_data = tf.placeholder("float", [None, num_predictors])
actual_classes = tf.placeholder("float", [None, num_classes])

# define weight matrices and assign some random values to it
# for now, use a 1 hidden layer neural network
# Input batch size  : 200 , TODO: now is whatever the input size is
# hidden 1 units    : 50 , mapping size 50*200 + bias
# output units      : 1  , mapping size  1* 50 + bias
INPUT_BATCH_SIZE    = len(training_predictors_tf)
HIDDEN_1_SIZE       = 10
OUTPUT_SIZE         = num_classes

# input -> hidden layer
weights1 = tf.Variable(tf.truncated_normal([num_predictors, HIDDEN_1_SIZE], stddev=0.0001))
biases1 = tf.Variable(tf.ones([HIDDEN_1_SIZE]))

# hidden layer -> output
weights2 = tf.Variable(tf.truncated_normal([HIDDEN_1_SIZE, OUTPUT_SIZE], stddev=0.0001))
biases2 = tf.Variable(tf.ones([OUTPUT_SIZE]))

# hidden layer 1 calculation
hidden_layer_1 = tf.matmul(feature_data, weights1) + biases1
model = tf.nn.sigmoid(tf.matmul(hidden_layer_1, weights2) + biases2)
# hidden_layer_2 = tf.nn.relu(tf.matmul(hidden_layer_1, weights2) + biases2)
# model = tf.nn.softmax(tf.matmul(hidden_layer_2, weights3) + biases3)

cost = -tf.reduce_mean(actual_classes*tf.log(model) + (1-actual_classes)*tf.log(1-model))

train_op1 = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

# 0.01 learning rate
optimizer = tf.train.GradientDescentOptimizer(0.005)
train_op = optimizer.minimize(cost)

init = tf.initialize_all_variables()
sess.run(init)

# output: round the model
output = tf.round(model)

correct_prediction = tf.equal(output, actual_classes)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


## this section optimizes the matrix for 30000 times and, use all training data at a time. Which, is super
## computationally expensive
##
for i in range(1, 110001):
    sess.run(
        train_op,
        feed_dict={
            feature_data: training_predictors_tf.values,
            actual_classes: training_classes_tf.values.reshape(len(training_classes_tf.values), 1)
        }
    )

    if i%10000 == 0:
        print(i, sess.run(
            accuracy,
            feed_dict={
                feature_data: training_predictors_tf.values,
                actual_classes: training_classes_tf.values.reshape(len(training_classes_tf.values), 1)
            }
        )
        )
        # print(weights1.eval(session=sess))



