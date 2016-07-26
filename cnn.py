from yahoo_finance import Share
from pprint import pprint
from pandas_datareader import data
import pandas as pd
import datetime
import tensorflow as tf
import numpy as np
import sys
import math



flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 30000, 'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('output_classes', 1, 'Number of possible classifications')
flags.DEFINE_integer('batch_size', 100, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                     'for unit testing.')

TRAINING_SET_RATIO  = 0.8
NORMALIZATION       = 1000000.0

def fetch_financial_data(company_code, start_date):
    """
    Fetches financial data from Yahoo Finance with the company code and date
    Args:
        company_code: Company NYSE/NASDAQ codes
        start_date: date object
    Returns:
        the Panda representation of all the financial data

    """
    date = str(start_date)
    result = data.DataReader(company_code, 'yahoo', date)
    return result


def process_training_data(training_data):
    """
    Adds classification and drops useless columns

    Args:
        training_data: Panda data returned by data fetcher

    Returns:
        The processed data, divided, one is features array, another is classification
    """
    data_size = len(training_data["Open"])

    # add classification column
    training_data['Class'] = 0

    # remove the date indexing
    training_data = training_data.reset_index(drop=True)

    # Set the classification
    for i in range(data_size):
        # if close is higher
        if training_data['Close'][i] > training_data['Open'][i]:
            training_data.set_value(i, 'Class', 1)
        else:
            training_data.set_value(i, 'Class', 0)

        # Normalize the volume
        training_data.set_value(i, 'Volume', training_data['Volume'][i]/NORMALIZATION)

    # remove the close
    training_data = training_data.drop('Close', 1)
    training_data = training_data.drop('Adj Close', 1)

    return [training_data[training_data.columns[:-1]], training_data[training_data.columns[-1]]]


def divide_training_set(training_set):
    """
    Divides the training set into training and cross validation, according to TRAINING_SET_RATIO
    Args:
        training_set: The training set, can be either feature set or classification set

    Returns:
        Training sets and validation sets: both are dataframe type

    """
    training_set_size = int(len(training_test_data) * TRAINING_SET_RATIO)

    return [training_set[:training_set_size], training_set[training_set_size:]]

############################################################################
## Training mathods
##


def inference(training_set, hidden1_sz, hidden2_sz):
    
    """ Builds the inference (neural network model)

    Args:
        training_set: placeholder for the training set
        hidden1_sz: hidden layer 1 size
        hidden2_sz: hidden layer 2 size

    Returns:
        logits: computed logits tensor
    """

    # hidden layer 1
    with tf.name_scope('hidden1'):
        weights = tf.Variable(
                tf.truncated_normal([FEATURE_SIZE, hidden1_sz], stddev=1.0/math.sqrt(FEATURE_SIZE)),
                name='weights')
        biases = tf.Variable(tf.ones([hidden1_sz]), name='biases')
        hidden1 = tf.nn.sigmoid(tf.matmul(training_set, weights) + biases)

    # hidden layer 2
    with tf.name_scope('hidden2'):
        weights = tf.Variable(
                tf.truncated_normal([hidden1_sz, hidden2_sz], stddev=1.0/math.sqrt(hidden1_sz)),
                name='weights')
        biases = tf.Variable(tf.ones([hidden2_sz]), name='biases')
        hidden2 = tf.nn.sigmoid(tf.matmul(hidden1, weights) + biases)

    # output layer
    with tf.name_scope('output'):
        weights = tf.Variable(
                tf.truncated_normal([hidden2_sz, CLASS_SIZE], stddev=1.0/math.sqrt(hidden2_sz)),
                name='weights')
        biases = tf.Variable(tf.ones([CLASS_SIZE]), name='biases')
        logits = tf.nn.sigmoid(tf.matmul(hidden2, weights) + biases)

    return logits

   
def cost(logits, classes):
    """ Calculates cost with the predicted logits and classes

    Args:
        logits:     logits tensor: float [INPUT_SIZE, CLASS_SZIE]
        classes:    logits tensor: int32 [INPUT_SIZE]

    Returns:
        cost: cost tensor: float
    """

    classes = tf.to_float(classes)
    
    # use the sigmoid version for now

#    logits = tf.Print(logits, [logits,classes], 'logits/actual')
    xentropy = classes*tf.log(tf.clip_by_value(logits, 1e-10, 1)) + \
               (1-classes)*tf.log(tf.clip_by_value(1-logits, 1e-10, 1))
#    xentropy = tf.Print(xentropy, [xentropy], 'xentropy', summarize=10)

    cost = -tf.reduce_mean(xentropy, name='xentropy_mean')
    return cost


def training(cost, learning_rate):
    
    # add scaler summary TODO
    """ Set up training operation
        - generate a summary to track cost in tensorboard
        - create gradient descent optimizer for all trainable variables

        The training op returned has to be called in sess.run()

    Args:
        cost: cost tensor from cost()
        learning_rate: gradient descent learning rate

    Returns:
        train_op: training op
    """
    tf.scalar_summary('Mean cost', cost)

    # create gradient descent optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)

    # create global step variable to track global step: TODO
    global_step = tf.Variable(0, name='global_step', trainable=False)

    train_op = optimizer.minimize(cost, global_step=global_step)
    return train_op


def evaluation(logits, classes):
    """ Evaluate the quality of the logits at predicting the label.

    Note that, we are using sigmoid here, so we'll have to round the result to get correct prediction
    Args:
        logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        labels: Labels tensor, int32 - [batch_size], with values in the
            range [0, NUM_CLASSES).

    Returns:
        A scalar int32 tensor with the number of examples (out of batch_size)
        that were predicted correctly.
    """

    correct_prediction = tf.equal(tf.round(logits), classes)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    tf.scalar_summary('Accuracy', accuracy)
    return accuracy


def do_eval(sess, eval_op, summary_op, features_pl, classes_pl, features_data, classes_data):
    """ Runs evaluation against other data (validation/test)

    Args:
        sess: session to run it
        eval_op: evaluation comparison tensor
        features_pl: features input placeholder
        classes_pl: classes input placeholder
        features_data: features input
        classes_data: classes input
    """

    feed_dict = {
        features_pl: features_data.values,
        classes_pl: classes_data.values.reshape(len(classes_data.values), 1)
    }

    precision = sess.run(eval_op, feed_dict=feed_dict) * 100

    sess.run(summary_op, feed_dict=feed_dict)

    tf.scalar_summary('Validation', precision)

    return precision


# training_test_data = data.DataReader('AAPL', 'yahoo', '2014-07-01')
training_test_data = fetch_financial_data('AAPL', datetime.date(2014, 7, 1))

# Split the data into x_i and y, in other words, features and results
features_tf, classes_tf = process_training_data(training_test_data)

# divide the training set into training and testing
tf_training_features, tf_validation_features = divide_training_set(features_tf)
tf_training_classes, tf_validation_classes = divide_training_set(classes_tf)

# define weight matrices and assign some random values to it
# for now, use a 1 hidden layer neural network
# Input batch size  : 200 , TODO: now is whatever the input size is
# hidden 1 units    : 50 , mapping size 50*200 + bias
# output units      : 1  , mapping size  1* 50 + bias
INPUT_BATCH_SIZE    = len(tf_training_features)
FEATURE_SIZE        = len(tf_training_features.columns)
HIDDEN_1_SIZE       = 30
HIDDEN_2_SIZE       = 10
CLASS_SIZE          = 1

with tf.Graph().as_default():
    # placeholders for feeding data
    # feature data size: BATCH_SIZE * num_predictors
    feature_data_pl = tf.placeholder("float", [None, FEATURE_SIZE])
    actual_classes_pl = tf.placeholder("float", [None, CLASS_SIZE])


    # make tf session
    sess = tf.Session()

    #### Training step

    # add the model to the graph
    logits = inference(feature_data_pl, FLAGS.hidden1, FLAGS.hidden2)

    # cost calculation -> graph
    cost = cost(logits, actual_classes_pl)

    # training op -> graph
    train_op = training(cost, FLAGS.learning_rate)

    # add op to evaluate result
    eval_op = evaluation(logits, actual_classes_pl)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # run init
    init = tf.initialize_all_variables()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

    sess.run(init)

    for i in range(FLAGS.max_steps):
        # setup the feeding
        feed_dict = {
            feature_data_pl: tf_training_features.values,
            actual_classes_pl: tf_training_classes.values.reshape(len(tf_training_classes.values), 1)
        }
        _, cost_value = \
            sess.run(
            [train_op, cost],
            feed_dict=feed_dict
        )

        if i%500 == 0:
            # do_eval(sess, eval_op, feature_data_pl, actual_classes_pl, tf_validation_features, tf_validation_classes)
            accuracy = sess.run(eval_op, feed_dict=feed_dict) * 100

            validation_accuracy = do_eval(sess, eval_op, summary_op, feature_data_pl, actual_classes_pl, tf_validation_features, \
                                          tf_validation_classes)

            print('Step: %d, cost = [%.4f], accuracy = [%.2f%%], validation [%.2f%%]' % \
                  (i, cost_value, accuracy, validation_accuracy))

            summary_str = sess.run(summary_op, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, i)
            summary_writer.flush()



