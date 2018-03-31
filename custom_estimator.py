from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import pandas as pd
import numpy as np
import sys
import os

tf.logging.set_verbosity(tf.logging.INFO)

# setting hyperparameters
BATCH_SIZE = 100
repeat_count = 1500    # epochs
PATH = r"C:\tmp\"
PATH_DATASET=r"C:\tmp\dataset"
FILE_TRAIN = PATH_DATASET + os.sep + "tf_train.csv"
FILE_TEST = PATH_DATASET + os.sep + "tf_test.csv"    
CSV_COLUMN_NAMES = []
numerical_feature_names = []
categorical_feature_names = []

# Get the column names and load the csv
CSV_COLUMN_NAMES = pd.read_csv(FILE_TRAIN, nrows=1).columns.tolist()
train = pd.read_csv(FILE_TRAIN, names=CSV_COLUMN_NAMES, header=0)
train_x, train_y = train, train.pop('labels')

test = pd.read_csv(FILE_TEST, names=CSV_COLUMN_NAMES, header=0)
test_x, test_y = test, test.pop('labels')

# Get the columns type and store them in an array
for column in train.columns:
    print (train[column].dtype)
    if(train[column].dtype == np.float64 or train[column].dtype == np.int64):
        numerical_feature_names.append(column)
    else:
        categorical_feature_names.append(column)

# building our feature columns
feature_columns = [tf.feature_column.numeric_column(k) for k in numerical_feature_names]
for k in categorical_feature_names:
    current_bucket = train[k].nunique()
    if current_bucket>10:
        feature_columns.append(
            tf.feature_column.indicator_column(
                tf.feature_column.categorical_column_with_vocabulary_list(key=k, vocabulary_list=train[k].unique())
            )
        )
    else:
        feature_columns.append(
            tf.feature_column.indicator_column(
                tf.feature_column.categorical_column_with_hash_bucket(key=k, hash_bucket_size=current_bucket)
            )
        )

# input_fn for training, convertion of dataframe to dataset
def train_input_fn(features, labels, batch_size, repeat_count):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(256).repeat(repeat_count).batch(batch_size)
    return dataset

# input_fn for evaluation and predicitions (labels can be null)
def eval_input_fn(features, labels, batch_size):
    features=dict(features)
    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)

    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)
    return dataset

def my_model_fn(features, labels, mode, params):

    if mode == tf.estimator.ModeKeys.PREDICT:
        tf.logging.info("my_model_fn: PREDICT, {}".format(mode))
    elif mode == tf.estimator.ModeKeys.EVAL:
        tf.logging.info("my_model_fn: EVAL, {}".format(mode))
    elif mode == tf.estimator.ModeKeys.TRAIN:
        tf.logging.info("my_model_fn: TRAIN, {}".format(mode))

    initializer = tf.contrib.layers.xavier_initializer()

    regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)

    input_layer = tf.feature_column.input_layer(features, feature_columns)
    h1 = tf.layers.Dense(100, activation=tf.nn.relu, 
                            kernel_regularizer=regularizer,
                            kernel_initializer=initializer
                            )(input_layer)
    h2 = tf.layers.Dense(80, activation=tf.nn.relu, 
                            kernel_regularizer=regularizer,
                            kernel_initializer= initializer
                            )(h1)
    logits = tf.layers.Dense(2)(h2)
    
    # compute predictions
    predicted_classes =tf.argmax(input=logits, axis=1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_classes, name='acc_op')
    precision = tf.metrics.auc(labels, predictions=predicted_classes, name='precision_op')
    recall = tf.metrics.recall(labels, predictions=predicted_classes, name='recall_op')
    auc = tf.metrics.auc(labels, predictions=predicted_classes, name='auc_op')
    metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'auc': auc}

    tf.summary.scalar('my_accuracy', accuracy[1])
    tf.summary.scalar('my_precision', precision[1])
    tf.summary.scalar('my_recall', recall[1])
    tf.summary.scalar('my_auc', auc[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # training op
    assert mode == tf.estimator.ModeKeys.TRAIN, "TRAIN is only ModeKey left"
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.05)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    # add gradients, weights and biases to tensorboard 
    grads = optimizer.compute_gradients(loss)
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)
    for var in tf.trainable_variables():
        tf.summary.histogram(var.name, var)

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)



classifier = tf.estimator.Estimator(
model_fn=my_model_fn,
model_dir=PATH)  

classifier.train(input_fn=lambda: train_input_fn(train_x, train_y, BATCH_SIZE, repeat_count))

evaluate_result = classifier.evaluate(input_fn=lambda: eval_input_fn(test_x, test_y, BATCH_SIZE))

print ("Evaluation results")
for key in evaluate_result:
   print("   {}, was: {}".format(key, evaluate_result[key]))

