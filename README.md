# tf.estimator_tutorial

In this tutorial we will see how you can use Tensorflow’s high level API on a real life dataset.  We will build our input functions by using tf.data.Dataset, loading first our csv into a pandas dataframe. We will then create a custom model consisting of a 4 hidden layers DNN with random initialization, L2 regularization and using ReLU for activation.
In most of the tutorials, it is assumed that your dataset have perfect quality or contains a reasonable amount of columns. However a lot of real life examples have to deal with missing values, non-numerical columns, high amount of features, …. We will try here to address these issues in an automated way.
We assume here that your dataset is in csv format, but any other format would fit as long as you can load it in a pandas dataframe.

## 1)	Preparations:

First of all let’s import our libraries and set verbosity higher so we have more details to look at during training:

```from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import pandas as pd
import numpy as np
import sys
import os

tf.logging.set_verbosity(tf.logging.INFO)
```

For simplicity’s sake I will not use an arg parser for my hyper parameters but rather define them explicitly. Feel free to tune them, create for loops to try different values, etc…
Being on a windows machine I use the “r” prefix for my file path to handle spaces in directories names and anti-slashes. If you are on Linux simply remove the “r” and paste your filepath.
Here I have 2 sets, one for training and one for test. But you can use the scikit train_test_split if you have only one file containing your train and test examples.

```# setting hyperparameters
BATCH_SIZE = 100
repeat_count = 1500    # epochs
PATH = r"C:\tmp\"
PATH_DATASET=r"C:\tmp\dataset"
FILE_TRAIN = PATH_DATASET + os.sep + "tf_train.csv"
FILE_TEST = PATH_DATASET + os.sep + "tf_test.csv"    
CSV_COLUMN_NAMES = []
numerical_feature_names = []
categorical_feature_names = []
```

## 2) Loading your dataset:

Next we load the csv files in a pandas data frame.  I will cover in another tutorial how to handle csv files or dataset bigger than your available RAM (or GPU memory).

To build your dataset, you need to know the data type of your columns, their names and separate your labels and features. The code below is doing that in an automated way. This way there is no need to declare manually your data structure when building the Tensorflow dataset.

### a.	Get the column names and load the csv

```CSV_COLUMN_NAMES = pd.read_csv(FILE_TRAIN, nrows=1).columns.tolist()
train = pd.read_csv(FILE_TRAIN, names=CSV_COLUMN_NAMES, header=0)
train_x, train_y = train, train.pop('labels')

test = pd.read_csv(FILE_TEST, names=CSV_COLUMN_NAMES, header=0)
test_x, test_y = test, test.pop('labels')
```

### b.	Get the columns type and store them in an array:

```for column in train.columns:
    print (train[column].dtype)
    if(train[column].dtype == np.float64 or train[column].dtype == np.int64):
        numerical_feature_names.append(column)
    else:
        categorical_feature_names.append(column)
```

## 3) Using Tensorflow feature_columns:

To handle non numerical value and perform efficient computation, Tensorflow has introduced feature_columns. 
Please see official tutorial here.

We will use tf.feature_column.numeric_column for our columns containing floats or integers. For the categorical column, we will use the categorical_column_with_hash_bucket for columns having a high amount of unique values, and categorical_column_with_vocabulary_list if we have a reasonable amount of unique values. It is basically creating a one hot vector out of the original column, where each new column is representing the presence or absence of one of the unique values of the original column (0 or 1).

Depending on the model you choose, if you pick a DNN, your estimator will expect that all columns fed to the model are dense columns. To address this you need to embed your categorical column in an indicator or embedding column. If your inputs are sparse for this column, choose an embedding column, otherwise choose an indicator column.

### a.	Numerical columns:

`feature_columns = [tf.feature_column.numeric_column(k) for k in numerical_feature_names]`

### b.	Categorical columns :

```for k in categorical_feature_names:
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
```

## 4) Input function:

Now that we have prepared our data structure for processing, let’s create an input function. It is creating a TF Dataset to be fed to the estimator. We will detail what is an estimator later.
Since we need to train, eval and predict, we will create 2 input functions handling these cases. We also shuffle and repeat a repeat_count amount of iteration for training (epochs), and use mini batches for both modes. 

```# input_fn for training, convertion of dataframe to dataset
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
```

## 5) Model function:

An estimator needs to get the data from the input function. Then it needs a model function in order to know how to process your data. 
There are 2 kind of model functions: pre-made models or custom ones. 

A list of pre-made models can be found here. They are easy to implement and already include all the necessary logging to display later in Tensorboard.
Custom models offer more flexibility, but everything needs to be declared explicitly. Layers, units, initialization, regularization, activations…
Here we want to be able to tune our model therefore we will use a custom model.

### a.	Declaration and modes:

First we want to handle the 3 modes that can be used by our estimator: train, evaluate, predict. Let’s print which mode we are in:

```def my_model_fn(features, labels, mode):

    if mode == tf.estimator.ModeKeys.PREDICT:
        tf.logging.info("my_model_fn: PREDICT, {}".format(mode))
    elif mode == tf.estimator.ModeKeys.EVAL:
        tf.logging.info("my_model_fn: EVAL, {}".format(mode))
    elif mode == tf.estimator.ModeKeys.TRAIN:
        tf.logging.info("my_model_fn: TRAIN, {}".format(mode))
```

### b.	Network architecture:

Secondly we want to declare how our NN will look like.

We first need an initializer:

    `initializer = tf.contrib.layers.xavier_initializer()`

Then a L2 regularization for our weights:

    `regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)`

Each layer can then apply these 2 to either weights (kernel) and/or biases.

Let’s write the input layer:

    `input_layer = tf.feature_column.input_layer(features, feature_columns)`

And now our first fully connected hidden layer. It takes the input_layer as input, has 100 hidden units with initialization of the weights, L2 regularization and ReLU for activation:

```h1 = tf.layers.Dense(100, activation=tf.nn.relu, 
                            kernel_regularizer=regularizer,
                            kernel_initializer=initializer
                            )(input_layer)
```

Let’s have a second hidden layer with 80 units:

```h2 = tf.layers.Dense(80, activation=tf.nn.relu, 
                            kernel_regularizer=regularizer,
                            kernel_initializer= initializer
                            )(h1)
```

And finally our last layer which is another fully connected layer (Dense) outputs the logits. Since we want to classify 2 classes, our output layer has 2 units.:

`logits = tf.layers.Dense(2)(h2)`

These logits are basically the output of the linearity before activation. We can then calculate our loss on them or just output the prediction.

Let’s now write how each mode should be handled. 

The easiest is the prediction. We look at our logits and take the one with highest probability:

```    # compute predictions
    predicted_classes =tf.argmax(input=logits, axis=1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
```

We return an EstimatorSpec which defines our prediction model in the estimator.

For our training, we need to have a loss function. We use in this model the sparse_softmax_cross_entropy function for our output loss calculation:

`loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)`

For both train and eval, we want to record a few metrics so we can later review them in Tensorboard. 
Let’s have a look at accuracy, precision and recall: we first create the corresponding operation from tf.metrics and put them in a dictionary:

```    accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_classes, name='acc_op')
    precision = tf.metrics.auc(labels, predictions=predicted_classes, name='precision_op')
    recall = tf.metrics.recall(labels, predictions=predicted_classes, name='recall_op')
    auc = tf.metrics.auc(labels, predictions=predicted_classes, name='auc_op')
    metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'auc': auc}
```

Then we add them to tf.summary in order to log them for Tensorboard:

    ```tf.summary.scalar('my_accuracy', accuracy[1])
    tf.summary.scalar('my_precision', precision[1])
    tf.summary.scalar('my_recall', recall[1])
    tf.summary.scalar('my_auc', auc[1])
```


We have everything we need for evaluation so let’s handle this case now:

```if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)
```

We return again an EstimatorSpec as for the prediction mode.

Now if we want to handle the training case we still need a few more elements in our computation graph. We need to create an optimizer and the training operation, where the goal of the optimizer is to minimize the loss. In this model we use an Adam Optimizer:

```    # training op
    assert mode == tf.estimator.ModeKeys.TRAIN, "TRAIN is only ModeKey left"
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.05)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
```

I also like to see how my parameters are changing through iterations in Tensorboard. 
We can add a summary for every weight, bias and gradient the following way:

```    # add gradients, weights and biases to tensorboard 
    grads = optimizer.compute_gradients(loss)
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)
    for var in tf.trainable_variables():
        tf.summary.histogram(var.name, var)
```

Finally we return the EstimatorSpec for our training mode: 

```    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)```

## 6) Classifier, Training, Evaluation, Prediction:

Now that we have all the bricks to build our estimator, let’s put the final touch and use it.

### a.	Creating the estimator:

You can think of an estimator as a playground where we throw our examples, model, and/or requests, and that run the desired operations (train/eval/predict).
We need to pass it the model function name and a path to which it should store the training checkpoints and the Tensorboard logs.

```classifier = tf.estimator.Estimator(
    model_fn=my_model_fn,
    model_dir=PATH)  
```

If you wish to make changes to your model, make sure you change the path or remove the checkpoints from the folder, otherwise it will crash. If you don’t change your model but run your training multiple times, it will not start from scratch if you have not deleted the checkpoints. Instead it will start from where it stopped at the previous training. Loading the last learnt weights and biases.

### b.	Training:

Here we feed our estimator with the train input function and the features and labels created at the beginning. We set the batch size and the amount of iterations we want. If you use an arg parser you should input your argument values here.

`classifier.train(input_fn=lambda: train_input_fn(train_x, train_y, BATCH_SIZE, repeat_count))`

### c.	Evaluation:

Once we have trained, let’s look how good we do on the test set by evaluating it. This time we pass the evaluation input function with test features and labels, and the batch size. We do not pass the `repeat_count` since we will iterate only once over our test set:

```evaluate_result = classifier.evaluate(input_fn=lambda: eval_input_fn(test_x, test_y, BATCH_SIZE))```

Let’s print our results:

```print ("Evaluation results")

for key in evaluate_result:
   print("   {}, was: {}".format(key, evaluate_result[key]))
```

### d.	Prediction:

Finally, let’s use our model on unlabelled data and see how our predictions look like.

You could load prediction_input from a csv or craft a dictionary for the example purpose:

```prediction_input = {
    'c1': ["Sony", "Samsung", "Samsung"],
    'n2': [10, 55, 2],
    'n3': [1, 0, 0]
}
```

You will reuse the eval/predict input function but with None instead of labels and 1 as batch size or bigger if you want to predict in batches.

```predict_results = classifier.predict(
        input_fn=lambda:eval_input_fn(prediction_input, None, 1))
```

And finally let’s print the result of the prediction and the related probability:

```CLASSES = ['1', '0']
# Print results
tf.logging.info("Predictions:")
for pred_dict, expec in zip(predict_results, expected):
        template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(CLASSES[class_id],
                              100 * probability, expec))
```


## 7) Tensorboard:

Your model has been saved in the PATH directory. You will find there your model checkpoints and the log of the summaries for tensorboard.
You then start tensorboard the following way:
```tensorboard --logdir=c:/tmp/ --port=6006```

You might want to tweak your model and compare the different runs to find which models performs better on your data/problem. In order to do so you need to pass a different folder to the estimator for each run when you modify the structure of your model or your loss function or your optimizer...
This what we pass in this example:

```PATH = r"C:\tmp\"
…
classifier = tf.estimator.Estimator(
    model_fn=my_model_fn,
    model_dir=PATH)  
```

Once you have made your loop to try different models and logged your summaries in different folders, you can compare them in tensorboard by starting it the following way:

`tensorboard --logdir=run1:”C:/tmp/1”,run2:”C:/tmp/2” --port=6006`

