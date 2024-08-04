%tensorflow_version 1.x

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow import train,Session
import matplotlib.pyplot as plt

import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from google.colab import drive

num_epochs = 1

drive.mount('/content/drive')
Cifar_10_dir_path = '/content/drive/My Drive/Datasets/cifar-10-python/cifar-10-batches-py'
Datasets_folder = '/content/drive/My Drive/Datasets'

savedir = '/content/drive/My Drive/Quantized_model'
print(savedir)

######Download, dataset processing######
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

cifar10_1 = unpickle(Cifar_10_dir_path + '/data_batch_1')

cifar10_dataset_folder_path = Cifar_10_dir_path

list(cifar10_1.keys())


len(cifar10_1[b'data']) ,len(cifar10_1[b'labels'])


cifar10_1[b'batch_label']

cifar10_Label_names = unpickle(Cifar_10_dir_path + '/batches.meta')
cifar10_Label_names[b'label_names']


label_names = []
for x in cifar10_Label_names[b'label_names']:
    label = x.decode()
    label_names.append(label)

label_names     

def load_cfar10_batch(cifar10_dataset_folder_path, batch_id):
    with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:
        
        batch = pickle.load(file, encoding='latin1')

    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']

    return features, labels

def display_stats(cifar10_dataset_folder_path, batch_id, sample_id):
    features, labels = load_cfar10_batch(cifar10_dataset_folder_path, batch_id)

    if not (0 <= sample_id < len(features)):
        print('{} samples in batch {}.  {} is out of range.'.format(len(features), batch_id, sample_id))
        return None

    print('\nStats of batch #{}:'.format(batch_id))
    print('# of Samples: {}\n'.format(len(features)))

    label_counts = dict(zip(*np.unique(labels, return_counts=True)))
    for key, value in label_counts.items():
        print('Label Counts of [{}]({}) : {}'.format(key, label_names[key].upper(), value))

    sample_image = features[sample_id]
    sample_label = labels[sample_id]

    print('\nExample of Image {}:'.format(sample_id))
    print('Image - Min Value: {} Max Value: {}'.format(sample_image.min(), sample_image.max()))
    print('Image - Shape: {}'.format(sample_image.shape))
    print('Label - Label Id: {} Name: {}'.format(sample_label, label_names[sample_label]))

    plt.imshow(sample_image)

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import numpy as np

# Explore the dataset
batch_id = 3
sample_id = 7000
display_stats(cifar10_dataset_folder_path, batch_id, sample_id)

def normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x - min_val) / (max_val - min_val)

    x = x.astype('float32')

    return x

def one_hot_encode(x):
    encoded = np.zeros((len(x), 10))

    for idx, val in enumerate(x):
        encoded[idx][val] = 1

    return encoded

def _preprocess_and_save(normalize, one_hot_encode, features, labels, filename):
    features = normalize(features)
    labels = one_hot_encode(labels)

    pickle.dump((features, labels), open(filename, 'wb'))


def preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode):
    n_batches = 5
    valid_features = []
    valid_labels = []
    all_features = []
    all_labels = []

    for batch_i in range(1, n_batches + 1):
        features, labels = load_cfar10_batch(cifar10_dataset_folder_path, batch_i)

        # find index to be the point as validation data in the whole dataset of the batch (10%)
        index_of_validation = int(len(features) * 0.1)

        valid_features.extend(features[-index_of_validation:])
        valid_labels.extend(labels[-index_of_validation:])
        all_features.extend(features[:-index_of_validation])
        all_labels.extend(labels[:-index_of_validation])

    # preprocess the all stacked validation dataset
    _preprocess_and_save(normalize, one_hot_encode,
                         np.array(valid_features), np.array(valid_labels),
                         Datasets_folder + '/preprocess_validation.p')

    _preprocess_and_save(normalize, one_hot_encode,
                         np.array(all_features), np.array(all_labels),
                         Datasets_folder + '/preprocess_all.p')

preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode)


valid_features, valid_labels = pickle.load(open(Datasets_folder + '/preprocess_validation.p', mode='rb'))
print(valid_features.dtype)
print(valid_features.shape)


train_features, train_labels = pickle.load(open(Datasets_folder + '/preprocess_all.p', mode='rb'))
print(train_features.dtype)
print(train_features.shape)

######Neural Network definition######
def cnn_model_fn(features, labels, mode):
    with tf.name_scope('model_input') as scope:
        input_layer = tf.reshape(features, [-1, 32, 32, 3], name="input")

    with tf.name_scope('model_conv1') as scope:
        conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[2, 2],
                                 padding="same", activation=tf.nn.relu6,
                                 trainable=mode == tf.estimator.ModeKeys.TRAIN)
    with tf.name_scope('model_conv2') as scope:
        conv2 = tf.layers.conv2d(inputs=conv1, filters=32, kernel_size=[2, 2],
                                 padding="same", activation=tf.nn.relu6,
                                 trainable=mode == tf.estimator.ModeKeys.TRAIN)
        pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=(2, 2))
        drop1 = tf.layers.dropout(inputs=pool1, rate=0.2)

    with tf.name_scope('model_conv3') as scope:
        conv3 = tf.layers.conv2d(inputs=drop1, filters=64, kernel_size=[2, 2],
                                 padding="same", activation=tf.nn.relu6,
                                 trainable=mode == tf.estimator.ModeKeys.TRAIN)

    with tf.name_scope('model_conv4') as scope:
        conv4 = tf.layers.conv2d(inputs=conv3, filters=64, kernel_size=[2, 2],
                                 padding="same", activation=tf.nn.relu6,
                                 trainable=mode == tf.estimator.ModeKeys.TRAIN)
        pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=(2, 2))
        drop2 = tf.layers.dropout(inputs=pool2, rate=0.3)

    with tf.name_scope('model_conv5') as scope:
        conv5 = tf.layers.conv2d(inputs=drop2, filters=128, kernel_size=[2, 2],
                                 padding="same", activation=tf.nn.relu6,
                                 trainable=mode == tf.estimator.ModeKeys.TRAIN)

    with tf.name_scope('model_conv6') as scope:
        conv6 = tf.layers.conv2d(inputs=conv5, filters=128, kernel_size=[2, 2],
                                 padding="same", activation=tf.nn.relu6,
                                 trainable=mode == tf.estimator.ModeKeys.TRAIN)
        pool3 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[2, 2], strides=(2, 2))
        drop3 = tf.layers.dropout(inputs=pool3, rate=0.4)

    with tf.name_scope('model_dense') as scope:
        flat = tf.reshape(drop3, [-1, 4 * 4 * 128])

        dense = tf.layers.dense(inputs=flat, units=1024,
                                activation=tf.nn.relu6,
                                trainable=mode == tf.estimator.ModeKeys.TRAIN)

        drop4 = tf.layers.dropout(inputs=dense, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    with tf.name_scope('model_output') as scope:
        logits = tf.layers.dense(inputs=drop4, units=10, trainable=mode == tf.estimator.ModeKeys.TRAIN)
        # shape should be:[-1, 10]

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # PREDICT mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        export_outputs = {
            'predict_output': tf.estimator.export.PredictOutput(predictions)
        }

        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)

    # TRAIN mode
    if mode == tf.estimator.ModeKeys.TRAIN:
        g = tf.get_default_graph()
        tf.contrib.quantize.create_training_graph(input_graph=g, quant_delay=6000)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # EVAL mode
    g = tf.get_default_graph()
    tf.contrib.quantize.create_eval_graph(input_graph=g)
    labels = tf.argmax(labels, axis=1)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


######Training and saving######
run_cfg = tf.estimator.RunConfig(
    model_dir=savedir,
    tf_random_seed=2,
    save_summary_steps=2,
    # session_config = sess_config,
    save_checkpoints_steps=100,
    keep_checkpoint_max=1)


# Instantiate the Estimator
cifar10_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, config=run_cfg)


# Set up logging for predictions
tensors_to_log = {"probabilities": "softmax_tensor"}

logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=100)



def fit_all_batches(xEpochs):
    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=train_features,
        y=train_labels,
        batch_size=32,
        num_epochs=xEpochs,
        shuffle=False)

    # train one step and display the probabilties
    cifar10_classifier.train(
        input_fn=train_input_fn,
        steps=None,
        hooks=[logging_hook])

fit_all_batches(100)

defaultGraph = tf.get_default_graph()

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x=valid_features,
    y=valid_labels,
    num_epochs=1,
    shuffle=False)

eval_results = cifar10_classifier.evaluate(input_fn=eval_input_fn,
                                           hooks=[logging_hook])

print(eval_results)

sample_image = valid_features[3:4]

sample_image.shape


label_names



def predictions(nfeature):
    pred_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=valid_features[nfeature:nfeature + 1],
        y=valid_labels[nfeature:nfeature + 1],
        num_epochs=1,
        shuffle=False)

    pred_results = cifar10_classifier.predict(
        input_fn=pred_input_fn,
        hooks=[logging_hook])

    results = list(pred_results)
    results = results[0]

    y_pred_flat = results['probabilities']

    y_pred = np.column_stack((label_names, y_pred_flat))
    print(y_pred)
    plt.imshow(valid_features[nfeature])

predictions(6)

[print(n.name) for n in defaultGraph.as_graph_def().node]

def serving_input_receiver_fn():
    feature_tensor = tf.placeholder(tf.float32, [None, 32, 32, 3])
    return tf.estimator.export.TensorServingInputReceiver(feature_tensor, {'input': feature_tensor})

input_receiver_fn_map = {
    tf.estimator.ModeKeys.PREDICT: serving_input_receiver_fn,
}

cifar10_classifier.experimental_export_all_saved_models(
    savedir,
    input_receiver_fn_map)

