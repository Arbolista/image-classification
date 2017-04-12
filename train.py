import tensorflow as tf
import json
import datetime
import helper
import pickle
import numpy as np

valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))

n_classes = 10
image_shape = (32, 32, 3)
config = None
    #
    # Initializers
    #

def neural_net_image_input():
    """
    Return a Tensor for a batch of image input
    : image_shape: Shape of the images
    : return: Tensor for image input.
    """
    shape = [None]
    for dim in image_shape:
        shape.append(dim)
    return tf.placeholder(tf.float32, shape=shape, name="x")


def neural_net_label_input():
    """
    Return a Tensor for a batch of label input
    : return: Tensor for label input.
    """
    return tf.placeholder(tf.int32, shape=[None, n_classes], name="y")


def neural_net_keep_prob_input():
    """
    Return a Tensor for keep probability
    : return: Tensor for keep probability.
    """
    return tf.placeholder(tf.float32, name="keep_prob")

#
# Network Model
#

def conv2d_maxpool(x_tensor, conv={}):

    conv_num_outputs = conv['out']
    conv_ksize = conv['size']
    n_in = x_tensor.get_shape().as_list()[-1]
    W_shape = [conv_ksize[0], conv_ksize[1], n_in, conv_num_outputs];

    W = tf.Variable(tf.truncated_normal(W_shape, stddev=conv['weights']['stddev'], mean=conv['weights']['mean']))
    weights.append(W)
    B = tf.Variable(tf.zeros([conv_num_outputs]))

    conv_strides = conv['strides']
    x = tf.nn.conv2d(x_tensor, W, strides=[1, conv_strides[0], conv_strides[1], 1], padding='SAME')
    x = tf.nn.bias_add(x, B)
    x = tf.nn.relu(x)

    if conv['pool']:
        pool_ksize = conv['pool']['size']
        pool_strides = conv['pool']['strides']
        x = tf.nn.max_pool(x, ksize=[1, pool_ksize[0], pool_ksize[1], 1], strides=[1, pool_strides[0], pool_strides[1], 1], padding='SAME')

    return x

def l2_loss():
    if not config['l2']:
        return 0
    beta = config['l2']['beta']
    return beta * reduce(lambda x, y: x+y, map(lambda w: tf.nn.l2_loss(w), weights))


def flatten(x_tensor):
    shape = x_tensor.get_shape().as_list()
    n_flattened = shape[1] * shape[2] * shape[3]
    return tf.reshape(x_tensor, [-1, n_flattened])

def fully_conn(x_tensor, fc={}):
    n_in = x_tensor.get_shape().as_list()[-1]
    num_outputs = fc['out']
    W_shape = [n_in, num_outputs]

    W = tf.Variable(tf.truncated_normal(W_shape, stddev=fc['weights']['stddev'], mean=fc['weights']['mean']))
    weights.append(W)
    B = tf.Variable(tf.zeros([num_outputs]))
    x = tf.matmul(x_tensor, W) + B
    if fc['relu']:
        return tf.nn.relu(x)
    return x

def output(x_tensor):
    """
    Apply a output layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    n_in = x_tensor.get_shape().as_list()[-1]
    W_shape = [n_in, n_classes]

    out = config['out']
    W = tf.Variable(tf.truncated_normal(W_shape, stddev=1.0, mean=0.1))
    weights.append(W)
    B = tf.Variable(tf.zeros([n_classes]))
    x = tf.matmul(x_tensor, W) + B
    if out['relu']:
        return tf.nn.relu(x)
    return x


def conv_net(x, keep_prob):
    tensors = [x]

    for conv in config['conv']:
        tensors.append(conv2d_maxpool(tensors[-1], conv=conv))
        if conv['dropout']:
            tensors.append(tf.nn.dropout(tensors[-1], keep_prob))

    # Apply a Flatten Layer
    # Function Definition from Above:
    tensors.append(flatten(tensors[-1]))

    #  Apply 1, 2, or 3 Fully Connected Layers
    for fc in config['fc']:
        tensors.append(fully_conn(tensors[-1], fc=fc))
        if fc['dropout']:
            tensors.append(tf.nn.dropout(tensors[-1], keep_prob))

    return output(tensors[-1])

def train_neural_network(session, optimizer, feature_batch, label_batch):
    keep_probability = config['keep_probability']
    feed_dict = {
        'x:0': feature_batch,
        'y:0': label_batch,
        'keep_prob:0': keep_probability
    }
    session.run(optimizer, feed_dict=feed_dict)

def print_stats(session, cost, summary, accuracy, logits, train_writer, valid_writer, feature_batch, label_batch, epoch=0):
    train_feed_dict = {
        'x:0': feature_batch,
        'y:0': label_batch,
        'keep_prob:0': 1.0
    }
    cost_, summary_ = session.run([cost, summary], feed_dict=train_feed_dict)
    train_writer.add_summary(summary_, epoch)
    print('Train cost=', cost_)

    valid_feed_dict = {
        'x:0': valid_features,
        'y:0': valid_labels,
        'keep_prob:0': 1.0
    }
    valid_accuracy, summary_ = session.run([accuracy, summary], feed_dict=valid_feed_dict)
    valid_writer.add_summary(summary_, epoch)
    print('Valid accuracy=', valid_accuracy)

def run(conf):
    global config
    config = conf
    global weights
    weights = []
    tf.reset_default_graph()

    with tf.Graph().as_default() as graph:
        # Inputs
        x = neural_net_image_input()
        y = neural_net_label_input()
        keep_prob = neural_net_keep_prob_input()

        # Model
        logits = conv_net(x, keep_prob)

        # Name logits Tensor, so that is can be loaded from disk after training
        logits = tf.identity(logits, name='logits')

        # Loss and Optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
        tf.summary.scalar('cost', cost)
        if config['optimizer'] == 'GradientDescent':
            Optimizer = tf.train.GradientDescentOptimizer
        else:
            Optimizer = tf.train.AdamOptimizer


        optimizer = Optimizer().minimize(cost)

        # Accuracy
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
        tf.summary.scalar('accuracy', accuracy)

        init = tf.global_variables_initializer()

        summary = tf.summary.merge_all()
        dirname = '%s_%s' % (config['prefix'], datetime.datetime.now().strftime('%Y_%m_%d_%H.%M'))
        train_writer = tf.summary.FileWriter('output/train/' + dirname, graph)
        valid_writer = tf.summary.FileWriter('output/valid/' + dirname , graph)

        #
        # Train
        #

        epochs = config['epochs']
        batch_size = config['batch_size']
        batch_size = config['batch_size']
        n_batches = config['n_batches']

        with tf.Session(graph=graph) as sess:
            # Initializing the variables
            sess.run(init)

            # save hype for later comparisons
            with open('output/train/' + dirname + '/hypes.json', 'w') as f:
                json.dump(config, f, indent=2)
            # Training cycle
            step = 1
            for epoch in range(config['epochs']):
                print('Epoch: %s, Step: %s' % (str(epoch+1), step))
                for batch_i in range(1, n_batches + 1):
                    print('batch_i', batch_i)
                    for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, config['batch_size']):
                        step += config['batch_size']
                        train_neural_network(sess, optimizer, batch_features, batch_labels)
                print_stats(sess, cost, summary, accuracy, logits, train_writer, valid_writer, batch_features, batch_labels, epoch=epoch+1)

            # Save Model
            saver = tf.train.Saver()
            save_model_path = 'output/train/%s/image_classification' % dirname
            save_path = saver.save(sess, save_model_path)
            print('Done. Saved!')


if __name__ == '__main__':
    config = {
      "optimizer": "Adam",
      "conv": [
        {
          "dropout": False,
          "strides": [
            1,
            1
          ],
          "weights": {
            "stddev": 1.0,
            "mean": 0.1
          },
          "size": [
            2,
            2
          ],
          "pool": {
            "strides": [
              2,
              2
            ],
            "size": [
              4,
              4
            ]
          },
          "out": 128
        }
      ],
      "learning_rate": False,
      "batch_size": 256,
      "epochs": 4,
      "prefix": "current",
      "fc": [
        {
          "dropout": True,
          "weights": {
            "stddev": 1.0,
            "mean": 0.1
          },
          "relu": True,
          "out": 1024
        }
      ],
      "keep_probability": 0.9,
      "l2": False,
      "n_batches": 5,
      "out": {
        "relu": False
      }
    }
    run(config)
