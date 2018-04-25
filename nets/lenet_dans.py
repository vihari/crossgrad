from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import time
from nets import gan_utils

slim = tf.contrib.slim
DEBUG = False


def current_time_millis():
    return int(round(time.time() * 1000))

EMBEDDING_SIZE = 128


def lenet(images, is_training=True, dropout_keep_prob=.5, scope='LeNet'):
    end_points = {}

    with tf.variable_scope(scope, 'LeNet', [images], initializer=tf.truncated_normal_initializer(stddev=0.1)):
        net = slim.conv2d(images, 32, [5, 5], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], 2, scope='pool1')
        
        net = slim.conv2d(net, 64, [5, 5], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], 2, scope='pool2')

        net = slim.conv2d(net, 128, [5, 5], scope='conv3')
        net = slim.max_pool2d(net, [2, 2], 2, scope='pool3')

        
        net = slim.conv2d(net, 128, [5, 5], scope='convn4')
        net = slim.max_pool2d(net, [2, 2], 2, scope='pooln4')
        
        net = slim.flatten(net)
        end_points['Flatten'] = net

        net = slim.fully_connected(net, EMBEDDING_SIZE, scope='fc3')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout3')


        net = slim.fully_connected(net, EMBEDDING_SIZE, scope='label_fcn1')

        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout4')
    return net, end_points


"""
def lenet(images, is_training=True, dropout_keep_prob=.5, scope='LeNet', reuse=False):
    end_points = {}


    with tf.variable_scope(scope, 'LabelResNet', [images], initializer=tf.truncated_normal_initializer(stddev=0.1)):
        net = resnet1.resnet(images, 2, EMBEDDING_SIZE, reuse=reuse)


        # batch size x 128
        #net = tf.squeeze(net, axis=[1,2])
        end_points['Flatten'] = net

        #net = slim.fully_connected(net, EMBEDDING_SIZE, scope='fc3')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout3')

        net = slim.fully_connected(net, EMBEDDING_SIZE, scope='fc4')

    return net, end_points
"""

def style_model(net, num_styles, activation_fn=tf.nn.relu):

    net = slim.fully_connected(net, num_outputs=EMBEDDING_SIZE//2, scope='style_fc1', activation_fn=activation_fn)
    net = slim.fully_connected(net, num_outputs=num_styles, scope='style_fc2', activation_fn=tf.identity)
    return net
    
def label_model(net, num_labels, activation_fn=tf.nn.relu):
    print (tf.get_variable_scope().name)

    net = slim.fully_connected(net, num_outputs=EMBEDDING_SIZE//2, scope='label_fc1', activation_fn=activation_fn)
    net = slim.fully_connected(net, num_outputs=num_labels, scope='label_fc2', activation_fn=tf.identity)
    return net




def descriminator(net, label, num_classes, num_styles):
    with tf.variable_scope('GAN'):
        fun = tf.nn.relu
        net = slim.fully_connected( tf.concat(axis=1,values=[net, net]), 2*EMBEDDING_SIZE, scope='desc1', activation_fn=fun)
        net = slim.fully_connected(net, EMBEDDING_SIZE, scope='desc2', activation_fn=fun)
        net = slim.fully_connected(net, EMBEDDING_SIZE, scope='desc3', activation_fn=fun)
        net = slim.fully_connected(net, EMBEDDING_SIZE, scope='desc4', activation_fn=fun)
        #net = slim.fully_connected(net, EMBEDDING_SIZE, scope='desc2', activation_fn=tf.nn.relu6)
        net = slim.fully_connected(net, num_styles, scope='desc5', activation_fn=None)
        # in the descriminator, only learn the descriminator weights.
        logits = net
    return logits

def add_GAN_loss (net, style, label, num_styles, num_labels):
    style = slim.one_hot_encoding(style, num_styles)

    global_step = slim.get_global_step()
    p = tf.cast(tf.maximum(global_step - 2000, 0), tf.float32)/50000.
    l = tf.stop_gradient((2./(1. + tf.exp(-15.*p))) -1)
    net = gan_utils.reverse_grad(net, l)
    gan_logits = descriminator(net, label, num_labels, num_styles)

    gan_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = style, logits = gan_logits),name='gan_loss')
 
    global_step = slim.get_global_step()
    gan_loss = tf.cond(tf.equal(tf.mod(global_step, 100), 0), lambda: tf.Print(gan_loss, [gan_loss, l], message="GAN Loss: "), lambda: gan_loss)
    tf.losses.add_loss(gan_loss)

    
def add_losses (net, style, label, num_styles, num_labels, inputs, scales):
    style = slim.one_hot_encoding(style, num_styles)
    batch_size = tf.shape(inputs)[0]
    
    # inputs = tf.Print(inputs, [tf.reduce_max(lg), tf.reduce_min(lg), tf.reduce_max(sg), tf.reduce_min(sg)], message="Grads")
    with tf.variable_scope('crossgrad', reuse=True):
        label_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = label, logits = label_model(net, num_labels)), name='Label_loss')

    
    global_step = slim.get_global_step()
    label_loss = tf.cond(tf.equal(tf.mod(global_step, 100), 0), lambda: tf.Print(label_loss, [label_loss], message="Label Loss: "), lambda: label_loss)

    #tf.losses.add_loss(2*perturb_norm)
    tf.losses.add_loss(label_loss)
    # tf.losses.add_loss(label_perturb_loss)
    # tf.losses.add_loss(style_perturb_loss)
    #tf.losses.add_loss(style_loss)
        
def adaptive_lenet(instance_list, num_classes, is_training=False,
                   dropout_keep_prob=0.5,
                   prediction_fn=tf.nn.softmax,
                   scope='LeNet'):
    # TODO: make this an argument
    num_uids = 80

    import numpy as np
    assert type(instance_list) == list

    input_data = tf.stack([inst.data for inst in instance_list])
    labels = tf.stack([inst.label for inst in instance_list])
    uids = tf.stack([inst.uid for inst in instance_list])
    u_data = tf.stack([inst.get_user_data for inst in instance_list])
    u_labels = tf.stack([inst.get_user_labels for inst in instance_list])
    batch_size = tf.cast(tf.shape(labels)[0], tf.int32)
    f_path_batch = tf.stack([inst.file_path for inst in instance_list])
    u_file_paths_batch = tf.stack([inst.user_file_paths for inst in instance_list])

    with tf.variable_scope('crossgrad'):
        net, end_points = lenet(input_data, is_training)
        logits = label_model(net, num_classes)
        label_prob = tf.nn.softmax(logits)

    if is_training:
        add_losses(net, uids , labels, num_uids, num_classes, input_data, [1,1,1])
	add_GAN_loss(net, uids , label_prob, num_uids, num_classes)
    
    end_points['Logits'] = logits
    end_points['Predictions'] = prediction_fn(logits) 

    return logits, end_points

adaptive_lenet.default_image_size = 28
