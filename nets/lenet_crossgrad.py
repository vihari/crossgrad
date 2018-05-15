from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import time

slim = tf.contrib.slim
DEBUG = False


def current_time_millis():
    return int(round(time.time() * 1000))

EMBEDDING_SIZE = 128

EPSILON, PERTURB_LOSS_COEFF = 10, .5

def lenet(images, is_training=True, dropout_keep_prob=.5, scope='LeNet'):
    """Copied from TF-slim/nets/lenet.py"""
    end_points = {}

    with tf.variable_scope(scope, 'LeNet', [images], initializer=tf.truncated_normal_initializer(stddev=0.1)):
        net = slim.conv2d(images, 32, [5, 5], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], 2, scope='pool1')
        
        net = slim.conv2d(net, 64, [5, 5], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], 2, scope='pool2')

        net = slim.conv2d(net, 128, [5, 5], scope='conv3')
        net = slim.max_pool2d(net, [2, 2], 2, scope='pool3')
        net = slim.flatten(net)
        # batch size x 3136
        end_points['Flatten'] = net

        net = slim.fully_connected(net, EMBEDDING_SIZE, scope='fc3')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout3')
    return net, end_points

def style_model(net, num_styles, activation_fn=tf.nn.relu):
    net = slim.fully_connected(net, num_outputs=EMBEDDING_SIZE//2, scope='style_fc1', activation_fn=activation_fn)
    net = slim.fully_connected(net, num_outputs=num_styles, scope='style_fc2', activation_fn=activation_fn)
    return net
    
def label_model(net, num_labels, activation_fn=tf.nn.relu):
    print (tf.get_variable_scope().name)
    net = slim.fully_connected(net, num_outputs=EMBEDDING_SIZE//2, scope='label_fc1', activation_fn=activation_fn)
    net = slim.fully_connected(net, num_outputs=num_labels, scope='label_fc2', activation_fn=activation_fn)
    return net



def stylenet(images, is_training=True, dropout_keep_prob=.5, scope='StyleNet'):
    """Copied from TF-slim/nets/lenet.py"""
    end_points = {}

    with tf.variable_scope(scope, 'StyleNet', [images], initializer=tf.truncated_normal_initializer(stddev=0.1)):
        net = slim.conv2d(images, 32, [5, 5], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], 2, scope='pool1')
        
        net = slim.conv2d(net, 64, [5, 5], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], 2, scope='pool2')

        net = slim.conv2d(net, 128, [5, 5], scope='conv3')
        net = slim.max_pool2d(net, [2, 2], 2, scope='pool3')
        net = slim.flatten(net)
        # batch size x 3136
        end_points['Flatten'] = net

        net = slim.fully_connected(net, EMBEDDING_SIZE, scope='fc3')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout3')
    return net, end_points

    
def add_losses (net, style_net, style, label, num_styles, num_labels, inputs, scales):
    style = slim.one_hot_encoding(style, num_styles)
    batch_size = tf.shape(inputs)[0]
    

    # inputs = tf.Print(inputs, [tf.reduce_max(lg), tf.reduce_min(lg), tf.reduce_max(sg), tf.reduce_min(sg)], message="Grads")
    with tf.variable_scope('crossgrad', reuse=True):
        label_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = label, logits = label_model(net, num_labels)), name='Label_loss')
        style_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = style, logits = style_model(style_net, num_styles)), name='Style_loss')

        s = EPSILON
        
        sg = tf.gradients(style_loss, inputs)[0]        
        lg = tf.gradients(label_loss, inputs)[0]
    
        delJS_x = tf.clip_by_value(sg, clip_value_min=-0.1, clip_value_max=0.1)
        delJL_x = tf.clip_by_value(lg, clip_value_min=-0.1, clip_value_max=0.1)

        logit_label_perturb = label_model(lenet(inputs + s*tf.stop_gradient(delJS_x))[0], num_labels)

        logit_style_perturb = style_model(stylenet(inputs + s*tf.stop_gradient(delJL_x))[0], num_styles)

    label_perturb_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logit_label_perturb), name='Label_perturb_loss')

    style_perturb_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=style, logits=logit_style_perturb), name='Style_perturb_loss')

    global_step = slim.get_global_step()
    label_loss = tf.cond(tf.equal(tf.mod(global_step, 100), 0), lambda: tf.Print(label_loss, [label_perturb_loss, label_loss, style_loss,style_perturb_loss, tf.reduce_sum(sg*sg), tf.reduce_max(tf.abs(sg)), tf.reduce_sum(tf.abs(sg))], message="Loss: label_perturb, label, style, style_perturb, sgradl2norm, sgradl0norm, sgradl1norm,  "), lambda: label_loss)
    

    perturb_loss = tf.identity(label_perturb_loss + style_perturb_loss, name='Perturb_loss')

    alpha = PERTURB_LOSS_COEFF

    tf.losses.add_loss( alpha  * perturb_loss)
    tf.summary.scalar("Style_loss", style_loss)
    tf.summary.scalar("Label_loss", label_loss)
    tf.losses.add_loss( (1-alpha) * label_loss)
    tf.losses.add_loss( (1-alpha) * style_loss)
    #tf.losses.add_loss(primary_loss)
        
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
        style_net, _ = stylenet(input_data, is_training)
        logits = label_model(net, num_classes)
        style_logits = style_model(style_net, num_uids)


    if is_training:
        add_losses(net, style_net,  uids, labels, num_uids, num_classes, input_data, [1,1,1])
    
    end_points['Logits'] = logits
    end_points['Predictions'] = prediction_fn(logits) 

    return logits, end_points

adaptive_lenet.default_image_size = 32
