# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import tensorflow as tf

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory

from data_instance import DataStream, DataType, Task

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
    'batch_size', 100, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
                                'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')

tf.app.flags.DEFINE_string(
    'metadir', None, 'Folder containing the labels.tsv and the sprite image of the test set')

tf.app.flags.DEFINE_boolean(
    'target_style', False, 'Set to true if the target of the classifier is style rather than the content')

FLAGS = tf.app.flags.FLAGS


def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        tf_global_step = slim.get_or_create_global_step()

        ######################
        # Select the dataset #
        ######################
        dataset = dataset_factory.get_dataset(
            FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

        num_classes = dataset.num_styles if FLAGS.target_style else dataset.num_classes
        ####################
        # Select the model #
        ####################
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=(num_classes - FLAGS.labels_offset),
            is_training=False
        )

        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=False)

        eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size

        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            common_queue_capacity=20 * FLAGS.batch_size,
            common_queue_min=10 * FLAGS.batch_size, shuffle=False)
        [data, content_label, style_label, file_path, u_data, u_labels, u_file_paths] = \
            provider.get(['image', 'label', 'uid', 'file_path', 'user_set/images', 'user_set/labels',
                          'user_set/file_paths'])
        content_label -= FLAGS.labels_offset

        data = image_preprocessing_fn(data, eval_image_size, eval_image_size)
        u_data = tf.map_fn(
            lambda u_instance: image_preprocessing_fn(u_instance, eval_image_size, eval_image_size),
            u_data, dtype=tf.float32)

        image_batch, content_label_batch, style_label_batch, f_path_batch, u_data_batch, u_labels_batch, u_file_paths_batch = \
            tf.train.batch(
                [data, content_label, style_label, file_path, u_data, u_labels, u_file_paths],
                batch_size=FLAGS.batch_size,
                num_threads=FLAGS.num_preprocessing_threads,
                capacity=5 * FLAGS.batch_size,
                #allow_smaller_final_batch=True,
                dynamic_pad=True)

        label_batch = style_label_batch//2 if FLAGS.target_style else content_label_batch
        num_classes = dataset.num_styles if FLAGS.target_style else dataset.num_classes
        label_batch = slim.one_hot_encoding(
            label_batch, num_classes - FLAGS.labels_offset)
        image_batch = tf.Print(image_batch, [tf.reduce_mean(image_batch)], message="mean")
        ####################
        # Define the model #
        ####################
        ds = DataStream(Task.CLASSIFICATION, DataType.IMAGE)
        # = batch_queue.dequeue()
        data_instance_list = [ds.encode(*t) for t in
                              zip(tf.unstack(image_batch), tf.unstack(content_label_batch), tf.unstack(style_label_batch),
                                  tf.unstack(u_data_batch), tf.unstack(u_labels_batch), tf.unstack(f_path_batch),
                                  tf.unstack(u_file_paths_batch))]
        f_path_batch = tf.Print(f_path_batch, [tf.shape(f_path_batch)], message='path batch')
        with tf.variable_scope('network_fn'):
            logits, _ = network_fn(data_instance_list)

        if FLAGS.moving_average_decay:
            variable_averages = tf.train.ExponentialMovingAverage(
                FLAGS.moving_average_decay, tf_global_step)
            variables_to_restore = variable_averages.variables_to_restore(
                slim.get_model_variables())
            variables_to_restore[tf_global_step.op.name] = tf_global_step
        else:
            variables_to_restore = slim.get_variables_to_restore()

        predictions = tf.argmax(logits, 1)
        loss = tf.losses.softmax_cross_entropy(label_batch, logits, weights=1.0)
        label_batch = tf.argmax(label_batch, 1)
        # predictions = tf.Print(predictions, data=[loss], message="Loss value")
        
        # Define the metrics:
        idxs = tf.squeeze(tf.where(tf.not_equal(predictions, label_batch)))
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'Accuracy': slim.metrics.streaming_accuracy(predictions, label_batch),
            'Recall@5': slim.metrics.streaming_recall_at_k(
                logits, label_batch, 5),
            'uid_missed': slim.metrics.streaming_concat(tf.reshape(tf.gather(style_label_batch, idxs), [-1]))
        })

        _md = os.path.join(FLAGS.eval_dir, FLAGS.dataset_split_name, "mistakes/")
        if not os.path.exists(_md):
            os.system("mkdir -p %s" % _md)
        mis_dir = tf.constant(_md)
        # uid_mistakes = tf.get_variable("uids_mistaken", initializer=tf.zeros([0], dtype=tf.int64))
        # with tf.control_dependencies(names_to_updates.values()):
        #    um_v, um_u = 
            # uid_mistakes = tf.concat([uid_mistakes, ], 0)
            # uid_batch = tf.Print(uid_batch, [uid_batch, tf.shape(uid_batch)], "uids and its shape")
        """
        with tf.control_dependencies(names_to_updates.values()):
            idxs = tf.cast(idxs, tf.int32)
            # eval_op = tf.Print(idxs, [idxs, tf.shape(idxs)], message="IDX and shape")
            def body(i):
                fp = f_path_batch[idxs[i]]
                s = tf.string_split([fp], "/").values
                fp = tf.string_join([s[0], s[1]], "_")
                                    
                w_op = tf.write_file(
                    tf.string_join([mis_dir, fp]),
                    #label_batch[idx],
                    #tf.constant("_as_"),
                    #predictions[idx]]),
                    tf.image.encode_png(tf.cast(image_batch[idxs[i]]*128+128, tf.uint8)))
                deps = [tf.cond(tf.rank(f_path_batch)>0, lambda: w_op, lambda: tf.no_op())]
                with tf.control_dependencies(deps):
                    i += 1
                # i = tf.Print(i, [tf.string_join([mis_dir, fp])], message="filename")
                return [i]

            eval_op = tf.while_loop(
                lambda i: tf.less(i, tf.shape(f_path_batch)[0]),
                body,
                [tf.constant(0)])"""
                        
        # Print the summaries to screen.
        for name, value in names_to_values.iteritems():
            if name == 'uid_missed':
                continue
            summary_name = 'eval/%s' % name
            op = tf.summary.scalar(summary_name, value, collections=[])
            op = tf.Print(op, [value], summary_name)
            tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

        # TODO(sguada) use num_epochs=1
        if FLAGS.max_num_batches:
            num_batches = FLAGS.max_num_batches
        else:
            # This ensures that we make a single pass over all of the data.
            num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))

        if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        else:
            checkpoint_path = FLAGS.checkpoint_path

        tf.logging.info('Evaluating %s' % checkpoint_path)


        print ("debug %s %s " % (label_batch, predictions))
        _um, conf_matrix, otp, pred = slim.evaluation.evaluate_once(
            master=FLAGS.master,
            checkpoint_path=checkpoint_path,
            logdir=FLAGS.eval_dir,
            num_evals=num_batches,
            eval_op=names_to_updates.values(),
            variables_to_restore=variables_to_restore,
            final_op=[names_to_values['uid_missed'], tf.confusion_matrix(label_batch, predictions, num_classes=dataset.num_classes), label_batch, predictions]
        )

        import numpy as np
        import sys
        import collections
        
        # full print of the confusion matrix
        np.set_printoptions(threshold=np.nan, linewidth=np.inf) #, formatter={'int': '{: 03d}'.format})
        #names = [dataset.labels_to_names[label] for label in range(dataset.num_classes)[:36]]
        #sys.stdout.write("  " + str(np.asarray(names)) + "\n")
        #for i, row in enumerate(conf_matrix):
            #sys.stdout.write (names[i] + str(row) + '\n')

        #print (conf_matrix)
        print (len(pred), pred)
        print(len(otp), otp)

        #_um = [(u//2)*2 for u in _um]

        _um = [u for u in _um]

        font_freq = collections.Counter(_um)
        print ("Mistakes per font label: %s" % font_freq)
        print ("Total number of mistaken uids %d" % len(_um))
        
        if FLAGS.metadir:
            import scipy.misc as misc
            im_file = os.path.join(FLAGS.metadir, "sprite.png")
            labels_file = os.path.join(FLAGS.metadir, "labels.tsv")
            if os.path.exists(im_file) and os.path.exists(labels_file):
                sprite_img = misc.imread(im_file)
                with open(labels_file, "r") as lf:
                    labels = [int(line) for line in lf.readlines()]
                ncol = int(math.sqrt(len(labels)))+1
                ms_arr = np.zeros([ncol, ncol], np.int32)
                for li in range(len(labels)):
                    ms_arr[li//ncol, li%ncol] = font_freq[labels[li]]

                _ims = eval_image_size
                mask = np.zeros([_ims*ncol, _ims*ncol])
                mx = max(font_freq.values())
                for r in range(ncol):
                    for c in range(ncol):
                        mask[r*_ims:(r+1)*_ims, c*_ims:(c+1)*_ims] = (ms_arr[r][c]/mx)*255. 
                misc.imsave(os.path.join(FLAGS.eval_dir, FLAGS.dataset_split_name, "mask.png"), mask)
                print ("Number of mistakes per font %s", ms_arr)
            else:
                print ("Metadata dir supplied is missing either the sprite image or labels file")
        #print(conf_matrix)

if __name__ == '__main__':
    tf.app.run()
