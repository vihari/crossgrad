"""
Data description for offline hand-written image dataset
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os

from tensorflow.contrib.slim.python.slim.data.data_decoder import DataDecoder

slim = tf.contrib.slim

_FILE_PATTERN = 'hwoffline_%s_*.tfrecord'

SPLITS_TO_SIZES = {'train': 10112, 'test': 2533, 'validation': 1200, 'test_ext': 1200}
#SPLITS_TO_SIZES = {'train': 665, 'test': 180}
# SPLITS_TO_SIZES = {'train': 36, 'test': 12}

_NUM_CLASSES = 36
_NUM_STYLES = 80

#SPLITS_TO_SIZES = {'train': 15133, 'test': 3132, 'validation': 3132, 'test_ext': 1200}

#_NUM_CLASSES = 111
#_NUM_STYLES = 74

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A gray scale image of width and height 320.',
    'label': 'A single integer between 0 and 10',
    'uid': 'A single integer between 0 and 40 that encodes the id of the generating user',
    'file_path': 'Path of file corresponding to this datum',
    'user_set/images': 'Images from other examples from the same user given by uid',
    'user_set/labels': 'Labels from other examples from the same user given by uid',
    'user_set/file_paths': 'The paths corresponding to each og the examples of the user'
}


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
    """Gets a dataset tuple with instructions for reading flowers.

  Args:
    split_name: A train/validation split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.
    reader: The TensorFlow reader type.

  Returns:
    A `Dataset` namedtuple.

  Raises:
    ValueError: if `split_name` is not a valid train/validation split.
  """
    if split_name not in SPLITS_TO_SIZES:
        raise ValueError('split name %s was not recognized.' % split_name)

    # if not file_pattern:
    #    file_pattern = _FILE_PATTERN
    # file_pattern = os.path.join(dataset_dir, file_pattern % split_name)
    if split_name == 'train':
        file_pattern = 'hwoffline_train_0000*.tfrecord'
    elif split_name == 'test':
        file_pattern = 'hwoffline_test_0000*.tfrecord'
    elif split_name == 'validation':
        file_pattern = 'hwoffline_validation_0000*.tfrecord'
    else:
        file_pattern = 'hw_test/hwoffline_test_000*.tfrecord'
    file_pattern = os.path.join(dataset_dir, file_pattern)

    from tensorflow.python.ops import parsing_ops

    # Allowing None in the signature so that dataset_factory can use the default.
    if reader is None:
        reader = tf.TFRecordReader

    keys_to_features = {
        'image': tf.FixedLenFeature((), tf.string, default_value=''),
        'format': tf.FixedLenFeature((), tf.string, default_value='raw'),
        'label': tf.FixedLenFeature(
            [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
        'file_path': tf.FixedLenFeature((), tf.string, default_value=''),
        'uid': tf.FixedLenFeature(
            [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
        'user_set/images': tf.VarLenFeature(tf.string),
        'user_set/labels': tf.VarLenFeature(tf.int64),
        'user_set/file_paths': tf.VarLenFeature(tf.string),
    }

    class Decoder(DataDecoder):
        def __init__(self, _keys_to_features):
            self._keys_to_features = _keys_to_features

        def decode(self, data, items):
            example = parsing_ops.parse_single_example(data, self._keys_to_features)
            # Reshape non-sparse elements just once:
            for k in self._keys_to_features:
                v = self._keys_to_features[k]
                if isinstance(v, parsing_ops.FixedLenFeature):
                    example[k] = tf.reshape(example[k], v.shape)

            # example['image'] = tf.reshape(parsing_ops.decode_raw(example['image'], tf.uint8), [32, 32, 1])

            outputs = []
            for item in items:
                if item == 'image':
                    outputs.append(tf.reshape(parsing_ops.decode_raw(example['image'], tf.uint8), [32, 32]))
                elif item == 'label':
                    outputs.append(example['label'])
                elif item == 'uid':
                    outputs.append(example['uid'])
                elif item == 'file_path':
                    outputs.append(example['file_path'])
                elif item == 'user_set/labels':
                    st_labels = example['user_set/labels']
                    outputs.append(tf.sparse_to_dense(st_labels.indices, st_labels.dense_shape, st_labels.values))
                elif item == 'user_set/images':
                    st_labels = example['user_set/labels']
                    st_images = example['user_set/images']
                    num_examples = tf.shape(st_labels)[0]
                    outputs.append(tf.reshape(parsing_ops.decode_raw(st_images.values, tf.uint8),
                                              [num_examples, 32, 32]))
                elif item == 'user_set/file_paths':
                    st_paths = example['user_set/file_paths']
                    outputs.append(tf.sparse_to_dense(st_paths.indices, st_paths.dense_shape, st_paths.values,
                                                      default_value=''))
            return outputs

        def list_items(self):
            return _ITEMS_TO_DESCRIPTIONS.keys()

    #labels_to_names = {i: chr(ord('A') + i) for i in range(26)}
    #labels_to_names.update({len(labels_to_names) + i: chr(ord('0') + i) for i in range(10)})
    labels_to_names = {  chr(ord('0') + i):i for i in range(_NUM_CLASSES)}
    # confusive_chars = ['S', '8', 'E', 'F', 'J', 'I', 'Q', 'O', 'Y', 'V', 'N', 'W']
    # labels_to_names = {_i: c for _i, c in enumerate(confusive_chars)}
    
    return slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=reader,
        decoder=Decoder(keys_to_features),
        num_samples=SPLITS_TO_SIZES[split_name],
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
        num_classes=_NUM_CLASSES,
        num_styles=_NUM_STYLES,
        labels_to_names=labels_to_names)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    data_dir = os.path.expanduser("~/data/image_fonts/")

    with tf.Graph().as_default():
        dataset = get_split('train', data_dir)
        data_provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset, common_queue_capacity=32, common_queue_min=1)

        image, label, uid, u_images, u_labels = data_provider.get(
            ['image', 'label', 'uid', 'user_set/images', 'user_set/labels'])

        with tf.Session() as sess:
            with slim.queues.QueueRunners(sess):
                for i in xrange(4):
                    np_image, np_label, np_uid, np_uimages, np_ulabels = sess.run(
                        [image, label, uid, u_images, u_labels])
                    height, width = np_image.shape
                    class_name = name = np_label
                    print("Shape of labels %s, of images %s " % (np_ulabels.shape, np_uimages.shape))

                    # plt.figure()
                    # plt.imshow(np_image)
                    # plt.title('%s, %d x %d' % (name, height, width))
                    # plt.axis('off')
                    # plt.show()
                    #
                    # plt.figure()
                    # plt.imshow(np_uimages[0])
                    # plt.title('%s, %d x %d' % ('user set', height, width))
                    # plt.axis('off')
                    # plt.show()

                images, labels, uids, ui_b, ul_b = tf.train.batch(
                    [image, label, uid, u_images, u_labels],
                    batch_size=10, dynamic_pad=True)
                print (sess.run([tf.shape(ui_b), tf.shape(ul_b)]))
