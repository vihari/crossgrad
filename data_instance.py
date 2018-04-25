from enum import Enum
import tensorflow as tf


class Task(Enum):
    CLASSIFICATION = 1
    REGRESSION = 2


class DataType(Enum):
    IMAGE = 1
    TEXT = 2
    SPEECH = 3


class DataStream(object):
    """Data structure that encodes the general structure of the datasets of our interest

    * The data in a data instance can be of any type: image, text, bytes etc
    * target can be either discrete (classification) or continuous (regression)
    * Along with the data, there can be other information such as the identity of the generating agent, for example the
      id of the user who generated a character or a speech sample
    """

    def __init__(self, task, data_type):
        self._task = task
        self._data_type = data_type

    def encode(self, data, label, uid, u_images, u_labels, f_path=None, u_f_paths=None):
        # if self._task == Task.CLASSIFICATION:
        #     dts = [tf.int32, tf.int64, tf.uint8]
        #     assert label.dtype in dts, "The type of label: %s is not one of expected: %s" % (label.dtype, dts)

        return DataInstance(self._data_type, data, label, uid, u_images, u_labels, f_path, u_f_paths)

    @property
    def data_type(self):
        return self._data_type


class DataInstance(object):
    def __init__(self, data_type, data, label, uid, user_data, user_labels, file_path=None, u_file_paths=None):
        # if user_set:
        #   assert type(user_set) == list, "Argument user_set should be a list of tuples: (data, label)"
        #  for t in user_set:
        #      assert type(t) == tuple and len(t) == 2, "Unexpected entry in user set: %s; either the type or " \
        #                                              "length is not correct" % t

        self._data = data
        self._label = label
        self._uid = uid
        self._user_data = user_data
        self._user_labels = user_labels
        self._data_type = data_type
        self._file_path = file_path
        self._u_file_paths = u_file_paths

    @property
    def data(self):
        return self._data

    @property
    def label(self):
        return self._label

    @property
    def is_image(self):
        return self._data_type == DataType.IMAGE

    @property
    def is_text(self):
        return self._data_type == DataType.TEXT

    @property
    def get_user_data(self):
        return self._user_data

    @property
    def get_user_labels(self):
        return self._user_labels

    @property
    def uid(self):
        return self._uid

    @property
    def file_path(self):
        return self._file_path

    @property
    def user_file_paths(self):
        return self._u_file_paths
