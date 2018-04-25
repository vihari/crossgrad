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
"""Provides utilities for preprocessing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import numpy as np

slim = tf.contrib.slim


def preprocess_image(image, output_height, output_width, is_training):
    image = tf.to_float(image)
    image = tf.reshape(image, [1, 32, 32, 1])
    image = tf.image.resize_bilinear(
        image, [output_height, output_width], align_corners=True)
    image = tf.subtract(image, 128.0)
    image = tf.div(image, 128.0)
    # if is_training:
    #    image += tf.random_uniform(image.get_shape(), 0, .5)
        
    return tf.reshape(image, [output_height, output_width, 1])
