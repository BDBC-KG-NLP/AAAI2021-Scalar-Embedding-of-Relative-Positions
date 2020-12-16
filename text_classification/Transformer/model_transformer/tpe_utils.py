# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 15:37:04 2020

@author: wujs
"""
import tensorflow as tf
import math

def get_timing_signal_1d(length, channels, min_timescale=1.0, max_timescale=1.0e4):
  """Gets a bunch of sinusoids of different frequencies.
  Each channel of the input Tensor is incremented by a sinusoid of a different
  frequency and phase.
  This allows attention to learn to use absolute and relative positions.
  Timing signals should be added to some precursors of both the query and the
  memory inputs to attention.
  The use of relative position is possible because sin(x+y) and cos(x+y) can be
  experessed in terms of y, sin(x) and cos(x).
  In particular, we use a geometric sequence of timescales starting with
  min_timescale and ending with max_timescale.  The number of different
  timescales is equal to channels / 2. For each timescale, we
  generate the two sinusoidal signals sin(timestep/timescale) and
  cos(timestep/timescale).  All of these sinusoids are concatenated in
  the channels dimension.
  Args:
  length: scalar, length of timing signal sequence.
  channels: scalar, size of timing embeddings to create. The number of
  different timescales is equal to channels / 2.
  min_timescale: a float
  max_timescale: a float
  Returns:
  a Tensor of timing signals [1, length, channels]
  """
  position = tf.to_float(tf.range(length))
  num_timescales = channels // 2
  log_timescale_increment = (
    math.log(float(max_timescale) / float(min_timescale)) /
    (tf.to_float(num_timescales) - 1))
  inv_timescales = min_timescale * tf.exp(
    tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
  scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
  signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
  signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
  signal = tf.reshape(signal, [1, length, channels])
  return signal