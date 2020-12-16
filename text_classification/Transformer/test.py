# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 21:07:43 2020
"""

import tensorflow as tf
import numpy as np

def f_positive(relative_position, alpha):
  return 1-tf.math.exp(-alpha*relative_position)
  
def f_negative(relative_position, beta):
  return tf.math.exp(beta*relative_position)
  
def relative_position_soft_bucket(relative_position,
                                  A=0.01,bidirectional=True):
  alpha =tf.Variable(initial_value=0.01, trainable=True)
  beta = tf.Variable(initial_value=0.01, trainable=True)
  alpha =A*tf.math.sigmoid(alpha)
  beta = A*tf.math.sigmoid(beta)
  relative_position = tf.cast(relative_position,tf.float32)
  
  '''
  Parameters
  ----------
  relative_position : tf.int32
    DESCRIPTION.
  bidirectional : a boolean, whether the attention is bidirectional
    DESCRIPTION. The default is True.
  num_buckets : an integer, optional
    DESCRIPTION. The default is 32.
  max_distance : an integer, optional
    DESCRIPTION. The default is 128.

  Returns
  -------
  a Tensor with the same shape withe relative_position, containing int32 value 
  in the range (0, num_buckets]
  '''
  ret = 0
  position_ret=0
  n =  relative_position
  
  if bidirectional:
   n = n
  else:
    n = tf.math.minimum(n, 0)
  
  #now n is in range [0, inf]
  is_small = tf.cast(tf.math.less_equal(n, 0),tf.float32)
  
  positive_soft_bucket = f_positive(relative_position, alpha)
  value_if_positive= tf.layers.dense(
    tf.layers.dense(tf.expand_dims(positive_soft_bucket,-1),
                    50,
                    activation=tf.nn.relu,
                    kernel_initializer=tf.random_normal_initializer()
                    ),1,
      kernel_initializer=tf.random_normal_initializer())[:,:,0]
  
  negative_soft_bucket = f_negative(relative_position, beta)
  
  value_if_negative = tf.layers.dense(
    tf.layers.dense(tf.expand_dims(negative_soft_bucket,-1),
                    50,
                    activation=tf.nn.relu,
                    kernel_initializer=tf.random_normal_initializer()
                    ),1,
      kernel_initializer=tf.random_normal_initializer())[:,:,0]
  
  ret = is_small*value_if_negative+(1.0-is_small)*value_if_positive
  
  position_ret = is_small*negative_soft_bucket + (1.0-is_small)*positive_soft_bucket
  
  return position_ret, ret

if __name__ =='__main__':
  c_maxlen=10
  c_context_position = tf.range(c_maxlen,dtype=tf.int32)[:,None]
  c_memory_postion = tf.range(c_maxlen, dtype=tf.int32)[None,:]
  c_relative_position = c_memory_postion - c_context_position
  
  position_ret,ret = relative_position_soft_bucket(c_relative_position)
  
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  
  
  position_ret_val,t5_bias_val=sess.run([position_ret,ret])
  
  #print(np.round(position_ret_val,3),np.round(t5_bias_val,3))
  
  contxt_pos, mem_pos = sess.run([c_context_position, c_memory_postion])
  print(contxt_pos)
  print(mem_pos)