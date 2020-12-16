# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:14:02 2020
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as pyplot
import pickle
import math

def relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
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
  n = - relative_position
  if bidirectional:
    num_buckets //= 2
    ret += tf.dtypes.cast(tf.math.less(n,0),tf.int32) * num_buckets
    n = tf.math.abs(n)
  else:
    n = tf.math.maximum(n, 0)
  
  #now n is in range [0, inf]
  max_exact = num_buckets // 2
  is_small = tf.math.less(n, max_exact)
  value_if_large = max_exact + tf.dtypes.cast(
    tf.math.log(tf.dtypes.cast(n, tf.float32)/max_exact) / 
    math.log(max_distance/max_exact) *(num_buckets-max_exact),tf.int32)
  
  value_if_large = tf.math.minimum(value_if_large, num_buckets-1)
  ret += tf.where(is_small, n, value_if_large)
  return ret

def gen_relative_pos(seq_lent,directed=False):
  rel_mask_matrix = np.zeros((seq_lent,seq_lent),dtype=np.int32)
  for i in range(seq_lent):
    for j in range(seq_lent):
      if directed and j>i:
          continue
      
      rel_mask_matrix[i][j]=j-i
  
  t5_relative_pos_matrix = []
  for i in range(seq_lent):
    t5_relative_pos_matrix.append(t5_pos(rel_mask_matrix[i,:]))
  
  
  return rel_mask_matrix,np.array(t5_relative_pos_matrix,dtype=np.int32)


def f_positive(relative_position,alpha):
  #alpha = np.log(np.exp(alpha)+1)
  return 1-tf.exp(-alpha*relative_position)

def f_negative(relative_position,beta):
    return 1-tf.exp(beta*relative_position)


def main(_):
  seq_lent = 50
  
  relative_position, t5_bucket = gen_relative_pos(seq_lent)
   
  is_small = tf.cast(tf.math.less_equal(tf.constant(relative_position), 0),tf.float32)
  print(is_small)
  alpha = tf.compat.v1.get_variable('alpha', 
                           [32],
            initializer=tf.compat.v1.random_uniform_initializer(),
            trainable=True)
  A=1/seq_lent
  alpha = A*tf.math.softplus(alpha)
  
  bias = tf.compat.v1.get_variable('bias', 
                           [1],
            initializer=tf.compat.v1.random_uniform_initializer(),
            trainable=True)
  
  pos_bucket=0
  neg_bucket=0
  for i in range(8):
    #we need an mask to control the mask
    t1= tf.compat.v1.get_variable('bias_1'+str(i), 
                           [seq_lent,5],
           trainable=True)
    t2= tf.compat.v1.get_variable('bias_2'+str(i), 
                           [5,seq_lent],
           trainable=True)
    bias_1 = tf.matmul(t1,t2)
    #bias_1 = tf.compat.v1.get_variable('bias_1'+str(i), 
    #                       [seq_lent,seq_lent],
    #        trainable=True)
    
    #bias_2 = tf.compat.v1.get_variable('bias_2'+str(i), 
    #                       [seq_lent,seq_lent],
    #        trainable=True)
    
    pos_bucket += bias_1*f_positive(relative_position, alpha[i])
    
    neg_bucket+=bias_1*f_negative(relative_position, alpha[i])
  
  soft_t5_bucket = is_small*(neg_bucket+bias)+(1.0-is_small)*pos_bucket
  print(soft_t5_bucket)
  
  loss = tf.compat.v1.losses.mean_squared_error(tf.constant(t5_bucket,tf.float32), soft_t5_bucket)
  print(loss)
  
  
  sess = tf.Session()
  
  global_step = tf.Variable(0, name="global_step", trainable=False)
      
  optimizer = tf.train.AdamOptimizer(1e-3)
  grads_and_vars = optimizer.compute_gradients(loss)
  train_op = optimizer.apply_gradients(
    grads_and_vars, global_step=global_step)
      
  sess.run(tf.global_variables_initializer())
  
  for i in range(10000):
    _,loss_val,soft_t5_bucket_val =sess.run([train_op,loss,soft_t5_bucket,])
    if i%100==0:
      print('i:%d loss:%.4f', (i,loss_val))
  
  pyplot.plot(relative_position, t5_bucket)
  pyplot.plot(relative_position, soft_t5_bucket_val)
  pyplot.show()
  
if __name__ == '__main__':
  #tf.compat.v1.app.run()
  qlen=klen=128
  
  context_position = tf.range(qlen,dtype=tf.int32)[:,None]
  memory_postion = tf.range(klen, dtype=tf.int32)[None,:]
  relative_position = memory_postion - context_position
  
  ret_bucket = relative_position_bucket(relative_position,num_buckets=64,max_distance=256)
  sess=tf.Session()
  print(sess.run(relative_position))
  print(sess.run(ret_bucket[0]))
  print(sess.run(ret_bucket[-1]))