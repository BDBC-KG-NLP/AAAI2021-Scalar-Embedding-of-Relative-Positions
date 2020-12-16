# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 21:39:22 2020
"""
import tensorflow as tf
import numpy as np
from thumt.layers.nn import linear
import math

def create_initializer(initializer_range=0.02):
  """Creates a `truncated_normal_initializer` with the given range."""
  return tf.truncated_normal_initializer(stddev=initializer_range)

def polynomial_score(f,l1_width=100,l2_width=40,stddev=0.1,
                     dropout_prob=0.5,activation='relu',name='poly'):
  """
  Parameters
  ----------
  f : Tensor shape (from_seq, to_seq)
    soft buckets of relatative position.
  poly_order : TYPE, optional
    the order of polynomial. The default is 5.
  name : TYPE, optional
    variable scope. The default is 'poly'.
  
  Returns
  -------
  soft-t5 score: (from_seq, to_seq)
  """
  if activation=='relu':
    act_fn=tf.nn.relu
  elif activation=='sigmoid':
    act_fn=tf.nn.sigmoid
  else:
    act_fn=tf.nn.tanh
  
  with tf.variable_scope(name):
    #we change the activation function into relu to check wheather we can achieve the SOTA model...
    #initialzier into 0.01
    hidden = tf.layers.dense(tf.expand_dims(f,-1),
                             units=l1_width,
                        activation=act_fn,
                        kernel_initializer=create_initializer(stddev),
                    bias_initializer=tf.constant_initializer(stddev),
                    name='layer_1'
                    )
    h_drop=tf.nn.dropout(hidden,keep_prob=1.0-dropout_prob)
    
    hidden2 = tf.layers.dense(h_drop,
                              units=l2_width,
                        activation=act_fn,
                        kernel_initializer=create_initializer(stddev),
                     bias_initializer=tf.constant_initializer(stddev),
                     name='layer_2'
                    )
    
    g=tf.layers.dense(hidden2,
                      units=1,
                        activation=None,
                        kernel_initializer=create_initializer(stddev),
                     bias_initializer=tf.constant_initializer(stddev),
                     name='layer_3'
                    )
  return g[:,:,0]

def compute_bias(num_heads, qlen, klen,
                 relative_att_random_bucket_mat,
                 l1_width=100,
                 l2_width=40,
                 stddev=0.1,
                 dropout_prob=1.0,
                 activation='relu',
                 bidirectional=True,name='soft_t5'):
  
  context_position = tf.range(qlen,dtype=tf.int32)[:,None]
  memory_postion = tf.range(klen, dtype=tf.int32)[None,:]
  relative_position = memory_postion - context_position + qlen ##(qlen, klen)
  
  rp_bucket = tf.nn.embedding_lookup(relative_att_random_bucket_mat, relative_position) #(qlen, klen,num_heads)
  
  soft_t5_bias_list=[]
  for headi in range(num_heads):
    soft_t5_bias_list.append(polynomial_score(rp_bucket[:,:,headi],
                                              l1_width,l2_width,stddev,
                                              name='head-'+str(headi))
                             )
  
  soft_t5_bias = tf.stack(soft_t5_bias_list,axis=0) #(num_heads, qlen, klen)
  
  #soft_t5_bias: (1, num_heads, qlen, klen)
  soft_t5_bias=tf.expand_dims(soft_t5_bias,0)
  print('soft_t5_bias:',soft_t5_bias)
  return soft_t5_bias

def should_generate_summaries():
    """Is this an appropriate context to generate summaries.
    :returns: a boolean
    """
    if "while/" in tf.contrib.framework.get_name_scope():
        return False
    if tf.get_variable_scope().reuse:
        return False
    return True


def attention_image_summary(weights, rgb=True):
    """Compute attention image summary.
    :param weights: a Tensor with shape [batch, heads, queries, memories]
    :param rgb: use RGB color to represent a head
    """
    shape = tf.shape(weights)
    batch_size = shape[0]
    num_heads = shape[1]
    num_queries = shape[2]
    num_memories = shape[3]

    if rgb:
        # [batch, queries, memories, heads]
        image = tf.transpose(weights, [0, 2, 3, 1])
        # for high-dynamic-range
        image = tf.pow(image, 0.2)
        # Each head will correspond to one of RGB
        image = tf.pad(image, [[0, 0], [0, 0], [0, 0],
                               [0, tf.mod(-num_heads, 3)]])
        shape = tf.shape(image)
        # [batch, queries, memories, 3, heads]
        image = tf.reshape(image, [batch_size, num_queries, num_memories,
                                   3, shape[-1] // 3])
        image = tf.reduce_max(image, 4)
    else:
        image = tf.reshape(weights, [-1, num_queries, num_memories, 1])

    # [batch, height, width, channel]
    tf.summary.image("attention", image, max_outputs=1)
    
def multiplicative_attention(queries, keys, values, bias, keep_prob=None,
                             name=None, soft_t5_bias=None):
    """ Multiplicative attention mechanism. This layer is implemented using
        dot-product operation.

    :param queries: A tensor with shape [batch, heads, length_q, depth_q]
    :param keys: A tensor with shape [batch, heads, length_kv, depth_k]
    :param values: A tensor with shape [batch, heads, length_kv, depth_v]
    :param bias: A tensor
    :param keep_prob: a scalar in (0, 1]
    :param name: the name of this operation
    :param rpr: the name of this operation

    :returns: A dict with the following keys:
        weights: A tensor with shape [batch, heads, length_q, length_kv]
        outputs: A tensor with shape [batch, heads, length_q, depth_v]
    """

    with tf.name_scope(name, default_name="multiplicative_attention",
                       values=[queries, keys, values, bias]):
        q_shape = tf.shape(queries)
        bs, hd, lq, dk = q_shape[0], q_shape[1], q_shape[2], q_shape[3]
        lk = tf.shape(keys)[2]
        dv = tf.shape(values)[3]
        
        logits = tf.matmul(queries, keys, transpose_b=True)
        
        if soft_t5_bias is not None:
          logits += soft_t5_bias
        
        # shape: [batch, heads, length_q, length_kv]
        if bias is not None:
            logits += bias
        
        weights = tf.nn.softmax(logits, name="attention_weights")
        
        if keep_prob is not None and keep_prob < 1.0:
            weights = tf.nn.dropout(weights, keep_prob)
        
        outputs = tf.matmul(weights, values)  # bs, hd, lq, dv
        
        return {"weights": weights, "outputs": outputs}


def multihead_attention(queries, memories, bias, num_heads, key_size,
                        value_size, output_size, keep_prob=None, output=True,
                        state=None, summary=True, dtype=None, scope=None,
                        has_relative_attention_bias=False,
                        soft_t5_bias=None,
                        bidirectional=True
                        ):
    """ Multi-head scaled-dot-product attention with input/output
        transformations.

    :param queries: A tensor with shape [batch, length_q, depth_q]
    :param memories: A tensor with shape [batch, length_m, depth_m]
    :param bias: A tensor (see attention_bias)
    :param num_heads: An integer dividing key_size and value_size
    :param key_size: An integer
    :param value_size: An integer
    :param output_size: An integer
    :param keep_prob: A floating point number in (0, 1]
    :param output: Whether to use output transformation
    :param state: An optional dictionary used for incremental decoding
    :param summary: Use image summary
    :param dtype: An optional instance of tf.DType
    :param scope: An optional string
    :param max_relative_dis: An integer

    :returns: A dict with the following keys:
        weights: A tensor with shape [batch, heads, length_q, length_kv]
        outputs: A tensor with shape [batch, length_q, depth_v]
    """

    if key_size % num_heads != 0:
        raise ValueError("Key size (%d) must be divisible by the number of "
                         "attention heads (%d)." % (key_size, num_heads))

    if value_size % num_heads != 0:
        raise ValueError("Value size (%d) must be divisible by the number of "
                         "attention heads (%d)." % (value_size, num_heads))

    with tf.variable_scope(scope, default_name="multihead_attention",
                           values=[queries, memories], dtype=dtype):
        next_state = {}

        if memories is None:
            # self attention
            size = key_size * 2 + value_size
            combined = linear(queries, size, True, True, scope="qkv_transform")
            q, k, v = tf.split(combined, [key_size, key_size, value_size],
                               axis=-1)
            
            if state is not None:
                k = tf.concat([state["key"], k], axis=1)
                v = tf.concat([state["value"], v], axis=1)
                next_state["key"] = k
                next_state["value"] = v
        else:
            q = linear(queries, key_size, True, True, scope="q_transform")
            combined = linear(memories, key_size + value_size, True,
                              scope="kv_transform")
            k, v = tf.split(combined, [key_size, value_size], axis=-1)
        
        # split heads
        q = split_heads(q, num_heads)
        k = split_heads(k, num_heads)
        v = split_heads(v, num_heads)
        
        key_depth_per_head = key_size // num_heads
        q *= key_depth_per_head ** -0.5
            
        if has_relative_attention_bias:
            if state != None:
                soft_t5_bias = soft_t5_bias[:,:,-1:,:]
                
        else:
            soft_t5_bias=None
                                                  
        results = multiplicative_attention(q, k, v, bias, keep_prob,
                                           soft_t5_bias=soft_t5_bias)
        
        print('soft_t5_bias:',soft_t5_bias)
        
        # combine heads
        weights = results["weights"]
        x = combine_heads(results["outputs"])

        if output:
            outputs = linear(x, output_size, True, True,
                             scope="output_transform")
        else:
            outputs = x

        if should_generate_summaries() and summary:
            attention_image_summary(weights)

        outputs = {"weights": weights, "soft_t5_bias":soft_t5_bias,"outputs": outputs}

        if state is not None:
            outputs["state"] = next_state
        
        return outputs
      
      
def add_timing_signal(x, min_timescale=1.0, max_timescale=1.0e4, name=None):
    """
    This function adds a bunch of sinusoids of different frequencies to a
    Tensor. See paper: `Attention is all you need'

    :param x: A tensor with shape [batch, length, channels]
    :param min_timescale: A floating point number
    :param max_timescale: A floating point number
    :param name: An optional string

    :returns: a Tensor the same shape as x.
    """

    with tf.name_scope(name, default_name="add_timing_signal", values=[x]):
        length = tf.shape(x)[1]
        channels = tf.shape(x)[2]
        position = tf.to_float(tf.range(length))
        num_timescales = channels // 2

        log_timescale_increment = (
                math.log(float(max_timescale) / float(min_timescale)) /
                (tf.to_float(num_timescales) - 1)
        )
        inv_timescales = min_timescale * tf.exp(
            tf.to_float(tf.range(num_timescales)) * -log_timescale_increment
        )

        scaled_time = (tf.expand_dims(position, 1) *
                       tf.expand_dims(inv_timescales, 0))
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
        signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
        signal = tf.reshape(signal, [1, length, channels])

        return x + tf.cast(signal, x.dtype)


def split_heads(inputs, num_heads, name=None):
    """ Split heads
    :param inputs: A tensor with shape [batch, ..., channels]
    :param num_heads: An integer
    :param name: An optional string
    :returns: A tensor with shape [batch, heads, ..., channels / heads]
    """

    with tf.name_scope(name, default_name="split_heads", values=[inputs]):
        x = inputs
        n = num_heads
        old_shape = x.get_shape().dims
        ndims = x.shape.ndims

        last = old_shape[-1]
        new_shape = old_shape[:-1] + [n] + [last // n if last else None]
        ret = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [n, -1]], 0))
        ret.set_shape(new_shape)
        perm = [0, ndims - 1] + [i for i in range(1, ndims - 1)] + [ndims]
        return tf.transpose(ret, perm)


def combine_heads(inputs, name=None):
    """ Combine heads
    :param inputs: A tensor with shape [batch, heads, length, channels]
    :param name: An optional string
    :returns: A tensor with shape [batch, length, heads * channels]
    """

    with tf.name_scope(name, default_name="combine_heads", values=[inputs]):
        x = inputs
        x = tf.transpose(x, [0, 2, 1, 3])
        old_shape = x.get_shape().dims
        a, b = old_shape[-2:]
        new_shape = old_shape[:-2] + [a * b if a and b else None]
        x = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
        x.set_shape(new_shape)

        return x

