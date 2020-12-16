# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 21:37:31 2020
"""
import tensorflow as tf
from thumt.layers.nn import linear
import math

def get_t5_relative_position(qlen,klen):
  context_position = tf.range(qlen,dtype=tf.int32)[:,None]
  memory_postion = tf.range(klen, dtype=tf.int32)[None,:]
  relative_position = memory_postion - context_position
  
  return relative_position

#we directly generate the t5 relative
#we need to fine the task...
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

def compute_bias( rp_bucket, relative_attention_bias):
  '''compute binned relative postion bias'''
  
  values = tf.nn.embedding_lookup(relative_attention_bias, rp_bucket) #(qlen, klen, num_heads)
  values = tf.expand_dims(values,0) #(1, qlen, klen, num_heads) 
  
  values = tf.transpose(values,[0,3,1,2]) #(1, num_heads, qlen, klen)
  
  return values

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
                             name=None, t5_bias=None):
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
        
        if t5_bias!=None:
            print('t5_bias:', t5_bias)
            logits = logits + t5_bias
            
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
                        has_relative_attention_bias=True,
                        t5_bias=None,
                        bidirectional=True):
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
        
        #for t5 we do not scale the score..
        key_depth_per_head = key_size // num_heads
        q *= key_depth_per_head ** -0.5
        
        if has_relative_attention_bias:
            #Maybe there are something wrong herein~
            if state!=None:
                t5_bias = t5_bias[:,:,-1:,:]
        else:
            t5_bias=None
          
        results = multiplicative_attention(q, k, v, bias, keep_prob, 
                                           t5_bias=t5_bias)
        
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

        outputs = {"weights": weights, "outputs": outputs}
        
        
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


