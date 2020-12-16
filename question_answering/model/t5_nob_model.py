# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 21:19:13 2020
"""

import tensorflow as tf
import math
from .layers import initializer, regularizer, residual_block, highway, conv, mask_logits, trilinear, total_params, optimized_trilinear_for_attention

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
  
  values = tf.expand_dims(values,2) #(qlen, klen, 1, num_heads) 
  
  values = tf.transpose(values,[2,3,0,1]) #(1, num_heads, qlen, klen)
  
  return values

class T5_Nob_Model(object):
  def __init__(self, config, batch, word_mat=None, char_mat=None, trainable=True, opt=True, demo = False, graph = None):
    self.config = config
    self.demo = demo
    self.graph = graph if graph is not None else tf.Graph()
    with self.graph.as_default():
      self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                            initializer=tf.constant_initializer(0),
                            trainable=False)
      self.dropout = tf.placeholder_with_default(0.0, (), name="dropout")
      if self.demo:
        self.c = tf.placeholder(tf.int32, [None, config.test_para_limit],"context")
        self.q = tf.placeholder(tf.int32, [None, config.test_ques_limit],"question")
        self.ch = tf.placeholder(tf.int32, [None, config.test_para_limit, config.char_limit],"context_char")
        self.qh = tf.placeholder(tf.int32, [None, config.test_ques_limit, config.char_limit],"question_char")
        self.y1 = tf.placeholder(tf.int32, [None, config.test_para_limit],"answer_index1")
        self.y2 = tf.placeholder(tf.int32, [None, config.test_para_limit],"answer_index2")
      else:
        self.c, self.q, self.ch, self.qh, self.y1, self.y2, self.qa_id = batch.get_next()
      
      # self.word_unk = tf.get_variable("word_unk", shape = [config.glove_dim], initializer=initializer())
      self.word_mat = tf.get_variable("word_mat", initializer=tf.constant(
        word_mat, dtype=tf.float32), trainable=False)
      self.char_mat = tf.get_variable(
        "char_mat", initializer=tf.constant(char_mat, dtype=tf.float32))
      
      self.c_mask = tf.cast(self.c, tf.bool)
      self.q_mask = tf.cast(self.q, tf.bool)
      self.c_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=1)
      self.q_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=1)
      
      if opt:
        N, CL = config.batch_size if not self.demo else 1, config.char_limit
        self.c_maxlen = tf.reduce_max(self.c_len)
        self.q_maxlen = tf.reduce_max(self.q_len)
        self.c = tf.slice(self.c, [0, 0], [N, self.c_maxlen])
        self.q = tf.slice(self.q, [0, 0], [N, self.q_maxlen])
        self.c_mask = tf.slice(self.c_mask, [0, 0], [N, self.c_maxlen])
        self.q_mask = tf.slice(self.q_mask, [0, 0], [N, self.q_maxlen])
        self.ch = tf.slice(self.ch, [0, 0, 0], [N, self.c_maxlen, CL])
        self.qh = tf.slice(self.qh, [0, 0, 0], [N, self.q_maxlen, CL])
        self.y1 = tf.slice(self.y1, [0, 0], [N, self.c_maxlen])
        self.y2 = tf.slice(self.y2, [0, 0], [N, self.c_maxlen])
      else:
        self.c_maxlen, self.q_maxlen = config.para_limit, config.ques_limit
        
      self.ch_len = tf.reshape(tf.reduce_sum(
        tf.cast(tf.cast(self.ch, tf.bool), tf.int32), axis=2), [-1])
      self.qh_len = tf.reshape(tf.reduce_sum(
        tf.cast(tf.cast(self.qh, tf.bool), tf.int32), axis=2), [-1])
      
      self.forward()
      total_params()
      
      if trainable:
        self.lr = tf.minimum(config.learning_rate, 0.001 / tf.log(999.) * tf.log(tf.cast(self.global_step, tf.float32) + 1))
        self.opt = tf.train.AdamOptimizer(learning_rate = self.lr, beta1 = 0.8, beta2 = 0.999, epsilon = 1e-7)
        grads = self.opt.compute_gradients(self.loss)
        gradients, variables = zip(*grads)
        capped_grads, _ = tf.clip_by_global_norm(
          gradients, config.grad_clip)
        self.train_op = self.opt.apply_gradients(
          zip(capped_grads, variables), global_step=self.global_step)
    
  def forward(self):
    config = self.config
    N, PL, QL, CL, d, dc, nh = config.batch_size if not self.demo else 1, self.c_maxlen, self.q_maxlen, config.char_limit, config.hidden, config.char_dim, config.num_heads
    
    with tf.variable_scope("Input_Embedding_Layer"):
      ch_emb = tf.reshape(tf.nn.embedding_lookup(
        self.char_mat, self.ch), [N * PL, CL, dc])
      qh_emb = tf.reshape(tf.nn.embedding_lookup(
        self.char_mat, self.qh), [N * QL, CL, dc])
      ch_emb = tf.nn.dropout(ch_emb, 1.0 - 0.5 * self.dropout)
      qh_emb = tf.nn.dropout(qh_emb, 1.0 - 0.5 * self.dropout)
      
      # Bidaf style conv-highway encoder
      ch_emb = conv(ch_emb, d,
        bias = True, activation = tf.nn.relu, 
        kernel_size = 5, name = "char_conv", reuse = None
      )
      
      qh_emb = conv(qh_emb, d,
          bias = True, activation = tf.nn.relu,
          kernel_size = 5, name = "char_conv", reuse = True
      )
      
      ch_emb = tf.reduce_max(ch_emb, axis = 1)
      qh_emb = tf.reduce_max(qh_emb, axis = 1)
      
      ch_emb = tf.reshape(ch_emb, [N, PL, ch_emb.shape[-1]])
      qh_emb = tf.reshape(qh_emb, [N, QL, ch_emb.shape[-1]])
      
      c_emb = tf.nn.dropout(tf.nn.embedding_lookup(self.word_mat, self.c), 1.0 - self.dropout)
      q_emb = tf.nn.dropout(tf.nn.embedding_lookup(self.word_mat, self.q), 1.0 - self.dropout)
      
      c_emb = tf.concat([c_emb, ch_emb], axis=2)
      q_emb = tf.concat([q_emb, qh_emb], axis=2)
      
      c_emb = highway(c_emb, size = d, scope = "highway", dropout = self.dropout, reuse = None)
      q_emb = highway(q_emb, size = d, scope = "highway", dropout = self.dropout, reuse = True)
    
    with tf.variable_scope("Embedding_Encoder_Layer"):
      #we need to add relative bias herein...
      #I do not know, wheather this will generate the best information...
      c_context_position = tf.range(config.para_limit,dtype=tf.int32)[:,None]
      c_memory_postion = tf.range(config.para_limit, dtype=tf.int32)[None,:]
      c_relative_position = c_memory_postion - c_context_position + config.para_limit
      
      self.c_t5_bias_mat = tf.get_variable('c_t5_bias_mat', 
                                                   [config.para_limit*2,nh],
                                                   initializer=tf.random_uniform_initializer())
      
      ## [batch, num_heads, query_length, memory_length]
      c_t5_bias = compute_bias(c_relative_position, self.c_t5_bias_mat) #(qlen, klen, 1, num_heads)
      print('c_t5_bias:',c_t5_bias)
      
      self.c_layer_weights,c = residual_block(c_emb,
            num_blocks = 1,
            num_conv_layers = 4,
            kernel_size = 7,
            mask = self.c_mask,
            num_filters = d,
            num_heads = nh,
            seq_len = self.c_len,
            scope = "Encoder_Residual_Block",
            bias = False,
            dropout = self.dropout,
            t5_bias=tf.slice(c_t5_bias, [0,0,0,0], [1,nh,self.c_maxlen, self.c_maxlen]))
      
      q_context_position = tf.range(config.ques_limit,dtype=tf.int32)[:,None]
      q_memory_postion = tf.range(config.ques_limit, dtype=tf.int32)[None,:]
      q_relative_position = q_memory_postion - q_context_position + config.ques_limit
      
      self.q_t5_bias_mat = tf.get_variable('q_t5_bias_mat', 
                                                   [config.ques_limit*2,nh],
                                                   initializer=tf.random_uniform_initializer())
      
      ## [batch, num_heads, query_length, memory_length]
      q_t5_bias = compute_bias(q_relative_position, self.q_t5_bias_mat) #(qlen, klen, 1, num_heads)
      
      self.q_layer_weights,q = residual_block(q_emb,
            num_blocks = 1,
            num_conv_layers = 4,
            kernel_size = 7,
            mask = self.q_mask,
            num_filters = d,
            num_heads = nh,
            seq_len = self.q_len,
            scope = "Encoder_Residual_Block",
            reuse = True, # Share the weights between passage and question
            bias = False,
            dropout = self.dropout,
            t5_bias=tf.slice(q_t5_bias, [0,0,0,0], [1,nh,self.q_maxlen, self.q_maxlen]))
    
    #we need to revise this into multiple head attention~~
    with tf.variable_scope("Context_to_Query_Attention_Layer"):
      # C = tf.tile(tf.expand_dims(c,2),[1,1,self.q_maxlen,1])
      # Q = tf.tile(tf.expand_dims(q,1),[1,self.c_maxlen,1,1])
      # S = trilinear([C, Q, C*Q], input_keep_prob = 1.0 - self.dropout)
      S = optimized_trilinear_for_attention([c, q], self.c_maxlen, self.q_maxlen, input_keep_prob = 1.0 - self.dropout)
      mask_q = tf.expand_dims(self.q_mask, 1)
      S_ = tf.nn.softmax(mask_logits(S, mask = mask_q))
      mask_c = tf.expand_dims(self.c_mask, 2)
      S_T = tf.transpose(tf.nn.softmax(mask_logits(S, mask = mask_c), dim = 1),(0,2,1))
      self.c2q = tf.matmul(S_, q)
      self.q2c = tf.matmul(tf.matmul(S_, S_T), c)
      attention_outputs = [c, self.c2q, c * self.c2q, c * self.q2c]
      
    with tf.variable_scope("Model_Encoder_Layer"):
      inputs = tf.concat(attention_outputs, axis = -1)
      self.enc = [conv(inputs, d, name = "input_projection")]
      self.model_c_t5_bias_mat=[None,None,None]
      self.model_c_layer_weights=[]
      for i in range(3):
        if i % 2 == 0: # dropout every 2 blocks
          self.enc[i] = tf.nn.dropout(self.enc[i], 1.0 - self.dropout)
        
        '''
        @we set different information for each layers...
        '''
        self.model_c_t5_bias_mat[i] = tf.get_variable('model_c_t5_bias_mat'+str(i), 
                      [config.para_limit*2*2,nh],
         initializer=tf.random_uniform_initializer())
        
        layer_weights,model_c =residual_block(self.enc[i],
                      num_blocks = 7,
                      num_conv_layers = 2,
                      kernel_size = 5,
                      mask = self.c_mask,
                      num_filters = d,
                      num_heads = nh,
                      seq_len = self.c_len,
                      scope = "Model_Encoder",
                      bias = False,
                      reuse = True if i > 0 else None,
                      dropout = self.dropout,
                      t5_bias=tf.slice(compute_bias(c_relative_position,
                                           self.model_c_t5_bias_mat[i]),
                                        [0,0,0,0], [1,nh,self.c_maxlen, self.c_maxlen]))
        
        self.model_c_layer_weights.append(layer_weights)
        self.enc.append(model_c)
    
    with tf.variable_scope("Output_Layer"):
      start_logits = tf.squeeze(conv(tf.concat([self.enc[1], self.enc[2]],axis = -1),1, bias = False, name = "start_pointer"),-1)
      end_logits = tf.squeeze(conv(tf.concat([self.enc[1], self.enc[3]],axis = -1),1, bias = False, name = "end_pointer"), -1)
      self.logits = [mask_logits(start_logits, mask = self.c_mask),
                     mask_logits(end_logits, mask = self.c_mask)]
      
      logits1, logits2 = [l for l in self.logits]
      
      outer = tf.matmul(tf.expand_dims(tf.nn.softmax(logits1), axis=2),
                        tf.expand_dims(tf.nn.softmax(logits2), axis=1))
      outer = tf.matrix_band_part(outer, 0, config.ans_limit)
      self.yp1 = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
      self.yp2 = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)
      losses = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits1, labels=self.y1)
      losses2 = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits2, labels=self.y2)
      self.loss = tf.reduce_mean(losses + losses2)
      
    if config.l2_norm is not None:
      variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
      l2_loss = tf.contrib.layers.apply_regularization(regularizer, variables)
      self.loss += l2_loss
    
    if config.decay is not None:
      self.var_ema = tf.train.ExponentialMovingAverage(config.decay)
      ema_op = self.var_ema.apply(tf.trainable_variables())
      with tf.control_dependencies([ema_op]):
        self.loss = tf.identity(self.loss)
        
        self.assign_vars = []
        for var in tf.global_variables():
          v = self.var_ema.average(var)
          if v:
            self.assign_vars.append(tf.assign(var,v))
  
  def get_loss(self):
    return self.loss
  
  def get_global_step(self):
    return self.global_step
