# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 21:20:16 2020

@revise: 2020/July/20

"""

import tensorflow as tf
from .layers import initializer, regularizer, residual_block, highway, conv, mask_logits, trilinear, total_params, optimized_trilinear_for_attention
import numpy as np

def create_initializer(initializer_range=0.02):
  """Creates a `truncated_normal_initializer` with the given range."""
  return tf.truncated_normal_initializer(stddev=initializer_range)

def dropout(input_tensor, dropout_prob):
  """Perform dropout.

  Args:
    input_tensor: float Tensor.
    dropout_prob: Python float. The probability of dropping out a value (NOT of
      *keeping* a dimension as in `tf.nn.dropout`).

  Returns:
    A version of `input_tensor` with dropout applied.
  """
  if dropout_prob is None or dropout_prob == 0.0:
    return input_tensor

  output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
  return output

def layer_norm(input_tensor, name=None):
  """Run layer normalization on the last dimension of the tensor."""
  return tf.contrib.layers.layer_norm(
      inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)

def f_positive(relative_position, alpha):
  return 1.0-tf.math.exp(-alpha*relative_position)

def f_negative(relative_position, beta):
  return 1.0-tf.math.exp(beta*relative_position)


def polynomial_score(f,l1_width,l2_width,stddev,dropout_prob=0.5,activation='relu',name='poly'):
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
    '''
    hidden = tf.layers.dense(tf.expand_dims(f,-1),
                             units=100,
                        activation=act_fn,
                        kernel_initializer=create_initializer(0.1),
                    bias_initializer=tf.constant_initializer(0.1),
                    name='layer_1'
                    )
    h_drop=tf.nn.dropout(hidden,keep_prob=1.0-dropout_prob)
    
    hidden2 = tf.layers.dense(h_drop,
                              units=40,
                        activation=act_fn,
                        kernel_initializer=create_initializer(0.1),
                     bias_initializer=tf.constant_initializer(0.1),
                     name='layer_2'
                    )
    
    g=tf.layers.dense(hidden2,
                      units=1,
                        activation=None,
                        kernel_initializer=create_initializer(0.1),
                     bias_initializer=tf.constant_initializer(0.1),
                     name='layer_3'
                    )
    '''
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

def relative_position_soft_bucket(num_heads,
                                  qlen,klen,
                                  relative_position,
                                  alpha,beta,
                                  l1_width=100,
                                  l2_width=40,
                                  stddev=0.1,
                                  dropout_prob=0.5,
                                  activation='relu',
                                  bidirectional=True,
                                  name='layer'):
  
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
  n =  relative_position
  
  if bidirectional:
   n = n
  else:
    n = tf.math.minimum(n, 0)
  
  #now n is in range [0, inf]
  is_small = tf.cast(tf.math.less_equal(n, 0),tf.float32)
  
  positive_soft_bucket = f_positive(relative_position,
                              alpha)
    
  negative_soft_bucket = f_negative(relative_position,
                                    beta)
  
  #herein we need 8 polynomial function
  value_if_positive=[]
  value_if_negative=[]
  for i in range(num_heads):
    value_if_positive.append(polynomial_score(positive_soft_bucket[i,:,:],
                                              l1_width=l1_width,
                                              l2_width=l2_width,
                                              stddev=stddev,
                                              dropout_prob=dropout_prob,
                                              activation=activation,
                                            name=name+'_poly_pos_h-'+str(i)),
                           )
  
    value_if_negative.append(polynomial_score(negative_soft_bucket[i,:,:],
                                              l1_width=l1_width,
                                              l2_width=l2_width,
                                              stddev=stddev,
                                              dropout_prob=dropout_prob,
                                              activation=activation,
                                       name=name+'_poly_neg_h-'+str(i))
                             )
  
  value_if_positive = tf.stack(value_if_positive,axis=0)
  value_if_negative = tf.stack(value_if_negative,axis=0)
  
  ret = is_small*value_if_negative+(1.0-is_small)*value_if_positive
  
  #(None, num_heads, qlen, klen)
  return tf.expand_dims(ret,0)

def get_clip(num_heads,A,config,name='layer'):
  alpha =  tf.compat.v1.get_variable(name+'_alpha', 
                           [num_heads],
            initializer=tf.compat.v1.random_uniform_initializer(
                                              minval=config.bucket_slop_min, 
                                              maxval=config.bucket_slop_max),
                                              trainable=True)
  
  beta =  tf.get_variable(name+'_beta', 
                    [num_heads],
              initializer=tf.compat.v1.random_uniform_initializer(
                                              minval=config.bucket_slop_min, 
                                              maxval=config.bucket_slop_max),
                                              trainable=True)
  print(A)
  print(alpha)
  
  #herein
  alpha =A*tf.nn.relu(alpha)
  beta = A*tf.nn.relu(beta)
  
  
  return alpha, beta


def compute_bias(num_heads, qlen, klen, alpha, beta,l1_width,
                                              l2_width,
                                              stddev,dropout_prob,
                 activation='relu',
                 bidirectional=True,name='layer'):
  context_position = tf.range(qlen,dtype=tf.int32)[:,None]
  memory_postion = tf.range(klen, dtype=tf.int32)[None,:]
  relative_position = memory_postion - context_position
  
  #shape: (num_heads, qlen, klen)
  relative_position = tf.tile(tf.expand_dims(relative_position,0),
                              [num_heads,1,1])
  
  alpha = tf.tile(tf.expand_dims(tf.expand_dims(alpha,-1),-1),
                  [1,qlen,klen])
  
  beta = tf.tile(tf.expand_dims(tf.expand_dims(beta,-1),-1),
                  [1,qlen,klen])
  
  soft_t5_bias = relative_position_soft_bucket(
    num_heads,qlen,klen,
    relative_position,
    alpha,beta,
    l1_width,
    l2_width,
    stddev,
    dropout_prob,
    activation,
    bidirectional,
    name=name)
  
  #soft_t5_bias: (1, num_heads, qlen, klen)
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
  
class Soft_T5_Model(object):
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
    PRTIN_ATT=8
    
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
    
    #we utilize the maximum length to control the upper bound of dataset...
    #we assume all the head using the same bucketing methods, but just learn to update the attention parts...
    self.soft_t5_alpha, self.soft_t5_beta = get_clip(
                                               nh,
                            A=1/config.fixed_c_maxlen,
                            config=config,
                            name='layer_c')
      
    with tf.variable_scope("Embedding_Encoder_Layer"):
      self.c_t5_bias = compute_bias(nh,
                                    config.para_limit,
                                    config.para_limit,
                                    self.soft_t5_alpha,
                                    self.soft_t5_beta,
                                    l1_width=config.l1_width,
                                    l2_width=config.l2_width,
                                    stddev=config.stddev,
                                    dropout_prob=self.dropout,
                                    activation=config.soft_t5_activation,
                                    bidirectional=True,
                                    name='layer_c')
      print('[!!!-c_t5_bias:]',self.c_t5_bias)
      '''
      @we add head mask for c_t5_bias
      '''
      head_mask = np.zeros((nh,config.para_limit,config.para_limit))
      #hidx=[7,2,3,0,6,1,4,5]
      low2high=[5,4,1,6,0,3,2,7]
      for tt in range(PRTIN_ATT):
        head_mask[low2high[tt],:,:]=np.ones((config.para_limit,config.para_limit))
      
      self.c_t5_bias=self.c_t5_bias*head_mask
      
      self.c_layer_weights,c = residual_block(c_emb,
            num_blocks=1,
            num_conv_layers = 4,
            kernel_size = 7,
            mask = self.c_mask,
            num_filters = d,
            num_heads = nh,
            seq_len = self.c_len,
            scope = "Encoder_Residual_Block",
            bias = False,
            dropout = self.dropout,
            t5_bias=self.c_t5_bias[:,:,:self.c_maxlen,:self.c_maxlen])
      
      self.q_t5_bias = compute_bias(nh,
                                    config.ques_limit,
                                    config.ques_limit,
                                    self.soft_t5_alpha,
                                    self.soft_t5_beta,
                                    l1_width=config.l1_width,
                                    l2_width=config.l2_width,
                                    stddev=config.stddev,
                                    dropout_prob=self.dropout,
                                    activation=config.soft_t5_activation,
                                    bidirectional=True,
                                    name='layer_q')
      print('[!!!-q_t5_bias:]',self.q_t5_bias)
      
      head_mask = np.zeros((nh,config.ques_limit,config.ques_limit))
      #hidx=[7,0,6,2,4,1,3,5]
      low2high=[5,3,1,4,2,6,0,7]
      for tt in range(PRTIN_ATT):
        head_mask[low2high[tt],:,:]=np.ones((config.ques_limit,config.ques_limit))
      
      self.q_t5_bias=self.q_t5_bias*head_mask
      
      #num_blocks = 1,
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
            t5_bias=self.q_t5_bias[:,:,:self.q_maxlen,:self.q_maxlen])
      
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
    
    self.model_c_t5_bias_list=[]
    self.model_c_layer_weights=[]
    
    '''
    hidx_list = [[2,7,4,3,1,0,6,5],
                 [5,3,0,2,1,6,4,7],
                 [5,0,1,6,3,2,7,4]]'''
    
    hidx_list = [[5,6,0,1,3,4,7,2],
                  [7,4,6,1,2,0,3,5],
                  [4,7,2,3,6,1,0,5]]
    
    with tf.variable_scope("Model_Encoder_Layer"):
      inputs = tf.concat(attention_outputs, axis = -1)
      self.enc = [conv(inputs, d, name = "input_projection")]
      for i in range(3):
        if i % 2 == 0: # dropout every 2 blocks
          self.enc[i] = tf.nn.dropout(self.enc[i], 1.0 - self.dropout)
        
        c_t5_bias = compute_bias(nh,
                                 config.para_limit,
                                 config.para_limit,
                                 self.soft_t5_alpha,
                                 self.soft_t5_beta,
                                 l1_width=config.l1_width,
                                 l2_width=config.l2_width,
                                 stddev=config.stddev,
                                 dropout_prob=self.dropout,
                                 activation=config.soft_t5_activation,
                                 bidirectional=True,
                                 name='model_layer_'+str(i))
        head_mask = np.zeros((nh,config.para_limit,config.para_limit))
        for tt in range(PRTIN_ATT):
          head_mask[hidx_list[i][tt],:,:]=np.ones((config.para_limit,config.para_limit))
        
        c_t5_bias=c_t5_bias*head_mask
      
        self.model_c_t5_bias_list.append(c_t5_bias)
        print('[!!!-c_t5_bias:]', c_t5_bias)
        
        
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
                      t5_bias=self.model_c_t5_bias_list[i][:,:,:self.c_maxlen,:self.c_maxlen]
                      )
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
      print('self.loss:',self.loss)
      print('self.yp1:',self.yp1)
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