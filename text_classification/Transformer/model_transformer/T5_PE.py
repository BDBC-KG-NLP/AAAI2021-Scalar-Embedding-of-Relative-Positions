# -*- coding: utf-8 -*-


import tensorflow as tf
import numpy as np
import math
from scipy import linalg
from numpy.random import RandomState
rng = np.random.RandomState(23455)
import math
import modeling

regularizer = tf.contrib.layers.l2_regularizer(scale = 1e-3)

#no activation
initializer = lambda: tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                            mode='FAN_AVG',
                                            uniform=True,
                                          dtype=tf.float32)
#having activation
initializer_relu = lambda: tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                              mode='FAN_IN',
                                            uniform=False,
                                            dtype=tf.float32)

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


class Transformer(object):
  def __init__(
            self, bert_config,dataset,max_input_left,max_input_right,
            vocab_size,embeddings, embedding_size, batch_size, 
            l2_reg_lambda=0.0, is_Embedding_Needed=False, 
            trainable=False,
            overlap_needed=True, 
            position_needed=True, hidden_num=10, 
            extend_feature_dim=10,transformer_ret_pooling="last",
            t5_bucket=50,
            is_training=False):
    self.dataset = dataset
    self.config = bert_config
    self.embeddings=embeddings
    self.embedding_size = embedding_size
    self.overlap_needed = overlap_needed
    self.vocab_size = vocab_size
    self.is_training = is_training
    self.trainable=trainable
   
    self.position_needed = position_needed
    if self.overlap_needed:
      self.total_embedding_dim = embedding_size + extend_feature_dim
    else:
      self.total_embedding_dim = embedding_size
    if self.position_needed:
      self.total_embedding_dim = self.total_embedding_dim + extend_feature_dim
    self.batch_size = batch_size
    self.l2_reg_lambda = l2_reg_lambda
    self.para = []
    self.max_input_left = max_input_left
    self.max_input_right = max_input_right
    self.hidden_num = hidden_num
    self.extend_feature_dim = extend_feature_dim
    self.is_Embedding_Needed = is_Embedding_Needed
    self.transformer_ret_pooling=transformer_ret_pooling
    #we need to adjust the bucket number and max distance to achive the best performance...
    if 'adding_problem' in self.dataset or 'temporal_order' in self.dataset:
      self.t5_bucket= t5_bucket
      self.t5_max_distance = self.max_input_left
    else:
      self.t5_bucket = 32
      self.t5_max_distance=128
      
  def create_placeholder(self):
    if 'adding_problem' not in self.dataset:
      self.question = tf.placeholder(tf.int32,[None,self.max_input_left],name = 'input_question')
      self.input_y = tf.placeholder(tf.float32, [None,self.max_input_right], name = "input_y")
      input_mask = tf.cast(tf.cast(self.question,tf.bool),tf.int32)
      
      self.seq_lent = tf.reduce_sum(tf.cast(input_mask,tf.float32),-1,keepdims=True)
      
      self.attention_mask = modeling.create_attention_mask_from_input_mask(self.question,input_mask)
    else:
      self.question = tf.placeholder(tf.float32,[None,self.max_input_left,2],name = 'input_question')
      self.input_y = tf.placeholder(tf.float32, [None], name = "input_y")
      self.attention_mask = None
      self.seq_lent = self.max_input_left*1.0
    
    print('attention_mask:',self.attention_mask)
    
    self.q_position = tf.placeholder(tf.int32,[None,self.max_input_left],
                                     name = 'q_position')
    
    self.input_dropout_prob = tf.placeholder(tf.float32, name="input_dropout_prob")
     
    self.hidden_dropout_prob = tf.placeholder(tf.float32, name="hidden_dropout_prob")
    
    self.attention_probs_dropout_prob = tf.placeholder(tf.float32, name="attention_probs_dropout_prob")
    
  def add_embeddings(self):
    with tf.name_scope("embedding"):
      if self.is_Embedding_Needed:
        W = tf.Variable(np.array(self.embeddings),name="word_embed" ,dtype="float32",trainable = self.trainable )
      else:
        W=tf.get_variable(
          name='word_embed',
          shape=[self.vocab_size, self.embedding_size],
          initializer=modeling.create_initializer(0.02),trainable=True)
      
      if 'adding_problem' not in self.dataset:
        self.embedding_W = W
        self.embedded_chars_q = tf.nn.embedding_lookup(self.embedding_W,self.question)
      else:
        #mapping 2 dim into high dim
        if self.embedding_size == 2:
          self.embedded_chars_q = self.question
        else:
          self.embedded_chars_q =  tf.layers.dense(self.question,self.embedding_size)
      print('embedded_chars_q:',self.embedded_chars_q)
      
      if 'adding_problem' not in self.dataset:
        self.embedded_chars_q = modeling.layer_norm(
          tf.nn.dropout(self.embedded_chars_q,
                      keep_prob=1.0-self.input_dropout_prob))
    
    context_position = tf.range(self.max_input_left,dtype=tf.int32)[:,None]
    memory_postion = tf.range(self.max_input_left, dtype=tf.int32)[None,:]
    relative_position = memory_postion - context_position
      
    rp_bucket = relative_position_bucket(relative_position,
                        num_buckets=self.t5_bucket,
                        max_distance=self.t5_max_distance
                                          )
    
    #why this embedding is very sensitive...
    self.t5_pos_embedding = tf.get_variable('t5_pos_mat', 
                                    [self.t5_bucket,
                                     self.config.num_attention_heads],
                                    initializer=modeling.create_initializer(0.02),
                                    trainable=True
                                    )
      
    self.single_t5_att_bias = compute_bias(rp_bucket, self.t5_pos_embedding)
    ## [batch, num_heads, query_length, memory_length]
    self.t5_att_bias = tf.tile(self.single_t5_att_bias,
                                 [tf.shape(self.question)[0],1,1,1]) 
    print('t5_bias:',self.t5_att_bias)
    
    '''
    @2020/9/7 we can directly add the head mask during inference
    '''
    
    head_mask = np.zeros((self.config.num_attention_heads,
                            self.max_input_left,
                            self.max_input_left))
    #high2low=[3,1,4,0,5,2]
    low2high=[2,5,0,4,1,3]
    for tt in range(6):
      print('tt:',tt)
      head_mask[low2high[tt],:,:]=np.ones((self.max_input_left,self.max_input_left))
    
    self.t5_att_bias = self.t5_att_bias * tf.constant(head_mask,tf.float32)
    
  def feed_neural_work(self):
    '''
        input_tensor,
                      attention_mask=None,
                      hidden_size=768,
                      num_hidden_layers=12,
                      num_attention_heads=12,
                      intermediate_size=3072,
                      intermediate_act_fn=gelu,
                      hidden_dropout_prob=0.1,
                      attention_probs_dropout_prob=0.1,
                      initializer_range=0.02,
                      do_return_all_layers=False'''
    # `sequence_output` shape = [batch_size, seq_length, hidden_size].
    self.all_encoder_layers,self.context_bias = modeling.transformer_model(self.embedded_chars_q,
            attention_mask=self.attention_mask,
            hidden_size=self.config.hidden_size,
            num_hidden_layers=self.config.num_hidden_layers,
            num_attention_heads=self.config.num_attention_heads,
            intermediate_size=self.config.intermediate_size,
            intermediate_act_fn=modeling.get_activation(self.config.hidden_act),
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            initializer_range=self.config.initializer_range,
            do_return_all_layers=True,
            t5_relative_bias=self.t5_att_bias)
    self.sequence_output = self.all_encoder_layers[-1]
    # The "pooler" converts the encoded sequence tensor of shape
    # [batch_size, seq_length, hidden_size] to a tensor of shape
    # [batch_size, hidden_size]. This is necessary for segment-level
    # (or segment-pair-level) classification tasks where we need a fixed
    # dimensional representation of the segment.
    with tf.variable_scope("pooler"):
      # We "pool" the model by simply taking the hidden state corresponding
      # to the first token. We assume that this has been pre-trained
      
      if self.transformer_ret_pooling=="mean":
        print('self.seq_lent:',self.seq_lent)
        print('tf.reduce_sum(self.sequence_output,axis=1):',tf.reduce_sum(self.sequence_output,axis=1))
        
        self.pooled_output = tf.reduce_sum(self.sequence_output,
                              axis=1) * self.seq_lent
      elif self.transformer_ret_pooling=="last":
        self.pooled_output = self.sequence_output[:,-1,:]
      elif self.transformer_ret_pooling=="max":
        self.pooled_output = tf.reduce_max(self.sequence_output,
                                           axis=1)
      else:
        print('wrong transformer_ret_pooling:',self.transformer_ret_pooling)
        exit(0)
      
      if 'adding_problem' not in self.dataset:
        #we add dropout for pooled_output
        self.pooled_output = modeling.layer_norm(
          tf.nn.dropout(self.pooled_output,
                        keep_prob=1.0-self.input_dropout_prob))
    
    # Final (unnormalized) scores and predictions
    with tf.name_scope("output"):
      W = tf.get_variable(
                "W",
                shape=[self.config.hidden_size, self.max_input_right],
                initializer=initializer())
      b = tf.Variable(tf.constant(0.1, shape=[self.max_input_right]), name="b")
      l2_loss = tf.constant(0.0)
      l2_loss += tf.nn.l2_loss(W)
      self.scores = tf.nn.xw_plus_b(self.pooled_output, W, b, name="scores")
      print(self.scores)
      
      self.predictions = tf.argmax(self.scores, 1, name="predictions")
        
    if 'adding_problem' not in self.dataset:
      # Calculate mean cross-entropy loss
      with tf.name_scope("loss"):
        losses = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.scores, labels=self.input_y)
        self.l2_loss = l2_loss*self.l2_reg_lambda
        self.loss = tf.reduce_mean(losses) + self.l2_loss
      # Accuracy
      with tf.name_scope("accuracy"):
        correct_predictions = tf.equal(
                self.predictions, tf.argmax(self.input_y, 1))
        self.accuracy = tf.reduce_mean(
                tf.cast(correct_predictions, "float"), name="accuracy")
    else:
      with tf.name_scope("loss"):
        losses = tf.nn.l2_loss(
                  self.scores-tf.expand_dims(self.input_y,-1))
        print('losses:',losses)
        
        self.l2_loss = self.l2_reg_lambda * l2_loss
        self.loss = tf.reduce_mean(losses) + self.l2_loss*1e-3
          
      with tf.name_scope("accuracy"):
        correct_predictions = tf.less_equal(tf.abs(self.scores[:,0] - self.input_y),tf.constant([0.04]))
        self.accuracy = tf.reduce_mean(
                  tf.cast(correct_predictions, "float"), name="accuracy")
    
  def build_graph(self):
    self.create_placeholder()
    self.add_embeddings()
    self.feed_neural_work()
    
    