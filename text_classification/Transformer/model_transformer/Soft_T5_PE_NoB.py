# -*- coding: utf-8 -*-


import tensorflow as tf
import numpy as np
import math
from scipy import linalg
from numpy.random import RandomState
rng = np.random.RandomState(23455)
import math
import modeling

#no activation
initializer = lambda: tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                            mode='FAN_AVG',
                                            uniform=True,
                                          dtype=tf.float32)
def polynomial_score(f,l1_width=100,l2_width=1,stddev=0.02,dropout_prob=0.5,name='poly'):
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
  #the original is: 100,40, stddev=0.1
  #now we need to reduce the parameters
  """
  w1 = tf.Variable(tf.truncated_normal([1,l1_width],stddev=stddev))
  b1 = tf.Variable(tf.constant(stddev,shape=[l1_width]))
  
  w2 = tf.Variable(tf.truncated_normal([l1_width,l2_width],stddev=stddev))
  b2 = tf.Variable(tf.constant(stddev,shape=[l2_width]))
  
  #w3 = tf.Variable(tf.truncated_normal([l2_width,1],stddev=stddev))
  #b3 = tf.Variable(tf.constant(stddev,shape=[1]))
  
  hidden =tf.nn.relu(tf.einsum('aij,jk->aik',tf.expand_dims(f,-1),w1)+b1)
  
  #we still contain the relu activation...
  h_drop=tf.nn.dropout(hidden,keep_prob=1.0-dropout_prob)
  g = tf.nn.relu(tf.einsum('aij,jk->aik',h_drop,w2)+b2)
  
  #hidden2 =tf.nn.relu(tf.einsum('aij,jk->aik',h_drop,w2)+b2)
  
  #g=tf.einsum('aij,jk->aik',hidden2,w3)+b3
  
  return g[:,:,0]

def compute_bias(num_heads, qlen, klen,
                 relative_att_random_bucket_mat,
                 l1_width=100,l2_width=1,stddev=0.02,
                 bidirectional=True,name='layer'):
  """
  
  Parameters
  ----------
  num_heads : integer
    head nums.
  qlen : TYPE
    query length
  klen : TYPE
    key length(qlen==klen).
  relative_att_random_bucket_mat : #shape (qlen*2, num_heads)
    random bucket mat.
  bidirectional : TYPE, optional
    DESCRIPTION. The default is True.
  name : TYPE, optional
    DESCRIPTION. The default is 'layer'.

  Returns
  -------
  soft_t5_bias : TYPE
    DESCRIPTION.

  """
  context_position = tf.range(qlen,dtype=tf.int32)[:,None]
  memory_postion = tf.range(klen, dtype=tf.int32)[None,:]
  relative_position = memory_postion - context_position + qlen ##(qlen, klen)
  
  rp_bucket = tf.nn.embedding_lookup(relative_att_random_bucket_mat, relative_position) #(qlen, klen,num_heads)
  
  soft_t5_bias_list=[]
  for headi in range(num_heads):
    soft_t5_bias_list.append(polynomial_score(rp_bucket[:,:,headi],
                                              l1_width=l1_width,
                                              l2_width=l2_width,
                                              stddev=stddev))
  
  soft_t5_bias = tf.stack(soft_t5_bias_list,axis=0) #(num_heads, qlen, klen)
  
  #soft_t5_bias: (1, num_heads, qlen, klen)
  soft_t5_bias=tf.expand_dims(soft_t5_bias,0)
  print('soft_t5_bias:',soft_t5_bias)
  return soft_t5_bias

class Transformer(object):
  def __init__(
            self, bert_config,dataset,max_input_left,max_input_right,
            vocab_size,embeddings, embedding_size, batch_size, 
            l2_reg_lambda=0.0, is_Embedding_Needed=False,
            trainable=False,
            overlap_needed=True, position_needed=True, hidden_num=10, 
            extend_feature_dim=10,
            transformer_ret_pooling="last",
            t5_bucket=5,is_training=False):
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
    
  def create_placeholder(self):
    if 'adding_problem' not in self.dataset:
      self.question = tf.placeholder(tf.int32,[None,self.max_input_left],name = 'input_question')
      self.input_y = tf.placeholder(tf.float32, [None,self.max_input_right], name = "input_y")
      input_mask = tf.cast(tf.cast(self.question,tf.bool),tf.int32)
      self.attention_mask = modeling.create_attention_mask_from_input_mask(self.question,input_mask)
      self.seq_lent = tf.reduce_sum(tf.cast(input_mask,tf.float32),-1,keepdims=True)
      
    else:
      self.question = tf.placeholder(tf.float32,[None,self.max_input_left,2],name = 'input_question')
      self.input_y = tf.placeholder(tf.float32, [None], name = "input_y")
      self.attention_mask = None
      self.seq_lent = self.max_input_left*1.0
      
    print('attention_mask:',self.attention_mask)
    
    self.q_position = tf.placeholder(tf.int32,[None,self.max_input_left],name = 'q_position')
    
    self.input_dropout_prob = tf.placeholder(tf.float32, 
                                             name="input_dropout_prob")
    
    self.hidden_dropout_prob = tf.placeholder(tf.float32, name="hidden_dropout_prob")
    
    self.attention_probs_dropout_prob = tf.placeholder(tf.float32, name="attention_probs_dropout_prob")
    
  def add_embeddings(self):
    with tf.name_scope("embedding"):
      if self.is_Embedding_Needed:
        W = tf.get_variable(name="embeddings" ,dtype="float32",
                            initializer=np.array(self.embeddings,np.float32),
                            trainable = self.trainable )
      else:
        #I think we need to utilize more fine-grained word embedding~
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
      
      #(0,1)
      self.soft_t5_rd_bucket_mat = tf.sigmoid(tf.get_variable('t5_rd_bucket_mat', 
                                    [2*self.max_input_left,
                                     self.config.num_attention_heads],
                                    initializer=modeling.create_initializer(0.1),
                                    trainable=True
                                    ))
       
      self.single_t5_att_bias = compute_bias(self.config.num_attention_heads,
              self.max_input_left,self.max_input_left,
              self.soft_t5_rd_bucket_mat,
              l1_width=self.config.l1_width,
              l2_width=self.config.l2_width,
              stddev=self.config.stddev,
              bidirectional=True
              )
      
      self.t5_att_bias  = tf.tile(self.single_t5_att_bias,
                        [tf.shape(self.question)[0],1,1,1],name='t5_att_bias')
      print('[!!!--t5_bias:]',self.t5_att_bias)
      
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
    
    with tf.variable_scope("pooler"):
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
        print('wrong transformer_ret_pooling:',
              self.transformer_ret_pooling)
        exit(0)
      
      #we add dropout for pooled_output
      if 'adding_problem' not in self.dataset:
        self.pooled_output = modeling.layer_norm(
            tf.nn.dropout(self.pooled_output,
                          keep_prob=1.0-self.input_dropout_prob))
    
    # Final (unnormalized) scores and predictions
    with tf.name_scope("output"):
      W = tf.get_variable(
                "W",
                shape=[self.config.hidden_size,
                       self.max_input_right],
                initializer=initializer(),
                )
      b = tf.Variable(tf.constant(0.1, shape=[self.max_input_right]), name="b")
      l2_loss = tf.constant(0.0)
      l2_loss += tf.nn.l2_loss(W)
      
      self.scores = tf.nn.xw_plus_b(self.pooled_output, W, b, name="scores")
      self.predictions = tf.argmax(self.scores, 1, name="predictions")
    
    
    if 'adding_problem' not in self.dataset:
      # Calculate mean cross-entropy loss
      with tf.name_scope("loss"):
        self.l2_loss = self.l2_reg_lambda * l2_loss
        losses = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.scores, labels=self.input_y)
        
        self.loss = tf.reduce_mean(losses) #+ self.l2_loss
      # Accuracy
      with tf.name_scope("accuracy"):
        correct_predictions = tf.equal(
                self.predictions, tf.argmax(self.input_y, 1))
        self.accuracy = tf.reduce_mean(
                tf.cast(correct_predictions, "float"), name="accuracy")
    else:
      with tf.name_scope("loss"):
        self.l2_loss = self.l2_reg_lambda * l2_loss
        losses = tf.nn.l2_loss(
                  self.scores-tf.expand_dims(self.input_y,-1))
        print('losses:',losses)
        self.loss = tf.reduce_mean(losses)  #+ self.l2_loss
      
      with tf.name_scope("accuracy"):
        correct_predictions = tf.less_equal(tf.abs(self.scores[:,0] - self.input_y),tf.constant([0.04]))
        self.accuracy = tf.reduce_mean(
                  tf.cast(correct_predictions, "float"), name="accuracy")
  def build_graph(self):
    self.create_placeholder()
    self.add_embeddings()
    self.feed_neural_work()