# -*- coding: utf-8 -*-


import tensorflow as tf
import numpy as np
import math
from scipy import linalg
from numpy.random import RandomState
rng = np.random.RandomState(23455)
import math
import modeling

class Transformer(object):
  def __init__(
            self, bert_config,dataset,
            max_input_left,max_input_right,vocab_size,embeddings,
            embedding_size, batch_size, l2_reg_lambda=0.0,
            is_Embedding_Needed=False,
            trainable=False,
            overlap_needed=True, 
            position_needed=True, hidden_num=10,
            extend_feature_dim=10,transformer_ret_pooling="last",
            t5_bucket=50,
            is_training=False):
    self.config = bert_config
    self.dataset = dataset
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
    
  def get_timing_signal_1d(self,length, channels, min_timescale=1.0, max_timescale=1.0e4):
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
  
  def add_embeddings(self):
    with tf.name_scope("embedding"):
      if self.is_Embedding_Needed:
        W = tf.Variable(np.array(self.embeddings),name="word_embed" ,dtype="float32",trainable = self.trainable )
      else:
        W=tf.get_variable(
          name='word_embed',
          shape=[self.vocab_size, self.embedding_size],
          initializer=modeling.create_initializer(0.02),trainable=True)
        
      self.embedding_W = W
      
      self.embedded_chars_q_pos= self.get_timing_signal_1d(self.max_input_left, self.embedding_size)
      
      if 'adding_problem' not in self.dataset:
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
      
      #add the position embedding may be lead to poor performance...
      self.embedded_chars_q= self.embedded_chars_q +self.embedded_chars_q_pos
      
      print('embedded_chars_q:',self.embedded_chars_q)
      
      
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
            )
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
                initializer=tf.contrib.layers.xavier_initializer())
      b = tf.Variable(tf.constant(0.1, shape=[self.max_input_right]), name="b")
      self.l2_loss = tf.constant(0.0)
      self.l2_loss += tf.nn.l2_loss(W)
      self.l2_loss += tf.nn.l2_loss(b)
      self.scores = tf.nn.xw_plus_b(self.pooled_output, W, b, name="scores")
      self.predictions = tf.argmax(self.scores, 1, name="predictions")
        
    if 'adding_problem' not in self.dataset:
      # Calculate mean cross-entropy loss
      with tf.name_scope("loss"):
        losses = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.scores, labels=self.input_y)
        self.loss = tf.reduce_mean(losses)# + self.l2_reg_lambda * self.l2_loss
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
        self.loss = tf.reduce_mean(losses) #+ self.l2_reg_lambda * self.l2_loss
          
      with tf.name_scope("accuracy"):
        correct_predictions = tf.less_equal(tf.abs(self.scores[:,0] - self.input_y),tf.constant([0.04]))
        self.accuracy = tf.reduce_mean(
                  tf.cast(correct_predictions, "float"), name="accuracy")
  
  def build_graph(self):
    self.create_placeholder()
    self.add_embeddings()
    self.feed_neural_work()