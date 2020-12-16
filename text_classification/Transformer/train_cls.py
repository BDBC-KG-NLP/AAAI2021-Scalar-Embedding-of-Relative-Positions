# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 20:14:48 2020

@author: wujs

@revise: 2020-1-20
@transformer: train_cls
"""
import tensorflow as tf
import numpy as np
import os
import time
import datetime
from utils import load_seq_data, yield_data,load_temporal_order_data, load_adding_problem_data,load_reber_data
from sklearn.metrics import accuracy_score
from model_transformer.T5_PE import Transformer as T5_PE
from model_transformer.T5_PE_NoB import Transformer as T5_PE_NoB
from model_transformer.TPE_reduce import Transformer as TPE_reduce
from model_transformer.Non_PE import Transformer as Non_PE
from model_transformer.Soft_T5_PE import Transformer as Soft_T5_PE
from model_transformer.Soft_T5_PE_NoB import Transformer as Soft_T5_PE_NoB
import pickle
import config
import modeling
import random

random.seed(1024)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
now = int(time.time())
timeArray = time.localtime(now)
timeStamp = time.strftime("%Y%m%d%H%M%S", timeArray)
timeDay = time.strftime("%Y%m%d", timeArray)
print (timeStamp)

FLAGS = config.flags.FLAGS
bert_config_file = 'config.json'
bert_config = modeling.BertConfig.from_json_file(bert_config_file)
bert_config.num_hidden_layers=FLAGS.num_hidden_layers
bert_config.num_attention_heads=FLAGS.num_attention_heads
#I do not know, how much is more apprioprate
bert_config.intermediate_size=FLAGS.embedding_dim*2

print(bert_config.num_hidden_layers)
print(bert_config.num_attention_heads)

def predict(sess, model, q_max_sent_length, test,idx2vocab):
  scores = []
  true_label = []
  for question,input_y,q_position,t5_position in yield_data(FLAGS.batch_size, q_max_sent_length, test):
    if 'adding_problem' not in FLAGS.data:
      question=np.array(question,np.int32)
    else:
      question=np.array(question,np.float32)
    feed_dict = {
      model.question: question,
      model.input_y: input_y,
      model.q_position: q_position,
      model.input_dropout_prob:0.0,
      model.hidden_dropout_prob: 0.0,
      model.attention_probs_dropout_prob:0.0
      }
    
    #if 'T5' in FLAGS.model_name:
    #  feed_dict[model.t5_position]=t5_position
    score,pred = sess.run([model.scores,model.predictions], feed_dict)
    if 'adding' not in FLAGS.data:
      scores.extend(score)
    else:
      score=list(score[:,0])
      scores.extend(score)
    true_label.extend(input_y)
    
    if FLAGS.is_training==False:
      context_bias=sess.run(model.context_bias,feed_dict=feed_dict)
      t5_att_bias=None
      if 'T5' in FLAGS.model_name:
        t5_att_bias = sess.run(model.single_t5_att_bias,feed_dict)
      
      #['0', '1', '0', '1', '1', '1', '0', '0', '1', '0', '0', '1', '0', '1', '0', '0', '0', '0', '1', '0', '0', '0', '0', '1', '0', '1', '1', '0', '0', '1', '1', '1', '1', '1', '0', '0', '0', '0', '1', '0', '0', '1', '1', '1', '1', '1', '1', '1', '0', '1']
      if FLAGS.data=='process_cls_50':
        strs = ['0', '0', '0', '1', '0', '0', '1', '1', '0', '1', '0', '0', '0', '0', '0', '0', '0', '1', '1', '0', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1', '1', '0', '1', '1', '1', '1', '0', '0', '0', '0', '0', '0', '0', '0', '1', '0', '0']
      elif FLAGS.data=='reber':
        ret_list='P B P T T T V P X T T V V E T B P T T T T T T V P S E P'.split(' ')[:-1]
        
        strs=ret_list+['PAD' for i in range(66-len(ret_list))]
        #print(strs)
      
      for qidx in range(len(score)):
        question_i = list(question[qidx])
        y_i = np.argmax(input_y[qidx])
        
        question_text = [idx2vocab[wid] for wid in question_i]
        #print(question_text)
        #if y_i != pred[qidx]:
        if question_text==strs:
          print('gold y:',y_i)
          print('pred y:',pred[qidx])
          print(question_text)
          print('--------------------')
          
          context_scores=context_bias[0][0]
          final_scores=context_bias[0][1]
          att_mask=context_bias[0][2]
          #print(context_scores)
          #print(final_scores)
          #print(att_mask)
          
          save_param={'question_text':question_text,
                      'input_y':y_i,
                      'predict_y':scores[qidx],
                      'context_scores':context_scores,
                      'final_scores':final_scores,
                      'att_mask':att_mask,
                      't5_att_bias':t5_att_bias
                      }
          pickle.dump(save_param,open('ret-3/'+FLAGS.model_name+'-'+FLAGS.data+'-att.pkl','wb'))
          
          
    #print(input_y)
    #print(score)
    #print('========================')
  return np.array(scores[:len(test)]), true_label

def main(_):
  now = int(time.time())
  timeArray = time.localtime(now)
  timeDay = time.strftime("%Y-%m-%d", timeArray)
  print (timeDay)
  max_acc_score = 0
  ret_test_acc = 0
  
  if 'adding_problem' in FLAGS.data:
    bert_config.attention_probs_dropout_prob=0.5
    bert_config.hidden_dropout_prob=0.5
    
  ckpt_path = 'model_save-3/'+FLAGS.data+'/'+ \
                      '_'.join([FLAGS.model_name,str(FLAGS.embedding_dim),
                              'L'+str(bert_config.num_hidden_layers),
                              'H'+ str(bert_config.num_attention_heads),
                              str(FLAGS.training_nums),
                              str(FLAGS.transformer_ret_pooling),
                              str(FLAGS.t5_bucket),
                              FLAGS.trail])
  if FLAGS.is_Embedding_Needed==False:
    ckpt_path += '-random'
  
  if FLAGS.is_Embedding_Needed==False:
    ckpt_path = ckpt_path+ '_l2-'+str(FLAGS.l2_reg_lambda)
  
  if FLAGS.batch_size!=64:
    ckpt_path = ckpt_path+ '_bz-'+str(FLAGS.batch_size)
        
  if str(FLAGS.learning_rate)!=str(0.0001):
    ckpt_path = ckpt_path+'_lr-'+str(FLAGS.learning_rate)
  
  if 'Soft_T5' in FLAGS.model_name:
    bert_config.bucket_slop_min=FLAGS.bucket_slop_min
    bert_config.bucket_slop_max=FLAGS.bucket_slop_max
    bert_config.l1_width=FLAGS.l1_width
    bert_config.l2_width=FLAGS.l2_width
    bert_config.stddev=FLAGS.stddev
        
    ckpt_path = ckpt_path+ '_'+str(bert_config.bucket_slop_min)
    ckpt_path = ckpt_path+ '_'+str(bert_config.bucket_slop_max)
    ckpt_path = ckpt_path+ '_'+str(bert_config.l1_width)
    ckpt_path = ckpt_path+ '_'+str(bert_config.l2_width)
    ckpt_path = ckpt_path+ '_'+str(bert_config.stddev)
  
  if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)
  
  writer = tf.summary.FileWriter(ckpt_path)
  
  data_file = ckpt_path + '/test'
  
  if FLAGS.is_training==True:
    precision = data_file + '_precise'
  else:
    precision = data_file + '_precise_test'
  ret_file = data_file+'_ret.p'
  print(data_file)
  if 'temporal_order' in FLAGS.data:
    (train_data,dev_data,test_data),vocab,q_max_sent_length, a_max_sent_length = load_temporal_order_data(FLAGS.data)
    embeddings= np.random.normal(scale=0.02,
                               size=(len(vocab),FLAGS.embedding_dim))
  elif FLAGS.data=='process_cls_50':
    (train_data,dev_data,test_data),vocab,q_max_sent_length, a_max_sent_length = load_seq_data(FLAGS.data)
    embeddings= np.random.normal(scale=0.02,
                               size=(len(vocab),FLAGS.embedding_dim))
  elif 'adding_problem' in FLAGS.data:
    (train_data,dev_data,test_data),vocab,q_max_sent_length, a_max_sent_length = load_adding_problem_data(FLAGS.data)
    vocab=[0]
    embeddings=np.array([[0],[1.0]],np.float32)
  elif 'reber' in FLAGS.data:
    (train_data,dev_data,test_data),vocab,q_max_sent_length, a_max_sent_length = load_reber_data(FLAGS.data)
    
    embeddings= np.random.normal(scale=0.02,
                               size=(len(vocab),FLAGS.embedding_dim))
  else:
    print('wrong dataset:',FLAGS.data)
  
  print(q_max_sent_length)
  
  idx2vocab = {vocab[key]:key for key in vocab}
  #2020-2-22
  #wujs
  #We want to solve this problem...
  #
  #random.shuffle(train_data)
  train_data = train_data[:(FLAGS.training_nums//2)] + train_data[10000:(10000+FLAGS.training_nums//2)]
  #train_data = random.Random(4).shuffle(train_data[:10000])[:(FLAGS.training_nums//2)] +random.Random(4).shuffle(train_data[10000:])[:(FLAGS.training_nums//2)] 
  
  with tf.Graph().as_default():
    session_conf = tf.compat.v1.ConfigProto()
    session_conf.allow_soft_placement = FLAGS.allow_soft_placement
    session_conf.log_device_placement = FLAGS.log_device_placement
    session_conf.gpu_options.allow_growth = True
    
    sess = tf.compat.v1.Session(config=session_conf)
    now = int(time.time())
    timeArray = time.localtime(now)
    timeStamp1 = time.strftime("%Y%m%d%H%M%S", timeArray)
    timeDay = time.strftime("%Y%m%d", timeArray)
    print (timeStamp1)
    final_ret = ''
    max_dev_acc_epoch=0
    with sess.as_default(), open(precision, "w") as log:
      model_dict = {'T5_PE':T5_PE,
                    'TPE_reduce':TPE_reduce,
                    'Non_PE':Non_PE,
                    'Soft_T5_PE':Soft_T5_PE,
                    'Soft_T5_PE_NoB':Soft_T5_PE_NoB,
                    'T5_PE_NoB':T5_PE_NoB}
      
      if 'concat' in FLAGS.model_name:
        bert_config.hidden_size = 2*FLAGS.embedding_dim
        bert_config.intermediate_size = bert_config.hidden_size*2
        hidden_num=2*FLAGS.embedding_dim
      else:
        bert_config.hidden_size = FLAGS.embedding_dim
        bert_config.intermediate_size = bert_config.hidden_size*2
        hidden_num=FLAGS.embedding_dim
      
      model = model_dict[FLAGS.model_name](
                dataset=FLAGS.data,
                max_input_left=q_max_sent_length,
                max_input_right=a_max_sent_length,
                vocab_size=len(vocab),
                batch_size=FLAGS.batch_size,
                embeddings=embeddings,
                embedding_size=FLAGS.embedding_dim,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                is_Embedding_Needed=True,
                trainable=FLAGS.trainable,
                hidden_num=hidden_num,
                extend_feature_dim=FLAGS.extend_feature_dim,
                bert_config = bert_config,
                transformer_ret_pooling=FLAGS.transformer_ret_pooling,
                t5_bucket=FLAGS.t5_bucket)
      
      model.build_graph()
      # Define Training procedure
      global_step = tf.Variable(0, name="global_step", trainable=False)
      optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
      grads_and_vars = optimizer.compute_gradients(model.loss)
      train_op = optimizer.apply_gradients(
          grads_and_vars, global_step=global_step)
      saver = tf.train.Saver()
      sess.run(tf.global_variables_initializer())
      
      all_trainable_vars = tf.reduce_sum([tf.reduce_prod(v.shape) for v in tf.trainable_variables()])
      
      sess.run(tf.global_variables_initializer())
      
      print('all_trainable_vars:',sess.run(all_trainable_vars))
      saver = tf.train.Saver(max_to_keep=5)
      
      if FLAGS.is_training==False:
        print("strating load", tf.train.latest_checkpoint(ckpt_path))
        saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))
        
        predicted, true_label = predict(
                          sess, model, q_max_sent_length,test_data,idx2vocab)
        predicted_label = np.argmax(predicted, 1)
        true_label = np.argmax(true_label,1)
        test_acc = accuracy_score(true_label,predicted_label)
        print('test_acc:',test_acc)
        #pickle.dump(test_acc,open(ret_file,'wb'))
        
      else:
        #if os.path.exists(ckpt_path+'/checkpoint'):
        #  print("strating load", tf.train.latest_checkpoint(ckpt_path))
        #  saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))
        iter_batch = 0
        for i in range(FLAGS.num_epochs):
          for question,input_y,q_position,t5_position in yield_data(FLAGS.batch_size, q_max_sent_length,train_data):
            iter_batch+=1
            if 'adding_problem' not in FLAGS.data:
              question=np.array(question,np.int32)
            else:
              question=np.array(question,np.float32)
      
            feed_dict = {
                          model.question: question,
                          model.input_y: input_y,
                          model.q_position: q_position,
                          model.input_dropout_prob: bert_config.input_dropout_prob,
            model.hidden_dropout_prob: bert_config.hidden_dropout_prob,
            model.attention_probs_dropout_prob:bert_config.attention_probs_dropout_prob
                    }
            
            #if 'T5' in FLAGS.model_name:
            #  feed_dict[model.t5_position]=t5_position
            
            _, step, loss, accuracy,scores = sess.run(
                          [train_op, global_step, model.loss, model.accuracy,model.scores], feed_dict)
            #if iter_batch%100==0:
            #  print('train scores:',scores[:,0])
            #  print('train input_y:',input_y)
            
            time_str = datetime.datetime.now().isoformat()
            now = int(time.time())
            timeArray = time.localtime(now)
            print("{}:, epoch {}, step {}, loss {:g}, acc {:g}  ".format(time_str,i, step, loss, accuracy))
            print('final_ret:', final_ret)
            curr_step = step
            
            loss_sum = tf.Summary(value=[tf.Summary.Value(
                      tag="model/loss", simple_value=loss), ])
            writer.add_summary(loss_sum, step)
          
            acc_sum = tf.Summary(value=[tf.Summary.Value(
                  tag="model/accuracy", simple_value=accuracy),])
            writer.add_summary(acc_sum, step)
            
            if FLAGS.model_name == 'Soft_T5_PE':
              soft_t5_alpha,soft_t5_beta = sess.run([model.soft_t5_alpha,
                                                     model.soft_t5_beta])
              params={'soft_t5_alpha':soft_t5_alpha,
                      'soft_t5_beta':soft_t5_beta}
              #pickle.dump(params, open(ckpt_path+'/soft_t5_ret.p','wb'))
              for head_idx in range(FLAGS.num_attention_heads):
                loss_sum = tf.Summary(value=[tf.Summary.Value(
                  tag="model/head_"+str(head_idx)+'_alpha', 
                  simple_value=soft_t5_alpha[head_idx]), ])
                writer.add_summary(loss_sum, step)
                
                acc_sum = tf.Summary(value=[tf.Summary.Value(
                  tag="model/head_"+str(head_idx)+'_beta', 
                  simple_value=soft_t5_beta[head_idx]),])
                writer.add_summary(acc_sum, step)
          
          if True:
            if True:
              dev_predicted, dev_true_label = predict(
                        sess, model, q_max_sent_length,dev_data,idx2vocab)
              if 'adding_problem' not in FLAGS.data:
                dev_predicted_label = np.argmax(dev_predicted, 1)
                dev_true_label = np.argmax(dev_true_label,1)
                dev_acc = accuracy_score(dev_true_label,dev_predicted_label)
              else:
                correct_prediction = np.less_equal(np.abs( np.array(dev_predicted) - np.array(dev_true_label)),np.array([0.04]))
                dev_acc = np.mean(correct_prediction.astype(np.float32))
              
              dev_acc_sum = tf.Summary(value=[tf.Summary.Value(
                    tag="dev/accuracy", simple_value=dev_acc),])
              writer.add_summary(dev_acc_sum, step)
            
              log.write('['+str(i)+']: '+'iter_batch:'+str(iter_batch)+'\n')
              log.write(str(max_acc_score)+'\t'+str(dev_acc)+'\t'+str(ret_test_acc))
              final_ret = str(max_acc_score)+'\t'+str(dev_acc)+'\t'+str(ret_test_acc)
              log.write("\n")
              log.flush()
              
              if dev_acc > max_acc_score:
                max_acc_score = dev_acc
                
                predicted, true_label = predict(
                          sess, model, q_max_sent_length,test_data,idx2vocab)
                if 'adding_problem' not in FLAGS.data:
                  predicted_label = np.argmax(predicted, 1)
                  true_label = np.argmax(true_label,1)
                  test_acc = accuracy_score(true_label,predicted_label)
                else:
                  correct_prediction = np.less_equal( np.abs(np.array(predicted) - np.array(true_label)),np.array([0.04]))
                  test_acc = np.mean(correct_prediction.astype(np.float32))
                
                ret_test_acc=test_acc
                pickle.dump(ret_test_acc,open(ret_file,'wb'))
                save_path = os.path.join(
                  ckpt_path, "model_{}.ckpt".format(curr_step))
                saver.save(sess, save_path,write_meta_graph=True)
                
                if 'T5_PE' in FLAGS.model_name:
                  t5_att_bias = sess.run(model.single_t5_att_bias,feed_dict)
                  param={'right':t5_att_bias[0,:,0,:],
                   'left':t5_att_bias[0,:,-1,:],
                   't5_att_bias':t5_att_bias}
            
                  pickle.dump(param, open(
                    ckpt_path+'/'+'t5_att_bias.p','wb'))
                
                if 'Soft_T5_PE' == FLAGS.model_name:
                  soft_t5_alpha,soft_t5_beta = sess.run([model.soft_t5_alpha,
                                                   model.soft_t5_beta])
                  params={'soft_t5_alpha':soft_t5_alpha,
                          'soft_t5_beta':soft_t5_beta}
                  pickle.dump(params, open(
                      ckpt_path+'/'+'params.p','wb'))
                
                if 'Soft_T5_PE_NoB' == FLAGS.model_name:
                  soft_t5_rd_bucket_mat_val=sess.run(model.soft_t5_rd_bucket_mat)
                  params={'soft_t5_rd_bucket_mat':soft_t5_rd_bucket_mat_val}
                  pickle.dump(params, open(
                      ckpt_path+'/'+'params.p','wb'))
                
                
        print('final_ret:', final_ret)
        log.close()
        pickle.dump(ret_test_acc,open(ret_file,'wb'))
        
if __name__ == '__main__':
  tf.compat.v1.app.run()