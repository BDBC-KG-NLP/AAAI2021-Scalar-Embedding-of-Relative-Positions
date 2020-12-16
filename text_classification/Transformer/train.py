# coding=utf-8
#! /usr/bin/env python3.4
import tensorflow as tf
import numpy as np
import os
import time
import datetime
from data_helper import batch_gen_with_point_wise, load, load_trec_sst2, prepare, batch_gen_with_single
import operator
from model_transformer.T5_PE import Transformer as T5_PE
from model_transformer.T5_PE_NoB import Transformer as T5_PE_NoB
from model_transformer.TPE_reduce import Transformer as TPE_reduce
from model_transformer.Non_PE import Transformer as Non_PE
from model_transformer.Soft_T5_PE import Transformer as Soft_T5_PE
from model_transformer.Soft_T5_PE_NoB import Transformer as Soft_T5_PE_NoB
import modeling
import optimization
import random
from sklearn.metrics import accuracy_score
import pickle
import config
from functools import wraps
from utils import unpack_params


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

acc_flod=[]


def log_time_delta(func):
  @wraps(func)
  def _deco(*args, **kwargs):
    start = time.time()
    ret = func(*args, **kwargs)
    end = time.time()
    delta = end - start
    print("%s runed %.2f seconds" % (func.__name__, delta))
    return ret
  return _deco

def predict(sess, model, dev, alphabet, batch_size, q_len):
  
  scores = []
  losses = []
  for data in batch_gen_with_single(dev, alphabet, batch_size, q_len):
    if len(data[0])==0:
      print(data)
      print('non data for predict')
      continue
    feed_dict = {
      model.question: data[0],
      model.input_y: data[1],
      model.q_position: data[2],
      model.input_dropout_prob:0.0,
      model.hidden_dropout_prob: 0.0,
      model.attention_probs_dropout_prob:0.0
      }
    
    score, loss = sess.run([model.scores, model.loss],feed_dict)
    scores.extend(score)
    losses.append(loss)
    
    
    if FLAGS.is_training==False:
      idx2vocab = {alphabet[key]:key for key in alphabet}
      
      context_bias, pred=sess.run([model.context_bias,
                                   model.predictions],feed_dict=feed_dict)
      t5_att_bias=None
      if 'T5' in FLAGS.model_name:
        t5_att_bias = sess.run(model.single_t5_att_bias,feed_dict)
      
      #['0', '1', '0', '1', '1', '1', '0', '0', '1', '0', '0', '1', '0', '1', '0', '0', '0', '0', '1', '0', '0', '0', '0', '1', '0', '1', '1', '0', '0', '1', '1', '1', '1', '1', '0', '0', '0', '0', '1', '0', '0', '1', '1', '1', '1', '1', '1', '1', '0', '1']
    
      strs=None
      if FLAGS.data=='sst2':
        strs=['there', 'are', 'slow', 'and', 'repetitive', 'parts', ',', 'but', 'it', 'has', 'just', 'enough', 'spice', 'to', 'keep', 'it', 'interesting', '.', '[UNKNOW]', '[UNKNOW]', '[UNKNOW]', '[UNKNOW]', '[UNKNOW]', '[UNKNOW]', '[UNKNOW]', '[UNKNOW]', '[UNKNOW]', '[UNKNOW]', '[UNKNOW]', '[UNKNOW]', '[UNKNOW]', '[UNKNOW]', '[UNKNOW]', '[UNKNOW]', '[UNKNOW]', '[UNKNOW]', '[UNKNOW]', '[UNKNOW]', '[UNKNOW]', '[UNKNOW]', '[UNKNOW]', '[UNKNOW]', '[UNKNOW]', '[UNKNOW]', '[UNKNOW]', '[UNKNOW]', '[UNKNOW]', '[UNKNOW]', '[UNKNOW]', '[UNKNOW]', '[UNKNOW]', '[UNKNOW]', 'END']
      question=data[0]
      input_y=data[1]
      
      for qidx in range(len(score)):
        question_i = list(question[qidx])
        y_i = np.argmax(input_y[qidx])
        
        question_text = [idx2vocab[wid] for wid in question_i]
        #print(question_text)
        #if y_i != pred[qidx]:
        
        if question_text==strs:
          print('qidx:',qidx)
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
                      'context_scores':context_scores[qidx],
                      'final_scores':final_scores[qidx],
                      'att_mask':att_mask[qidx],
                      't5_att_bias':t5_att_bias
                      }
          pickle.dump(save_param,open('ret-3/'+FLAGS.model_name+'-'+FLAGS.data+'-att.pkl','wb'))
  return np.array(scores[:len(dev)]), np.average(losses)

@log_time_delta
def dev_point_wise():
  if FLAGS.data=='TREC' or FLAGS.data=='sst2' or FLAGS.data=='aclImdb' or FLAGS.data=='sst5':
    train,dev,test=load_trec_sst2(FLAGS.data)
  else:
    train, dev = load(FLAGS.data)
  
  #we add the end of the sentence..
  q_max_sent_length = max(
    map(lambda x: len(x), train['question'].str.split())) + 1
  print(q_max_sent_length)
  print(len(train))
  print ('train question unique:{}'.format(len(train['question'].unique())))
  print ('train length', len(train))
  num_train_steps = len(train['question'].unique()) // FLAGS.batch_size * FLAGS.num_epochs
  
  print ('dev length', len(dev))
  if FLAGS.data=='TREC' or FLAGS.data=='sst2' or FLAGS.data=='aclImdb' or FLAGS.data=='sst5':
    alphabet,embeddings = prepare(FLAGS.data,[train, dev,test], max_sent_length=q_max_sent_length, dim=FLAGS.embedding_dim, is_embedding_needed=True, fresh=False)
  else:
    alphabet,embeddings = prepare(FLAGS.data,[train, dev], max_sent_length=q_max_sent_length, dim=FLAGS.embedding_dim, is_embedding_needed=True, fresh=False)
  print ('alphabet:', len(alphabet))
  print('word embeddings:',np.shape(embeddings))
  
  if FLAGS.data =='TREC':
    max_input_right = 6
  elif FLAGS.data=='sst5':
    max_input_right = 5
  else:
    max_input_right = 2
      
  model_dict = {'T5_PE':T5_PE,
                    'TPE_reduce':TPE_reduce,
                    'Non_PE':Non_PE,
                    'Soft_T5_PE':Soft_T5_PE,
                    'Soft_T5_PE_NoB':Soft_T5_PE_NoB,
                    'T5_PE_NoB':T5_PE_NoB
                  }
      
  ckpt_path = 'model_save-3/'+FLAGS.data+'_'+ \
                      '_'.join([FLAGS.model_name,str(FLAGS.embedding_dim),
                          'L'+str(bert_config.num_hidden_layers),
                          'H'+ str(bert_config.num_attention_heads),
                          str(FLAGS.training_nums),
                          str(FLAGS.transformer_ret_pooling),
                          FLAGS.trail,
                          str(FLAGS.t5_bucket),
                          str(FLAGS.learning_rate)])
  
  if FLAGS.is_Embedding_Needed==False:
    ckpt_path += '_random'
  
  if FLAGS.batch_size!=64:
    ckpt_path = ckpt_path+ '_bz-'+str(FLAGS.batch_size)
  
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
    
  if 'concat' in FLAGS.model_name:
    bert_config.hidden_size = 2*FLAGS.embedding_dim
    bert_config.intermediate_size = bert_config.hidden_size*2
    hidden_num=2*FLAGS.embedding_dim
  else:
    bert_config.hidden_size = FLAGS.embedding_dim
    bert_config.intermediate_size = bert_config.hidden_size*2
    hidden_num=FLAGS.embedding_dim
        
  
  data_file = ckpt_path + '/test'
  
  if FLAGS.is_training==True:
    precision = data_file + '_precise'
  else:
    precision = data_file + '_precise_test'
  
  with tf.Graph().as_default():
    session_conf = tf.ConfigProto()
    session_conf.allow_soft_placement = FLAGS.allow_soft_placement
    session_conf.log_device_placement = FLAGS.log_device_placement
    session_conf.gpu_options.allow_growth = True
    
    sess = tf.Session(config=session_conf)
    now = int(time.time())
    timeArray = time.localtime(now)
    timeStamp1 = time.strftime("%Y%m%d%H%M%S", timeArray)
    timeDay = time.strftime("%Y%m%d", timeArray)
    print (timeStamp1)
    
    with sess.as_default(), open(precision, "w") as log:
      log.write(str(FLAGS.__flags) + '\n')
      
      writer = tf.summary.FileWriter(ckpt_path)
      
      #model instantiation
      model = model_dict[FLAGS.model_name](
        dataset=FLAGS.data,
        max_input_left=q_max_sent_length,
        max_input_right=max_input_right,
        vocab_size=len(alphabet),
        embeddings=embeddings,
        embedding_size=FLAGS.embedding_dim,
        batch_size=FLAGS.batch_size,
        l2_reg_lambda=1.0,
        is_Embedding_Needed=FLAGS.is_Embedding_Needed,
        hidden_num=hidden_num,
        extend_feature_dim=FLAGS.extend_feature_dim,
        bert_config = bert_config,
        transformer_ret_pooling=FLAGS.transformer_ret_pooling,
        t5_bucket=FLAGS.t5_bucket,
        is_training=True)
      
      model.build_graph()
      
      global_step = tf.Variable(0, name="global_step", trainable=False)
      optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
      grads_and_vars = optimizer.compute_gradients(model.loss)
      train_op = optimizer.apply_gradients(
          grads_and_vars, global_step=global_step)
      saver = tf.train.Saver()
      sess.run(tf.global_variables_initializer())
      
      all_trainable_vars = tf.reduce_sum([tf.reduce_prod(v.shape) for v in tf.trainable_variables()])
      
      sess.run(tf.global_variables_initializer())
      print('tf.trainable_variables():',tf.trainable_variables())
      print('all_trainable_vars:',sess.run(all_trainable_vars))
      saver = tf.train.Saver(max_to_keep=5)
      
      if FLAGS.is_training==False:
        print("strating load", tf.train.latest_checkpoint(ckpt_path))
        saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))
        
        predicted_test,_ = predict(sess, model, test, alphabet, 
                                   FLAGS.batch_size, q_max_sent_length)
        predicted_label = np.argmax(predicted_test, 1)
        test_acc= accuracy_score(test['flag'], 
                              predicted_label)
        print('test_acc:',test_acc)
        #pickle.dump(test_acc,open(ret_file,'wb'))
        exit(0)
      else:
        acc_max, loss_min = 0.0000, 100000
        acc_test=0.000
        early_stop=20
        patience=0
        early_stop_flags=False
        train_flag='Full'
        if FLAGS.training_nums!=20000:
          train_flag='Small'
        
        idx=0
        for i in range(FLAGS.num_epochs):
          datas = batch_gen_with_point_wise(
            train, alphabet, FLAGS.batch_size, q_len=q_max_sent_length,flag=train_flag)
          
          for data in datas:
            idx+=1
            feed_dict = {
              model.question: data[0],
              model.input_y: data[1],
              model.q_position: data[2],
              model.input_dropout_prob: bert_config.input_dropout_prob,
              model.hidden_dropout_prob: bert_config.hidden_dropout_prob,
              model.attention_probs_dropout_prob:bert_config.attention_probs_dropout_prob
              }
              
            _, step, l2_loss,loss, accuracy = sess.run(
              [train_op, global_step, model.l2_loss,model.loss, model.accuracy], feed_dict)
            time_str = datetime.datetime.now().isoformat()
            
            loss_sum = tf.Summary(value=[tf.Summary.Value(
              tag="model/loss", simple_value=loss), ])
            writer.add_summary(loss_sum, step)
            
            acc_sum = tf.Summary(value=[tf.Summary.Value(
              tag="model/acc uracy", simple_value=accuracy),])
            writer.add_summary(acc_sum, step)
            
            if idx%50==0:
              print("{}: step {}, l2_loss {:g}, loss {:g}, acc {:g}  ".format(time_str, step,l2_loss, loss, accuracy))
          
          if True:
            if True:
              predicted_dev,dev_loss = predict(
                sess, model, dev, alphabet, FLAGS.batch_size, q_max_sent_length)
              
              predicted_label = np.argmax(predicted_dev, 1)
              print(predicted_label[:10])
              print(dev['flag'][:10])
              acc_dev= accuracy_score(
                    dev['flag'], predicted_label)
                  
              dev_acc_sum = tf.Summary(value=[tf.Summary.Value(
                              tag="dev/accuracy", simple_value=acc_dev),])
              writer.add_summary(dev_acc_sum, step)
              
              curr_step = step
              if acc_dev> acc_max:# and dev_loss < loss_min:
                save_path = os.path.join(
                  ckpt_path, "model_{}.ckpt".format(curr_step))
                
                saver.save(sess, save_path, write_meta_graph=True,write_state=True)
                
                acc_max = acc_dev
                loss_min = dev_loss
                
                if FLAGS.data in ['sst2','TREC','sst5','aclImdb']:
                  predicted_test,_ = predict(
                      sess, model, test, alphabet, FLAGS.batch_size, q_max_sent_length)
                  predicted_label = np.argmax(predicted_test, 1)
                  acc_test= accuracy_score(
                              test['flag'], predicted_label)
                  
                  #save the test performance at the best dev score...
                  pickle.dump(acc_test,open(ckpt_path+'/test_ret.p','wb'))
                
                if 'T5_PE' in FLAGS.model_name:
                  t5_att_bias = sess.run(model.single_t5_att_bias,feed_dict)
                  param={'right':t5_att_bias[0,:,0],
                         'left':t5_att_bias[0,:,-1]}
                  
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
                patience=0
              else:
                patience+=1
                if patience > early_stop:
                  early_stop_flags=True
                  break
              
              print ("{}:dev epoch:loss {}".format(i, loss_min))
              print ("{}:dev epoch:acc {}".format(i, acc_max))
              if FLAGS.data in ['sst2','TREC','sst5','aclImdb']:
                print ("{}:test epoch:acc {}".format(i,acc_test))
              line2 = " {}:test epoch: acc{}".format(i, acc_test)
              log.write(line2 + '\n')
              log.flush()
            if early_stop_flags:
              break
        
        acc_flod.append(acc_max)
      log.close()
      
if __name__ == '__main__':
  for name, value in FLAGS.__flags.items():
    value=value.value
    print(name, value)
  
  ckpt_path = 'model_save-3/'+FLAGS.data+'_'+ \
                      '_'.join([FLAGS.model_name,str(FLAGS.embedding_dim),
                              'L'+str(bert_config.num_hidden_layers),
                              'H'+ str(bert_config.num_attention_heads),
                              str(FLAGS.training_nums),
                              str(FLAGS.transformer_ret_pooling),
                              FLAGS.trail,
                              str(FLAGS.learning_rate)])
  if FLAGS.is_Embedding_Needed==False:
    ckpt_path += '_random'
  
  if FLAGS.batch_size!=64:
    ckpt_path = ckpt_path+ '_bz-'+str(FLAGS.batch_size)
  
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
  
  if FLAGS.data=='TREC' or FLAGS.data=='sst2' or FLAGS.data=='aclImdb' or FLAGS.data=='sst5':
    dev_point_wise()
  else:
    for i in range(1,FLAGS.n_fold+1):
      print("{} cross validation ".format(i))
      for name, value in FLAGS.__flags.items():
        value=value.value
        print(name, value)
      
      dev_point_wise()
    print("the average acc {}".format(np.mean(acc_flod)))
    acc_test= np.mean(acc_flod)
    pickle.dump(acc_test,open(ckpt_path+'/test_ret.p','wb'))