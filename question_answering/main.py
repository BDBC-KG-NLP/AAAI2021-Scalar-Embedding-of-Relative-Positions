import tensorflow as tf
import ujson as json
import numpy as np
from tqdm import tqdm
import os
import pickle
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

'''
This file is taken and modified from R-Net by HKUST-KnowComp
https://github.com/HKUST-KnowComp/R-Net
'''

from model import Raw_Model, TPE_Model, T5_Model, T5_Nob_Model, Soft_T5_Model,Soft_T5_NoB_Model
from demo import Demo
from util import get_record_parser, convert_tokens, evaluate, get_batch_dataset, get_dataset

def total_params():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    print("Total number of trainable parameters: {}".format(total_parameters))
    
def train(config):
  with open(config.word_emb_file, "r") as fh:
    word_mat = np.array(json.load(fh), dtype=np.float32)
  with open(config.char_emb_file, "r") as fh:
    char_mat = np.array(json.load(fh), dtype=np.float32)
  with open(config.train_eval_file, "r") as fh:
    train_eval_file = json.load(fh)
  with open(config.dev_eval_file, "r") as fh:
    dev_eval_file = json.load(fh)
  with open(config.dev_meta, "r") as fh:
    meta = json.load(fh)
    
  dev_total = meta["total"]
  print("Building model...")
  parser = get_record_parser(config)
  graph = tf.Graph()
  with graph.as_default() as g:
    train_dataset = get_batch_dataset(config.train_record_file, parser, config)
    dev_dataset = get_dataset(config.dev_record_file, parser, config)
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
      handle, train_dataset.output_types, train_dataset.output_shapes)
    train_iterator = train_dataset.make_one_shot_iterator()
    dev_iterator = dev_dataset.make_one_shot_iterator()
    
    model_dict={'Raw':Raw_Model,
                'TPE':TPE_Model,
                'T5':T5_Model,
                'T5_Nob':T5_Nob_Model,
                'Soft_T5':Soft_T5_Model,
                'Soft_T5_Nob':Soft_T5_NoB_Model
              }
    if config.model not in model_dict:
      print('wrong %s model name' %(config.model))
      exit(0)
    
    model = model_dict[config.model](config, iterator, word_mat, char_mat, graph = g)
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True
    
    loss_save = 100.0
    patience = 0
    best_f1 = 0.
    best_em = 0.
    
    with tf.Session(config=sess_config) as sess:
      writer = tf.summary.FileWriter(config.event_log_dir)
      sess.run(tf.global_variables_initializer())
      saver = tf.train.Saver()
      total_params()
      train_handle = sess.run(train_iterator.string_handle())
      dev_handle = sess.run(dev_iterator.string_handle())
      
      
      if os.path.exists(os.path.join(config.save_dir, "checkpoint")): #restore the model...
        saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))
      
      global_step = max(sess.run(model.global_step), 1) #to restore the global_step
      
      for _ in tqdm(range(global_step, config.num_steps + 1)):  #the max optimize steps
        global_step = sess.run(model.global_step) + 1
        if config.model in ['Soft_T5','Soft_Multi_T5']:
          loss, train_op,soft_t5_alpha= sess.run([model.loss, model.train_op,
                                   model.soft_t5_alpha,
                                   ], feed_dict={
          handle: train_handle, model.dropout: config.dropout})
        else:
          loss, train_op= sess.run([model.loss, model.train_op,
                                   ], feed_dict={
          handle: train_handle, model.dropout: config.dropout})
        
        if global_step % config.period == 0:
          loss_sum = tf.Summary(value=[tf.Summary.Value(
            tag="model/loss", simple_value=loss), ])
          writer.add_summary(loss_sum, global_step)
          
          if config.model in ['Soft_T5','Soft_Multi_T5']:
            for hidx in range(config.num_heads):
              hidx_val = soft_t5_alpha[hidx]*config.fixed_c_maxlen
              writer.add_summary(tf.Summary(
                      value=[tf.Summary.Value(
                      tag="model/alpha_"+str(hidx), 
                      simple_value=hidx_val),]),
                      global_step)
          
        
        if global_step % config.checkpoint == 0:
          _, summ = evaluate_batch(
            model, config.val_num_batches, train_eval_file, sess, "train", 
            handle, train_handle,config)
          
          for s in summ:
            writer.add_summary(s, global_step)
            
          metrics, summ = evaluate_batch(
            model, dev_total // config.batch_size + 1, 
            dev_eval_file, sess, "dev", handle, dev_handle,config)
          
          dev_f1 = metrics["f1"]  #early stop strategy ...
          dev_em = metrics["exact_match"]
          if dev_f1 < best_f1 and dev_em < best_em:
            patience += 1
            if patience > config.early_stop:
              break
          else:
            patience = 0
            best_em = max(best_em, dev_em)
            best_f1 = max(best_f1, dev_f1)
            
            #we save the best performance on evaluation...
            filename = os.path.join(
            config.save_dir, "model_{}.ckpt".format(global_step))
            
            saver.save(sess, filename)
            
            if config.model in ['T5','T5_TPE','T5_Nob']:
              c_t5_bias_mat,q_t5_bias_mat,model_c_t5_bias_mat = sess.run([
              model.c_t5_bias_mat,
              model.q_t5_bias_mat,
              model.model_c_t5_bias_mat])
              pkl_param = {'c_t5_bias_mat':c_t5_bias_mat,
                         'q_t5_bias_mat':q_t5_bias_mat,
                         'model_c_t5_bias_mat':model_c_t5_bias_mat}
              ret_filename = os.path.join(
            config.save_dir, 'att_ret.p')
              
              pickle.dump(pkl_param,open(ret_filename,'wb'))
            
            if config.model in ['Soft_T5','Soft_T5_TPE']:
              pkl_param = sess.run([[model.soft_t5_alpha,
                                     model.soft_t5_beta,
                                     model.c_t5_bias,
                                     model.q_t5_bias,
                        model.model_c_t5_bias_list]],feed_dict={
                handle: train_handle, model.dropout: config.dropout})
                          
              ret_filename = os.path.join(config.save_dir, 'att_ret.p')
              
              pickle.dump(pkl_param,open(ret_filename,'wb'))
            if config.model in ['Soft_T5_Nob']:
              pkl_param = sess.run([[model.c_soft_t5_rd_bucket_mat,
                                     model.q_soft_t5_rd_bucket_mat,
                                     model.model_soft_t5_rd_bucket_list,
                                     model.c_t5_bias,
                                     model.q_t5_bias,
                        model.model_c_t5_bias_list]],feed_dict={
                handle: train_handle, model.dropout: config.dropout})
                          
              ret_filename = os.path.join(config.save_dir, 'att_ret.p')
              
              pickle.dump(pkl_param,open(ret_filename,'wb'))
          for s in summ:
            writer.add_summary(s, global_step)
          writer.flush()

def evaluate_batch(model, num_batches, eval_file, sess, data_type, handle, str_handle,config):
  answer_dict = {}
  losses = []
  for _ in tqdm(range(1, num_batches + 1)):
    qa_id, loss, yp1, yp2 = sess.run(
      [model.qa_id, model.loss, model.yp1, model.yp2], feed_dict={handle: str_handle})
    
    answer_dict_, _ = convert_tokens(
      eval_file, qa_id.tolist(), yp1.tolist(), yp2.tolist())
    answer_dict.update(answer_dict_)
    losses.append(loss)
    
  loss = np.mean(losses)
  
  metrics = evaluate(eval_file, answer_dict)
  
  metrics["loss"] = loss
  loss_sum = tf.Summary(value=[tf.Summary.Value(
    tag="{}/loss".format(data_type), simple_value=metrics["loss"]), ])
  f1_sum = tf.Summary(value=[tf.Summary.Value(
    tag="{}/f1".format(data_type), simple_value=metrics["f1"]), ])
  em_sum = tf.Summary(value=[tf.Summary.Value(
    tag="{}/em".format(data_type), simple_value=metrics["exact_match"]), ])
  return metrics, [loss_sum, f1_sum, em_sum]


def demo(config):
  with open(config.word_emb_file, "r") as fh:
    word_mat = np.array(json.load(fh), dtype=np.float32)
  with open(config.char_emb_file, "r") as fh:
    char_mat = np.array(json.load(fh), dtype=np.float32)
  with open(config.test_meta, "r") as fh:
    meta = json.load(fh)
  model_dict={'Raw':Raw_Model,
              'TPE':TPE_Model,
              'T5':T5_Model,
              'T5_Nob':T5_Nob_Model,
              'Soft_T5':Soft_T5_Model,
              'Soft_T5_Nob':Soft_T5_NoB_Model
            }
  if config.model not in model_dict:
    print('wrong %s model name' %(config.model))
    exit(0)
    
  model = model_dict[config.model](config, None, word_mat, char_mat, trainable=False, demo = True)
  demo = Demo(model, config)


def test(config):
  with open(config.word_emb_file, "r") as fh:
    word_mat = np.array(json.load(fh), dtype=np.float32)
  with open(config.char_emb_file, "r") as fh:
    char_mat = np.array(json.load(fh), dtype=np.float32)
  with open(config.test_eval_file, "r") as fh:
    eval_file = json.load(fh)
  with open(config.test_meta, "r") as fh:
    meta = json.load(fh)
  
  total = meta["total"]
  
  graph = tf.Graph()
  print("Loading model...")
  with graph.as_default() as g:
    test_batch = get_dataset(config.test_record_file, get_record_parser(
      config, is_test=True), config).make_one_shot_iterator()
    
    model_dict={'Raw':Raw_Model,
                'TPE':TPE_Model,
                'T5':T5_Model,
                'T5_Nob':T5_Nob_Model,
                'Soft_T5':Soft_T5_Model,
                'Soft_T5_Nob':Soft_T5_NoB_Model
              }
    if config.model not in model_dict:
      print('wrong %s model name' %(config.model))
      exit(0)
    
    model = model_dict[config.model](config, test_batch, word_mat, char_mat, trainable=False, graph = g)
    
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True
        
    with tf.Session(config=sess_config) as sess:
      sess.run(tf.global_variables_initializer())
      saver = tf.train.Saver()
      saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))
      if config.decay < 1.0:
        sess.run(model.assign_vars)
      losses = []
      answer_dict = {}
      remapped_dict = {}
      for step in tqdm(range(total // config.batch_size + 1)):
        qa_id, loss, yp1, yp2 = sess.run(
          [model.qa_id, model.loss, model.yp1, model.yp2])
        answer_dict_, remapped_dict_ = convert_tokens(
          eval_file, qa_id.tolist(), yp1.tolist(), yp2.tolist())
        answer_dict.update(answer_dict_)
        remapped_dict.update(remapped_dict_)
        losses.append(loss)
      loss = np.mean(losses)
      
      metrics = evaluate(eval_file, answer_dict)
      
        
      with open(config.answer_file, "w") as fh:
        json.dump(remapped_dict, fh)
      print("Exact Match: {}, F1: {}".format(
      metrics['exact_match'], metrics['f1']))
