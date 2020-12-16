# -*- coding: utf-8 -*-
"""
Created on Fri May  1 12:13:08 2020
"""

import os
import sys
import time

import numpy as np
from functools import wraps
import pickle

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

@log_time_delta
def get_ret(data, model_name,embedding_dim,num_attention_heads,soft_t5_width=5,
            is_Embedding_Needed='1',pool='mean',train_nums=20000):
  
  if data == 'sst2' or data =='TREC':
    trail_num=3
  else:
    trail_num=1
  if True:
    if model_name == 'T5_PE' or model_name == 'Soft_T5_PE' or model_name=='Soft_Multi_T5_PE':
      soft_t5_width_list=[5,10,20,50]
      if model_name=='T5_PE' and ('adding' in data or 'temporal' in data):
        soft_t5_width_list=[32,64,100,128,256,300,500,512]
      
      if 'process_cls_' in data:
        soft_t5_width_list = [5,10,20,32,50,64,128,256]
      
      
      for soft_t5_width in soft_t5_width_list:
        acc_trail = []
        print('---soft_t5_width---:', soft_t5_width)
        for trail in range(trail_num):
          ckpt_path = 'model_save/'+data+'/'+ \
                            '_'.join([model_name,embedding_dim,
                                    'L1',
                                    'H'+ num_attention_heads,
                                    str(train_nums),
                                    pool,
                                    str(soft_t5_width),
                                    str(trail+1)])
          
          if is_Embedding_Needed=='0':
            ckpt_path += '-random'
          
          if is_Embedding_Needed=='0':
            ckpt_path = ckpt_path+ '_l2-'+str(0.0)
      
          if os.path.exists(ckpt_path):
            
            ret_file = ckpt_path+'/test_ret.p'
            
            if os.path.exists(ret_file)==False:
              continue
            acc_test = pickle.load(open(ret_file,'rb'))
            acc_trail.append(acc_test)
            print (acc_test)
          else:
            print(ckpt_path, ' do not exist...')
          
        if len(acc_trail)!=1 and len(acc_trail)>0:
          print('Avg:',np.mean(acc_trail))
          print('----------')
    else:
      acc_trail = []
      for trail in range(trail_num):
        ckpt_path = 'model_save/'+data+'/'+ \
                            '_'.join([model_name,embedding_dim,
                                    'L1',
                                    'H'+ num_attention_heads,
                                    str(train_nums),
                                    pool,
                                    soft_t5_width,
                                    str(trail+1)])
          
        if is_Embedding_Needed=='0':
          ckpt_path += '-random'
        
        if is_Embedding_Needed=='0':
          ckpt_path = ckpt_path+ '_l2-'+str(0.0)
            
        if os.path.exists(ckpt_path):
          ret_file = ckpt_path+'/test_ret.p'
          if os.path.exists(ret_file)==False:
            continue
          acc_test = pickle.load(open(ret_file,'rb'))
          acc_trail.append(acc_test)
          print (acc_test)
        else:
          print(ckpt_path, ' do not exist...')
        
      if len(acc_trail)!=1 and len(acc_trail)>0:
        print('Avg:',np.mean(acc_trail))
      
if __name__ =="__main__":
  data=sys.argv[1]
  model_name=sys.argv[2]
  embedding_dim=sys.argv[3]
  num_attention_heads = sys.argv[4]
  soft_t5_width=sys.argv[5]
  pool=sys.argv[6]
  is_Embedding_Needed=sys.argv[7]
  train_nums = sys.argv[8]
  print(is_Embedding_Needed,': is_Embedding_Needed')
  
  get_ret(data, model_name, embedding_dim, num_attention_heads, soft_t5_width, is_Embedding_Needed,pool,train_nums)