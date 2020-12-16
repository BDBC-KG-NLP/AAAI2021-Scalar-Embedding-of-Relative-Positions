# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 20:21:43 2020
"""
import os
import random
import numpy as np
from helper import gen_relative_pos
random.seed(1024)

def unpack_params(params):
  embed_params, other_params, wd_params = [],[],[]
  
  for v in params:
    k = v.name
    if 'embed' in k:
      embed_params.append(v)
    elif 'norm' in k or 'bias' in k:
      other_params.append(v)
    else:
      wd_params.append(v)
  
  return embed_params, other_params, wd_params

def position_index(sentence, length):
  index = [0 for i in range(length)]
  
  for idx in range(length):
    index[idx] = idx+1
    
  return index

def load_reber_data(dataset):
  data_dir = "../data/" + dataset
  q_max_seq = 0
  a_max_seq = 8
  
  for data_name in ['train', 'dev', 'test']:
    datas=[]
    data_file = os.path.join(data_dir, data_name)+'.txt'
    
    for line in open(data_file,'r'):
      line = line.strip()
      items = line.split(' ')
      
      question=items[:-1]
      
      q_max_seq = max(len(question),q_max_seq)
      
  t5_pos = gen_relative_pos(q_max_seq)
  vocabs = {'PAD':0,'B':1,'T':2,'P':3,'S':4,'X':5,'V':6,'E':7}
  ret_datas = []
  for data_name in ['train', 'dev', 'test']:
    datas=[]
    data_file = os.path.join(data_dir, data_name)+'.txt'
    
    for line in open(data_file,'r'):
      line = line.strip()
      items = line.split(' ')
      
      question = []
      for item in items[:-1]:
        question.append(vocabs[item])
        
      item={}
      item['question']=question
      
      tag = [0 for _ in range(len(vocabs))]
      tag[vocabs[items[-1]]]=1
      item['tag'] = tag
      
      item['position'] = position_index(question, len(question))
      item['t5_position']=t5_pos
      datas.append(item)
    
    ret_datas.append(datas)
    
  return ret_datas, vocabs,q_max_seq,a_max_seq

def load_adding_problem_data(dataset):
  data_dir = "../data/" + dataset
  vocabs=None
  
  ret_datas = []
  q_max_seq = int(dataset.split('_')[-1])
  a_max_seq = 1
  
  t5_pos = gen_relative_pos(q_max_seq)
  for data_name in ['train', 'dev', 'test']:
    datas=[]
    data_file = os.path.join(data_dir, data_name)+'.txt'
    
    for line in open(data_file,'r'):
      line = line.strip()
      items = line.split(' ')
      
      question = []
      for item in items[:-1]:
        question.append([float(item.split('_')[0]),float(item.split('_')[1])])
      
      item={}
      item['question']=question
      item['tag'] = float(items[-1])
      
      item['position'] = position_index(question, len(question))
      item['t5_position']=t5_pos
      datas.append(item)
    
    ret_datas.append(datas)
    
  return ret_datas,vocabs,q_max_seq,a_max_seq

def load_temporal_order_data(dataset):
  data_dir = "../data/" + dataset
  vocabs = {'a':0,'b':1,'c':2,'d':3,'X':4,'Y':5,'E':6,'B':7}
  question_dict={'Q':0,'R':1,'S':2,'U':3}
  
  ret_datas = []
  q_max_seq = int(dataset.split('_')[-1])
  a_max_seq = 4
  t2_position = gen_relative_pos(q_max_seq)
  for data_name in ['train', 'dev', 'test']:
    datas=[]
    data_file = os.path.join(data_dir, data_name)+'.txt'
    
    
    for line in open(data_file,'r'):
      line = line.strip()
      items = line.split(' ')
      tag_vector = [0,0,0,0]
      
      question = [vocabs[wd] for wd in items[:-1]]
      
      if items[-1] not in question_dict:
        print('wrong flag:',question)
        
      flag = int(question_dict[items[-1]])
      item = {}
      
      item['question'] = question
      
      tag_vector[flag]=1
      item['tag'] = tag_vector
      item['position'] = position_index(question, len(question))
      item['t5_position']=t2_position
      
      datas.append(item)
    
    ret_datas.append(datas)
  
  return ret_datas, vocabs, q_max_seq, a_max_seq

def load_seq_data(dataset):
  data_dir = "../data/" + dataset
  vocabs = {'PAD':0,'0':1,'1':2,'2':3}
  ret_datas = []
  q_max_seq = 0
  a_max_seq = 2
  
  for data_name in ['train', 'dev', 'test']:
    datas=[]
    data_file = os.path.join(data_dir, data_name)+'.txt'
    
    
    for line in open(data_file,'r'):
      line = line.strip()
      items = line.split(' ')
      tag_vector = [0,0]
      
      question = [vocabs[wd] for wd in items[:-1]]
      
      if len(question)>50:
        question=question[:49]
        question+=[3]
      
      q_max_seq = max(len(question),q_max_seq)
      
      
      flag = int(items[-1])
      item = {}
      
      item['question'] = question
      tag_vector[flag]=1
      item['tag'] = tag_vector
      item['position'] = position_index(question, len(question))
      item['t5_position']=gen_relative_pos(50)
      
      datas.append(item)
    
    ret_datas.append(datas)
  
  return ret_datas, vocabs, q_max_seq, a_max_seq
  
def yield_data(batch_size, q_max_seq, datas):
  batch_data = []
  batch_pos = []
  batch_tag = []
  batch_t5_pos =[]
  
  random.shuffle(datas)
  random.shuffle(datas)
  
  for idx, data in enumerate(datas):
    question = data['question']
    pos = data['position']
    len_q = len(question)
    
    if len_q < q_max_seq:
      add_len_q = q_max_seq - len_q
      question += [0 for i in range(add_len_q)]
      pos += [0 for i in range(add_len_q)]
    
    batch_data.append(question)
    batch_tag.append(data['tag'])
    batch_pos.append(pos)
    batch_t5_pos.append(data['t5_position'])
    
    
    if len(batch_data) == batch_size:
      batch_data = batch_data
      batch_tag = np.array(batch_tag,np.float32)
      batch_pos = np.array(batch_pos,np.int32)
      batch_t5_pos = np.array(batch_t5_pos,np.int32)
      yield batch_data, batch_tag, batch_pos,batch_t5_pos
      batch_data,batch_tag,batch_pos,batch_t5_pos =[],[],[],[]
    
  if len(batch_data)!=batch_size and len(batch_data)!=0:
    yield batch_data, batch_tag, batch_pos, batch_t5_pos

