# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 16:44:21 2020

"""

import sys
import os
import numpy as np
import collections
np.random.seed(1024)

def gen_seq(alpha,beta):
  stationary_dist=[2/3,1/3]
  seq=[]
  rnd_v = np.random.random(size=1)[0]
  if rnd_v <= 2/3:
    seq.append('0')
  else:
    seq.append('1')
  
  while len(seq)<50:
    char = seq[-1]
    rnd_val = np.random.random(size=1)[0]
    
    if char =='0':
      if rnd_val <= alpha:
        seq.append('1')
      else:
        seq.append('0')
    if char == '1':
      if rnd_val <= beta:
        seq.append('0')
      else:
        seq.append('1')

  return seq

if __name__=='__main__':
  
  rnd_dict = {}
  
  rnd_dict['0'] = ['0','1']
  rnd_dict['1'] = ['0','1']
  
  dir_path = '../data/process_cls_'+str(50)
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)
  
  train_out_file = open(dir_path+'/train.txt','w')
  dev_out_file = open(dir_path+'/dev.txt','w')
  test_out_file = open(dir_path+'/test.txt','w')
  
  train_nums = 15000
  seq_dict = collections.OrderedDict()
  
  while len(seq_dict)<train_nums:
    seqs = gen_seq(alpha=0.2,beta=0.4)
    
    seq_str = ' '.join(seqs)+' '+str(0)+'\n'
    
    if seq_str not in seq_dict:
      seq_dict[seq_str]=1
  
  while len(seq_dict)<30000:
    seqs = gen_seq(alpha=0.3,beta=0.6)

    seq_str = ' '.join(seqs)+' '+'1'+'\n'
    
    if seq_str not in seq_dict:
      seq_dict[seq_str]=0
  
  
  pos_list = []
  neg_list = []
  train_nums = 15000
  ids = 0
  for key in seq_dict:
    ids += 1
    if ids <= train_nums:
      neg_list.append(key)
    else:
      pos_list.append(key)

 
  for seq in pos_list[:10000]+neg_list[:10000]:
    train_out_file.write(seq)
    train_out_file.flush()
  
  
  for seq in pos_list[10000:12500]+neg_list[10000:12500]:
    dev_out_file.write(seq)
    dev_out_file.flush()
  dev_out_file.close()
  
  
  for seq in pos_list[12500:15000]+neg_list[12500:15000]:
    test_out_file.write(seq)
    test_out_file.flush()
  test_out_file.close()
