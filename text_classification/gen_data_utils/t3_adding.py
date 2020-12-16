# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 23:09:14 2020
"""
import sys
import random
import os
import collections


def genAddPairSeq(T):
  #we revise the adding problem setting the marks for starting and ending as [-1]
  seq_length = T
  mark = 0
  ret_list = []
  for i in range(seq_length):
    val = round(random.uniform(-1, 1),4)
    
    ret_list.append([val, mark])
    
  ret_list[0][1]=-1
  
  t1 = random.choice([i for i in  range(10)])
  
  if t1 ==0:
    ret_list[t1]=[0.0,1]
  else:
    ret_list[t1][1]=1
    
  first_half = []
  for i in range(T//2-1):
    if i!=t1:
      first_half.append(i)
  
  t2 = random.choice(first_half)
  ret_list[t2][1]=1
  
  ret_list[-1][1]=-1
  
  return ret_list, round(0.5+(ret_list[t1][0]+ret_list[t2][0])/4.0,4)

if __name__ == "__main__":
  seq_length=int(sys.argv[1])
  dir_path = '../data/adding_problem_'+str(seq_length)
  
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)
  
  train_out_file = open(dir_path+'/train.txt','w')
  dev_out_file = open(dir_path+'/dev.txt','w')
  test_out_file = open(dir_path+'/test.txt','w')
  
  train_nums = 30000
  seq_dict = collections.OrderedDict()
  
  while len(seq_dict)<train_nums:
    seqs, tag = genAddPairSeq(seq_length)
    
    seq_str = ''
    for seq in seqs:
      seq_str = seq_str+' '+str(seq[0])+'_'+str(seq[1])
    
    seq_str =seq_str.strip()
    seq_str = seq_str+' '+str(tag)+'\n'
    
    if seq_str not in seq_dict:
      seq_dict[seq_str]=1
  
  seq_list = [key for key in seq_dict]
  
  
  for seq in seq_list[:20000]:
    train_out_file.write(seq)
    train_out_file.flush()
  train_out_file.close()
  
  
  for seq in seq_list[20000:25000]:
    dev_out_file.write(seq)
    dev_out_file.flush()
  dev_out_file.close()
  
  for seq in seq_list[25000:30000]:
    test_out_file.write(seq)
    test_out_file.flush()
    
  test_out_file.close()