# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 20:39:09 2020

"""
import os
import random
import collections
import numpy as np

prob=0.5


#Return a portion of the string corresponding to the top half of the graph
#we generate the reber grammer sequence
def reber_phrase_a(string):
  string+='T'
  luck=random.random()
  
  while luck<prob:
    string += 'S'
    luck=random.random()
  
  string+='X'
  
  return string

#Return a portion of the string corresponding to the bottom half of the graph
def reber_phrase_b(string):
  string +='T'    #herein is 'T'  
  luck = random.random()
  
  while luck<prob:
    string+='T'
    luck = random.random()
  
  string+='V'
  
  luck1 = random.random()
  if luck1<=0.5:
    string+='P'
    luck2 = random.random()
    if luck2<=0.5:
      string+='X'
      string = reber_phrase_b(string)
    else:
      string +='SE'
  else:
    string+='VE'
  
  return string

def reber_string():
  string='B'
  luck1 = random.random()
  
  if luck1<=0.5:
    string = reber_phrase_a(string)
    luck2 =  random.random()
    
    #add loop probablity
    if luck2<=0.5:
      string+='X'
      string = reber_phrase_b(string)
    else:
      string +='SE'
  else:
    string+='P'
    string = reber_phrase_b(string)
    
  return string


def embedded_reber():
  string='B'
  luck = random.random()
  
  if luck<0.5:
    string += 'P'
    string+=reber_string()
    string +='T'
    string+=reber_string()
    string+='P'
  else:
    string += 'T'
    string += reber_string()
    string += 'P'
    string += reber_string()
    string += 'T'
  
  return string

if __name__ == '__main__':
  dir_path = '../data/reber'
  
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)
  
  train_out_file = open(dir_path+'/train.txt','w')
  dev_out_file = open(dir_path+'/dev.txt','w')
  test_out_file = open(dir_path+'/test.txt','w')
  
  train_seq=[]
  train_nums = 30000
  seq_dict = collections.OrderedDict()
  
  while len(seq_dict)<train_nums:
    strs = embedded_reber()
    
    if len(strs)>=500:
      print(len(strs))
      continue
    # if prob==0.99:
    #   if len(strs)>=1000:
    #     continue
    # elif prob==0.98:
    #   if len(strs)>=500:
    #     continue
    
    seqs = strs[1:-1]
    train_seq.append(len(seqs))
    
    tag = strs[-1]
    
    seq_str = ''
    for seq in seqs:
      seq_str = seq_str+' '+seq
    
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
  
  print(np.average(train_seq))

