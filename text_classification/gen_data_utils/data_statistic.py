# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 12:05:55 2020
"""

import sys
#draw pictures
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
def seq_count(seq):
  zero_num,one_num=0,0
  for key in seq:
    if key=='0':
      zero_num+=1
    else:
      one_num+=1
  return zero_num,one_num

def seq_prob(transfer_matrix,seq):
  stationary_dist = [2/3,1/3]
  prob = []
  if seq[0]=='0':
    prob.append(stationary_dist[0])
  else:
    prob.append(stationary_dist[1])

  for i in range(len(seq)-1):
    first = int(seq[i])
    second = int(seq[i+1])
    prob.append(transfer_matrix[first,second])
  prob = np.array(prob,np.float32)
  prob = np.exp(np.sum(np.log(prob)))
  #print(seq,prob)
  return prob

if __name__=='__main__':
  #we generate the seq with fixed length
  pic_type = sys.argv[1]
  seq_lent = sys.argv[2]
  dir_path = '../data/process_cls_'+str(seq_lent)
  train_out_file = dir_path+'/train.txt'
  if pic_type == 'oracle':
    train_out_file = dir_path +'/test.txt'
  pos_lent = []
  neg_lent = []
  
  pos_zero_number=[]
  pos_one_number = []
  
  neg_zero_number=[]
  neg_one_number = []
  
  neg_transfer_matrix=np.array([[0.8,0.2],[0.4,0.6]])
  pos_transfer_matrix=np.array([[0.7,0.3],[0.6,0.4]])
  same_prob_samples = 0
  all_samples = 0
  right_samples = 0
  with open(train_out_file,'r') as file_:
    for line in tqdm(file_):
      line = line.strip()
      
      items = line.split(' ')
      pos_prob = seq_prob(pos_transfer_matrix,items[:-1])
      neg_prob = seq_prob(neg_transfer_matrix,items[:-1])
      
      all_samples+=1
      flag=False
      if pos_prob>neg_prob:
        if items[-1]=='1':
          flag=True
      elif pos_prob<neg_prob:
        if items[-1]=='0':
          flag=True
      else:
        same_prob_samples+=1
      
      if flag:
        right_samples+=1
        
      if items[-1]=='0':
        neg_lent.append(len(items)-1)
        zero_num,one_num = seq_count(items[:-1])
        neg_one_number.append(one_num)
        neg_zero_number.append(zero_num)
      else:
        pos_lent.append(len(items)-1)
        zero_num,one_num = seq_count(items[:-1])
        pos_one_number.append(one_num)
        pos_zero_number.append(zero_num)
  
  
  pos_mean,pos_var = np.mean(np.array(pos_lent),0),np.std(np.array(pos_lent))
  neg_mean,neg_var = np.mean(np.array(neg_lent),0),np.std(np.array(neg_lent))
    
  print(pos_mean,pos_var)
  print(neg_mean,neg_var)
    
  #train_out_file.close()
  if pic_type=='process':
    sns.set_palette('hls')
    mpl.rc("figure",figsize=(4,3))
    sns.distplot(pos_lent,label='Positive')
    sns_plot=sns.distplot(neg_lent,label='Negative')
    sns_plot.figure.savefig('process.png')
  elif pic_type=='zero':
    sns.set_palette('hls')
    mpl.rc("figure",figsize=(4,3))
    sns.distplot(pos_zero_number,label='Positive')
    sns_plot=sns.distplot(neg_zero_number,label='Negative')
    sns_plot.figure.savefig('zero.png')
  elif pic_type=='one':
    sns.set_palette('hls')
    mpl.rc("figure",figsize=(4,3))
    sns.distplot(pos_one_number,label='Positive')
    sns_plot=sns.distplot(neg_one_number,label='Negative')
    sns_plot.figure.savefig('one.png')
  elif pic_type=='diff':
    sns.set_palette('hls')
    mpl.rc("figure",figsize=(4,3))
    sns.distplot(np.array(pos_one_number)-np.array(neg_one_number),label='One')
    sns_plot=sns.distplot(np.array(pos_zero_number)-np.array(neg_zero_number),label='Zero')
    sns_plot.figure.savefig('diff.png')
  else:
    #we need to statisitc naive bayesian
    print(all_samples)
    print(same_prob_samples)
    print(right_samples, right_samples*1.0/all_samples)
