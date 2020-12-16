# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 20:14:48 2020
"""
import sys
import os
import numpy as np
import pickle
from tqdm import tqdm

if __name__ == '__main__':
  #when the methods converge
  fname = sys.argv[1]
  
  #fout='out/temporal_order_b_100/non-pe.pkl'
  #fout='out/temporal_order_b_100/soft_t5-pe.pkl'
  #fout='out/temporal_order_b_100/tpe.pkl'
  #fout='out/temporal_order_b_100/t5-pe.pkl'
  
  #fout='out/temporal_order_b_500/non-pe.pkl'
  #fout='out/temporal_order_b_500/soft_t5-pe.pkl'
  #fout='out/temporal_order_b_500/tpe-pe.pkl'
  #fout='out/temporal_order_b_500/t5-pe.pkl'
  
  line_no = 0
  ret_epoch=[]
  ret_acc = []
  
  with open(fname) as file_:
    for line in tqdm(file_):
      line = line[:-1]
      if line_no%2==0:
        epoch_no = line.split(':')[0].replace('[','').replace(']', '')
        print(epoch_no)
        ret_epoch.append(epoch_no)
      
      if line_no%2==1:
        dev_acc=line.split('\t')[0]
        test_acc=line.split('\t')[2]
        print(dev_acc)
        ret_acc.append(dev_acc)
      
      line_no+=1
  #params={'ret_epoch':ret_epoch,'ret_acc':ret_acc}
  #pickle.dump(params,open(fout,'wb'))
  
  for i in range(len(ret_acc)-1):
    if ret_acc[i]<ret_acc[i+1]:
      converge_epoch=i+1
  
  print(converge_epoch)
  print(test_acc)
  
  
      
      
      
  