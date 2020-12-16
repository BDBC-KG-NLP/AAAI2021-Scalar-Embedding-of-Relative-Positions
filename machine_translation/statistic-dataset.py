# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 17:46:30 2020
"""
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import Counter


def get_sent_length_info():
  
  sentidx_2_len_dict = {}
    
  max_len = 0
  
  sent_idx = 0
  #with open('wmt17/newstest2017.tc.32k.en') as file_1, \
 #     open('wmt17/newstest2017.tc.32k.de') as file_2:
  #with open('wmt17/newstest2016.tc.32k.en') as file_1, \
  #    open('wmt17/newstest2016.tc.32k.de') as file_2:
  with open('wmt17/corpus.tc.32k.en.shuf') as file_1, \
      open('wmt17/corpus.tc.32k.de.shuf') as file_2:
  #with open('wmt17/dev/newstest2014.tc.32k.en') as file_1, \
  #    open('wmt17/dev/newstest2014.tc.32k.de') as file_2:
    for en_sent, de_sent in tqdm(zip(file_1,file_2)):
      line_len = len(en_sent.strip().split(' '))
      
      sentidx_2_len_dict[sent_idx] = line_len
      max_len = max(max_len, line_len)
      sent_idx += 1
      
  print(max_len)
  return sentidx_2_len_dict, max_len

def get_bucketing_samples(doc2lent):
  
  bins = np.array([0,20,40,60,80,100,120,140,160,180,200,220,240,260,280,300])
  
  fnames = list(doc2lent.keys())
  lengths = list(doc2lent.values()) 
  
  data = {'fname':fnames,'seq_length':lengths}
  
  data_frame = pd.DataFrame(data=data,)
   
  
  data_frame['bucket'] = pd.cut(data_frame.get('seq_length'),bins)
  
  print(data_frame.head())
  
  counter = Counter(data_frame.get('bucket'))
  
  for key in counter:
    print(key, counter[key])
  
  
if __name__ == '__main__':
  sentidx_2_len_dict, max_len = get_sent_length_info()
  
  get_bucketing_samples(sentidx_2_len_dict)