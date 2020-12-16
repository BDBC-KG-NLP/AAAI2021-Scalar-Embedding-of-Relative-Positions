# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:14:02 2020

@author: wujs
"""
import numpy as np

def t5_pos(relative_position,bidirectional=True,num_buckets=32,max_distance=128):
  '''
  if bidirectional=False, then positive relative positions are invalid.
  '''
  ret = [0 for i in range(len(relative_position))]
  n = -1 * relative_position
  
  if bidirectional:
    num_buckets //=2
    ret += np.less(n,0)*num_buckets
    n=np.abs(n)
  else:
    n = np.maximum(n,0)
  #print(n)
  #print(ret)
  #now n is in the range [0,inf)
  max_exact = num_buckets // 2
  #print(max_exact)
  is_small = np.less(n,max_exact)
  t1 = np.log((n+1e-9)*1.0/max_exact)
  #print(t1)
  t2 = np.log(max_distance/max_exact)
  #print(t2)
  val_if_large = max_exact+ t1/t2 *(num_buckets-max_exact)
  val_if_large = val_if_large.astype(np.int32)
  #print(val_if_large)
  val_if_large =np.minimum(val_if_large,num_buckets-1)
  ret += np.where(is_small,n,val_if_large)
  return ret

def gen_relative_pos(seq_lent,directed=False):
  rel_mask_matrix = np.zeros((seq_lent,seq_lent),dtype=np.int32)
  for i in range(seq_lent):
    for j in range(seq_lent):
      if directed and j>i:
          continue
      
      rel_mask_matrix[i][j]=j-i

  t5_relative_pos_matrix = []
  for i in range(seq_lent):
    t5_relative_pos_matrix.append(t5_pos(rel_mask_matrix[i,:]))
  
  
  return np.array(t5_relative_pos_matrix,dtype=np.int32)

if __name__ == '__main__':
  pos_index = gen_relative_pos(50)
  for i in [1]:
    print(list(pos_index[i,:]))
  
