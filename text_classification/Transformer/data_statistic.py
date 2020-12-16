# -*- coding: utf-8 -*-
"""
Created on Fri May 29 13:09:25 2020
"""
import sys
import numpy as np
import spacy

spacy_en = spacy.load('en')

def tokenizer(x):
  #lower case words
  return [tok.text.lower() for tok in spacy_en.tokenizer(x.replace('<br />',' '))]


if __name__ == "__main__":
  data = sys.argv[1]
  
  sent_lent_list=[]
  with open('../data/'+data+'/train.txt') as file_:
    for line in file_:
      line = line[:-1]
      if data=='TREC':
        sent_lent = len(line.split("\t")[1].split(' '))
      else:
        sent_lent = len(line.split("\t")[0].split(' '))
      sent_lent_list.append(sent_lent)
      if sent_lent > 500:
        print(line)
        print('--------------')
    
  print(np.average(sent_lent_list))
  print(max(sent_lent_list))