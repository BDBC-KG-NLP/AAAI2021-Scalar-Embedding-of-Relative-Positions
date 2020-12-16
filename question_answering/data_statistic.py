# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 18:25:00 2020
"""

import tensorflow as tf
import random
from tqdm import tqdm
import spacy
import ujson as json
from collections import Counter
import numpy as np
from codecs import open
import os

'''
This file is taken and modified from R-Net by HKUST-KnowComp
https://github.com/HKUST-KnowComp/R-Net
'''

nlp = spacy.blank("en")

#we need to do some statistic


def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]


def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def process_file(filename, data_type, word_counter, char_counter):
    print("Generating {} examples...".format(data_type))
    examples = []
    eval_examples = {}
    total = 0
    with open(filename, "r") as fh:
        source = json.load(fh)
        for article in tqdm(source["data"]):
            for para in article["paragraphs"]:
                context = para["context"].replace(
                    "''", '" ').replace("``", '" ')
                context_tokens = word_tokenize(context)
                context_chars = [list(token) for token in context_tokens]
                spans = convert_idx(context, context_tokens)
                for token in context_tokens:
                    word_counter[token] += len(para["qas"])
                    for char in token:
                        char_counter[char] += len(para["qas"])
                for qa in para["qas"]:
                    total += 1
                    ques = qa["question"].replace(
                        "''", '" ').replace("``", '" ')
                    ques_tokens = word_tokenize(ques)
                    ques_chars = [list(token) for token in ques_tokens]
                    for token in ques_tokens:
                        word_counter[token] += 1
                        for char in token:
                            char_counter[char] += 1
                    y1s, y2s = [], []
                    answer_texts = []
                    is_impossible=False
                    for answer in qa["answers"]:
                        answer_text = answer["text"]
                        answer_start = answer['answer_start']
                        answer_end = answer_start + len(answer_text)
                        answer_texts.append(answer_text)
                        answer_span = []
                        for idx, span in enumerate(spans):
                            if not (answer_end <= span[0] or answer_start >= span[1]):
                                answer_span.append(idx)
                        y1, y2 = answer_span[0], answer_span[-1]
                        y1s.append(y1)
                        y2s.append(y2)
                        print(y1s)
                        print(y2s)
                        print('--------------')
                        
                    if len(qa["answers"])==0:
                      y1s.append(0)
                      y2s.append(0)
                      is_impossible=True
                      print(y1s)
                      print(y2s)
                      print('--------------')
                    
                    example = {"context_tokens": context_tokens, "context_chars": context_chars, "ques_tokens": ques_tokens,
                               "ques_chars": ques_chars, "y1s": y1s, "y2s": y2s, "id": total,'is_impossible':is_impossible}
                    examples.append(example)
                    eval_examples[str(total)] = {
                        "context": context, "spans": spans, "answers": answer_texts, "uuid": qa["id"]}
        random.shuffle(examples)
        print("{} questions in total".format(len(examples)))
    return examples, eval_examples



def analysis( examples, data_type):
  print("Processing {} examples...".format(data_type))
  total = 0
  total_=0
  meta = {}
  is_impossible_total = 0
  para_len_list = []
  ques_len_list= []
  for example in tqdm(examples):
    total += 1
    
    para_len = len(example["context_tokens"])
    ques_len = len(example["ques_tokens"])
    
    para_len_list.append(para_len)
    ques_len_list.append(ques_len)
    try:
      ans_len = example["y2s"][0] - example["y1s"][0]
      start, end = example["y1s"][-1], example["y2s"][-1]
      if example["is_impossible"]:
        is_impossible_total+=1
    except:
      continue
    
    if para_len > 400:
      total_+=1
    if total % 1000==0:
      print(para_len, ques_len)
      
      
  print(np.mean(para_len_list),max(para_len_list))
  print(np.mean(ques_len_list),max(ques_len_list))
  print('total:',total)
  print('non answer:', is_impossible_total)
  print('total>400:',total_)


if __name__ == '__main__':
  word_counter, char_counter = Counter(), Counter()
  '''
  train_examples, train_eval = process_file(
        config.train_file, "train", word_counter, char_counter)
  dev_examples, dev_eval = process_file(
        config.dev_file, "dev", word_counter, char_counter)
  '''
  
  for tag in ['dev']:
    #test_file = os.path.join("datasets","squad-v1", tag+"-v1.1.json")
    test_file = os.path.join("datasets","squad-v2", tag+"-v2.0.json")
    
    test_examples, test_eval = process_file(
          test_file, tag, word_counter, char_counter)
    
    analysis(test_examples,tag)
