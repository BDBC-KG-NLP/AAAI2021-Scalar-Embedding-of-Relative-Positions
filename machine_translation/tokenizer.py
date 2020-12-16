# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 12:37:09 2020

@author: wujs
"""
import codecs
import html
from string import punctuation
import re

# tokens=nltk.word_tokenize('''"With everything else that's going wrong with the world, he was that diamond in the rough that was shining bright every day," he said.''')
# print(tokens)


# tokens=nltk.word_tokenize('''Seine Art mit seiner Familie zu verbinden war es, uns immer etwas zu kochen, uns das Abendessen zuzubereiten," sagte Louis Galicia.''')
# print(tokens)
'''
fout = codecs.open('wmt17/newstest2017.tc.de','w','utf-8')
with codecs.open('wmt17/newstest2017.de','r','utf-8') as file_:
  for line in file_:
    line=line[:-1]
    tokens = nltk.word_tokenize(line)
    print(tokens)
    fout.write(' '.join(tokens)+'\n')
    fout.flush()

fout.close()
'''
html_escape_table = {
  "&": "&amp;",
  '"': "&quot;",
  "'": "&apos;",
  ">": "&gt;",
  "<": "&lt;"
}

def html_escape(text):
  """Produce entities within text."""
  return "".join(html_escape_table.get(c,c) for c in text)

def get_tokenizer(strs):
  new_strs=''
  for char in strs:
    if char not in punctuation:
      new_strs+=char
    else:
      new_strs=new_strs+' '+char
  
  html_strs = html_escape(new_strs)
  return html_strs

#strs='''You can't ban everything you're opposed to, and I'm opposed to the wearing of the burka.'''
#print(get_tokenizer(strs))

fout = codecs.open('wmt17/newstest2017.tc.de','w','utf-8')
with codecs.open('wmt17/newstest2017.de','r','utf-8') as file_:
  for line in file_:
    line=line[:-1]
    strs = get_tokenizer(line)
    fout.write(strs+'\n')
    fout.flush()

fout.close()