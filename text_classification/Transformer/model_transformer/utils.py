# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 08:44:25 2020

@author: wujs
"""

def unpack_params(params):
  embed_params, other_params, wd_params = [],[],[]
  
  for var in params:
    k = var.name
    if 'embed' in k:
      embed_params.append(var)
    elif 'norm' in k or 'bias' in k:
      other_params.append(var)
    else:
      wd_params.append(v)
  
  return embed_params, other_params, wd_params