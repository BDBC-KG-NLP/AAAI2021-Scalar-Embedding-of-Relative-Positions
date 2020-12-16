# -*- coding: utf-8 -*-
"""
Created on Fri May  8 20:16:10 2020
"""

import numpy as np
import seaborn as sn
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import csv

head=2 
hiddens=[10,20,50]

model_name = 'Soft_T5_PE'
#model_name = 'T5_PE'
#model_name = 'T5_PE_NoB'
#model_name='TPE'
#task = 'reber'
#task = 'process_cls_50'
#task = 'TREC'
#task = 'temporal_order_500'
task = 'sst2'

if model_name == 'Soft_T5_PE':
  dir_path = task+'/Soft_T5_PE/'
  print(model_name)
  #ret = pickle.load(open(dir_path + 'test_ret.p','rb'))
  #print(ret)
  #params={'soft_t5_alpha':soft_t5_alpha,
  #                    'soft_t5_beta':soft_t5_beta}
  
  params = pickle.load(open(dir_path + 'params.p','rb'))
  soft_t5_alpha = params['soft_t5_alpha']
  soft_t5_beta = params['soft_t5_beta']
  #print('soft_t5_alpha:',soft_t5_alpha)
  #print('soft_t5_beta:',soft_t5_beta)
  t5_att_bias = pickle.load(open(dir_path+'t5_att_bias.p','rb'))
  att_right = t5_att_bias['right']
  att_left = t5_att_bias['left']
  
  seq_len = np.shape(att_right)[1]
  relative_position = np.arange(seq_len)[None,:] - np.arange(seq_len)[:,None] 
  #print(np.shape(att_right))
  #print(relative_position)
  '''
  att_dict={}
  for i in range(6):
    # if task in ['reber','process_cls_50']:
    #   print(np.shape(t5_att_bias['left']))
    #   weights =  list(t5_att_bias['left'][:,:seq_len,:seq_len][i,-1,:]) + list(t5_att_bias['right'][:,:seq_len,:seq_len][i,0,1:])
    # else:
    weights =  list(t5_att_bias['left'][i,:]) + list(t5_att_bias['right'][i,1:])
    
    att = np.exp(weights)/sum(np.exp(weights))
    
    #print(np.shape(att))
    #print(np.amax(att))
    #print(np.amax(weights))
    #print('---------------')
    att_dict[i]=np.amax(att)
    x_idx =list(-1*np.array(range(seq_len))[::-1]) +list(range(seq_len))[1:]
    #print('---------------------')
    
    with open(dir_path+'/csv/'+model_name+'-'+task+'-prior-h'+str(i),'w',newline='') as csvfile: 
      writer=csv.writer(csvfile,delimiter=',')
      for idx in range(len(x_idx)):
        writer.writerow([x_idx[idx],att[idx]])
    
    plt.plot(x_idx, att,label='h-'+str(i))
    plt.legend(loc='upper right', shadow=True,fontsize='x-large')
    plt.show()
    
  print(sorted(att_dict.items(),key=lambda item:item[1],reverse=True))
  '''
  '''
  save_param={'question_text':question_text,
                      'input_y':y_i,
                      'predict_y':scores[qidx],
                      'context_scores':context_scores,
                      'final_scores':final_scores,
                      'att_mask':att_mask,
                      't5_att_bias':t5_att_bias
                      }'''
  
  save_param=pickle.load(open(dir_path+model_name+'-'+task+'-att.pkl','rb'))
  question_text=save_param['question_text']
  
  print(question_text)
  
  input_y=save_param['input_y']
  predict_y=save_param['predict_y']
  
  print(np.shape(save_param['context_scores']))
  print('----------------')
  #没道理取1呢~这个图不是1的貌似~~
  context_scores=save_param['context_scores']
  final_scores=save_param['final_scores']
  att_mask=save_param['att_mask']
  t5_att_bias=save_param['t5_att_bias'][0]
  fig = plt.figure()
  
  '''
  ax1 = fig.add_subplot(311)
  ax2 = fig.add_subplot(312)
  ax3 = fig.add_subplot(313)
  
  print('t5_att_bias:',np.shape(t5_att_bias))
  ax_list =[ax1,ax2,ax3]
  for i in [4]:
    sns.heatmap(t5_att_bias[i][:17,:17],ax=ax_list[0],
                  xticklabels=question_text[:17],
                  yticklabels=question_text[:17])
    
    sns.heatmap(context_scores[i][:17,:17],ax=ax_list[1],
                  xticklabels=question_text[:17],
                  yticklabels=question_text[:17])
    
    sns.heatmap(final_scores[i][:17,:17],ax=ax_list[2],
                  xticklabels=question_text[:17],
                  yticklabels=question_text[:17])
  '''
  
  ax1 = fig.add_subplot(611)
  ax2 = fig.add_subplot(612)
  ax3 = fig.add_subplot(613)
  ax4 = fig.add_subplot(614)
  ax5 = fig.add_subplot(615)
  ax6 = fig.add_subplot(616)
  
  print('t5_att_bias:',np.shape(t5_att_bias))
  ax_list =[ax1,ax2,ax3,ax4,ax5,ax6]
  
  for i in range(6):
    if i<5:
      sns.heatmap(final_scores[i,7:8,:18],ax=ax_list[i])
    else:
      sns.heatmap(final_scores[i,7:8,:18],ax=ax_list[i],
                  xticklabels=question_text[:18])
  
  '''
  ax1 = fig.add_subplot(811)
  ax2 = fig.add_subplot(812)
  ax3 = fig.add_subplot(813)
  ax4 = fig.add_subplot(814)
  ax5 = fig.add_subplot(815)
  ax6 = fig.add_subplot(816)
  ax7 = fig.add_subplot(817)
  ax8 = fig.add_subplot(818)
  ax_list=[ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8]
  
  for i in range(8):
    sns.heatmap(t5_att_bias[i,:,:],ax=ax_list[i])
    with open(dir_path+'/csv/'+model_name+'-'+task+'-hm-prior-h'+str(i),'w',newline='') as csvfile: 
      writer=csv.writer(csvfile,delimiter=',')
      length = len(t5_att_bias[i,0])
      for idx_row in range(length):
        for idx_col in range(length):
          writer.writerow([idx_row,idx_col,t5_att_bias[i,idx_row,idx_col]])
          
    with open(dir_path+'/csv/'+model_name+'-'+task+'-hm-ctx-h'+str(i),'w',newline='') as csvfile: 
      writer=csv.writer(csvfile,delimiter=',')
      length = len(context_scores[i,0])
      for idx_row in range(length):
        for idx_col in range(length):
          writer.writerow([idx_row,idx_col,context_scores[i,idx_row,idx_col]])
    
    with open(dir_path+'/csv/'+model_name+'-'+task+'-hm-final-h'+str(i),'w',newline='') as csvfile: 
      writer=csv.writer(csvfile,delimiter=',')
      length = len(final_scores[i,0])
      for idx_row in range(length):
        for idx_col in range(length):
          writer.writerow([idx_row,idx_col,final_scores[i,idx_row,idx_col]])
  '''
  '''
  ax1 = fig.add_subplot(611)
  ax2 = fig.add_subplot(612)
  ax3 = fig.add_subplot(613)
  ax4 = fig.add_subplot(614)
  ax5 = fig.add_subplot(615)
  ax6 = fig.add_subplot(616)
  ax_list=[ax1,ax2,ax3,ax4,ax5,ax6]
  print(question_text)
  
  for i in range(6):
    sns.heatmap(final_scores[i,:17,:17],ax=ax_list[i],
                xticklabels=question_text[:17])
  '''
  w_pos = 7
  for i in range(6):
    with open(dir_path+'/csv/'+model_name+'-'+task+'-token-'+str(w_pos)+'-bar-prior-h'+str(i),'w',newline='') as csvfile: 
      writer=csv.writer(csvfile,delimiter=',')
      length = 18
      
      prior = t5_att_bias[i,w_pos]
      prior_norm = np.exp(prior)/np.sum(np.exp(prior))
      #print('prior_norm:',prior_norm)
      for idx_col in range(length):
        writer.writerow([question_text[idx_col],prior[idx_col]])
    
    with open(dir_path+'/csv/'+model_name+'-'+task+'-token-'+str(w_pos)+'-bar-ctx-h'+str(i),'w',newline='') as csvfile: 
      writer=csv.writer(csvfile,delimiter=',')
      length = 18
      
      ctx = context_scores[i,w_pos]
      ctx_norm = np.exp(ctx)/np.sum(np.exp(ctx))
      
      for idx_col in range(length):
        writer.writerow([question_text[idx_col],ctx[idx_col]])
    
    with open(dir_path+'/csv/'+model_name+'-'+task+'-token-'+str(w_pos)+'-bar-final-h'+str(i),'w',newline='') as csvfile: 
      writer=csv.writer(csvfile,delimiter=',')
      length = 18
      
      final = final_scores[i,w_pos]
      final_norm = np.exp(final)/np.sum(np.exp(final))
      
      for idx_col in range(length):
        writer.writerow([question_text[idx_col],final[idx_col]])
  
elif model_name =='T5_PE' or model_name=='T5_PE_NoB':
  #0,2,5
  dir_path = task+'/'+model_name+'/'
  print(model_name)
  
  ret = pickle.load(open(dir_path + 'test_ret.p','rb'))
  print(ret)
  '''
  t5_att_bias = pickle.load(open(dir_path+'t5_att_bias.p','rb'))
  
  seq_lent = np.shape(t5_att_bias['left'])[1]
  relative_position = np.arange(seq_lent)[None,:] - np.arange(seq_lent)[:,None]
  
  x_idx=list(relative_position[-1,:])+list(relative_position[0,1:])
  
  att_dict={}
  
  for i in range(6):
    weights =  list(t5_att_bias['left'][i,:]) + list(t5_att_bias['right'][i,1:])
    
    att = np.exp(weights)/sum(np.exp(weights))
    #print(np.amax(att))
    #print(np.amax(weights))
    #print('---------------')
    att_dict[i]=np.amax(att)
    
    with open(dir_path+'/csv/'+model_name+'-'+task+'-prior-h'+str(i),'w',newline='') as csvfile: 
      writer=csv.writer(csvfile,delimiter=',')
      for idx in range(len(x_idx)):
        writer.writerow([x_idx[idx],att[idx]])
    
    plt.plot(x_idx,att,label='h-'+str(i))
    plt.legend(loc='upper right', shadow=True,fontsize='x-large')
    plt.show()
    
  print(sorted(att_dict.items(),key=lambda item:item[1],reverse=True))
  '''
  
  save_param=pickle.load(open(dir_path+model_name+'-'+task+'-att.pkl','rb'))
  question_text=save_param['question_text']
  input_y=save_param['input_y']
  predict_y=save_param['predict_y']
  
  print(np.shape(context_scores))
  print('----------------')
  context_scores=save_param['context_scores'][1]
  final_scores=save_param['final_scores'][1]
  att_mask=save_param['att_mask'][1]
  t5_att_bias=save_param['t5_att_bias'][0]
  
  '''
  for i in range(6):
    #sns.heatmap(t5_att_bias[i,:,:],ax=ax_list[i])
    
    with open(dir_path+'/csv/'+model_name+'-'+task+'-hm-prior-h'+str(i),'w',newline='') as csvfile: 
      writer=csv.writer(csvfile,delimiter=',')
      length = len(t5_att_bias[i,0])
      for idx_row in range(length):
        for idx_col in range(length):
          writer.writerow([idx_row,idx_col,t5_att_bias[i,idx_row,idx_col]])
    
    with open(dir_path+'/csv/'+model_name+'-'+task+'-hm-ctx-h'+str(i),'w',newline='') as csvfile: 
      writer=csv.writer(csvfile,delimiter=',')
      length = len(context_scores[i,0])
      for idx_row in range(length):
        for idx_col in range(length):
          writer.writerow([idx_row,idx_col,context_scores[i,idx_row,idx_col]])
    
    with open(dir_path+'/csv/'+model_name+'-'+task+'-hm-final-h'+str(i),'w',newline='') as csvfile: 
      writer=csv.writer(csvfile,delimiter=',')
      length = len(final_scores[i,0])
      for idx_row in range(length):
        for idx_col in range(length):
          writer.writerow([idx_row,idx_col,final_scores[i,idx_row,idx_col]])
  '''
  #fig = plt.figure()
  '''
  ax1 = fig.add_subplot(811)
  ax2 = fig.add_subplot(812)
  ax3 = fig.add_subplot(813)
  ax4 = fig.add_subplot(814)
  ax5 = fig.add_subplot(815)
  ax6 = fig.add_subplot(816)
  #ax7 = fig.add_subplot(817)
  #ax8 = fig.add_subplot(818)
  ax_list=[ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8]
  for i in range(8):
    ax = ax_list[i]
    sns.heatmap(final_scores[i,:,:],ax=ax)'''
  '''
  ax1 = fig.add_subplot(311)
  ax2 = fig.add_subplot(312)
  ax3 = fig.add_subplot(313)
  
  print('t5_att_bias:',np.shape(t5_att_bias))
  ax_list =[ax1,ax2,ax3]
  
  for i in [0]:
    sns.heatmap(t5_att_bias[i,:17,:17],ax=ax_list[0],
                  xticklabels=question_text[:17])
    
    sns.heatmap(context_scores[i,:17,:17],ax=ax_list[1],
                  xticklabels=question_text[:17])
    
    sns.heatmap(final_scores[i,:17,:17],ax=ax_list[2],
                  xticklabels=question_text[:17])
  '''
  '''
  ax1 = fig.add_subplot(611)
  ax2 = fig.add_subplot(612)
  ax3 = fig.add_subplot(613)
  ax4 = fig.add_subplot(614)
  ax5 = fig.add_subplot(615)
  ax6 = fig.add_subplot(616)
  ax_list=[ax1,ax2,ax3,ax4,ax5,ax6]'''
  print(question_text)
  '''
  for i in range(6):
    if i<5:
      sns.heatmap(t5_att_bias[i,11:12,:17],ax=ax_list[i])
    else:
      sns.heatmap(t5_att_bias[i,11:12,:17],ax=ax_list[i],
                  xticklabels=question_text[:17])
  '''
  #print(t5_att_bias[3,11:12,:17])
  #print(context_scores[3,11:12,:17])
  #print(final_scores[3,11:12,:17])
  '''
  for i in range(6):
    with open(dir_path+'/csv/'+model_name+'-'+task+'-token-11-bar-prior-h'+str(i),'w',newline='') as csvfile: 
      writer=csv.writer(csvfile,delimiter=',')
      length = 18
      
      prior = t5_att_bias[i,11,:18]
      prior_norm = np.exp(prior)/np.sum(np.exp(prior))
      print('prior_norm:',prior_norm)
      for idx_col in range(length):
        writer.writerow([idx_col,prior_norm[idx_col]])
    
    with open(dir_path+'/csv/'+model_name+'-'+task+'-token-11-bar-ctx-h'+str(i),'w',newline='') as csvfile: 
      writer=csv.writer(csvfile,delimiter=',')
      length = 18
      
      ctx = context_scores[i,11,:18]
      ctx_norm = np.exp(ctx)/np.sum(np.exp(ctx))
      
      for idx_col in range(length):
        writer.writerow([idx_col,ctx_norm[idx_col]])
    
    with open(dir_path+'/csv/'+model_name+'-'+task+'-token-11-bar-final-h'+str(i),'w',newline='') as csvfile: 
      writer=csv.writer(csvfile,delimiter=',')
      length = 18
      
      final = final_scores[i,11,:18]
      final_norm = np.exp(final)/np.sum(np.exp(final))
      
      for idx_col in range(length):
        writer.writerow([idx_col,final_norm[idx_col]])
          
  '''
elif model_name=='TPE':
  dir_path = task+'/TPE/'
  
  save_param=pickle.load(open(dir_path+'TPE_reduce-'+task+'-att.pkl','rb'))
  question_text=save_param['question_text']
  input_y=save_param['input_y']
  predict_y=save_param['predict_y']
  
  context_scores=save_param['context_scores'][1]
  final_scores=save_param['final_scores'][1]
  att_mask=save_param['att_mask'][1]
  t5_att_bias=save_param['t5_att_bias']
  
  fig = plt.figure()
  #ax1 = fig.add_subplot(811)
  #ax2 = fig.add_subplot(812)
  #ax3 = fig.add_subplot(813)
  #ax4 = fig.add_subplot(814)
  #ax5 = fig.add_subplot(815)
  #ax6 = fig.add_subplot(816)
  #ax7 = fig.add_subplot(817)
  #ax8 = fig.add_subplot(818)
  
  for i in range(8):
    with open(dir_path+'/csv/'+model_name+'-'+task+'-hm-final-h'+str(i),'w',newline='') as csvfile: 
      writer=csv.writer(csvfile,delimiter=',')
      length = len(context_scores[i,0])
      for idx_row in range(length):
        for idx_col in range(length):
          writer.writerow([idx_row,idx_col,final_scores[i,idx_row,idx_col]])
  
  '''
  ax1 = fig.add_subplot(611)
  ax2 = fig.add_subplot(612)
  ax3 = fig.add_subplot(613)
  ax4 = fig.add_subplot(614)
  ax5 = fig.add_subplot(615)
  ax6 = fig.add_subplot(616)
  ax_list=[ax1,ax2,ax3,ax4,ax5,ax6]
  print(question_text)
  
  for i in range(6):
    sns.heatmap(final_scores[i,:17,:17],ax=ax_list[i],
                xticklabels=question_text[:17])'''