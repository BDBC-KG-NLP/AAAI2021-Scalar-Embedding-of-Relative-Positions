#!/usr/bin/env bash
#{'TREC','mr','subj','cr','mpqa','sst2'}
model_name='T5_PE'
##data=sst2
##data=TREC
log_dir=ret-3
if [ ! -d ${log_dir} ]
then 
  echo " ${log_dir} is not exist"
  mkdir -p ${log_dir}
fi 


#-----------text classification---------------#
training_nums=20000
embedding_dim=300

lr=2e-4

#{'mr','subj','cr','mpqa','sst2','TREC'}
for data in  TREC
do
{
  for num_attention_heads in 6
  do
  {
    for num_hidden_layers in 5
    do
    {
      for trail in 1
      do
      {
         python train.py \
      --model_name=${model_name} \
      --transformer_ret_pooling=last \
      --data=${data} \
      --num_epochs=100 \
      --num_hidden_layers=${num_hidden_layers} \
      --embedding_dim=${embedding_dim} \
      --num_attention_heads=${num_attention_heads} \
      --is_Embedding_Needed=True \
      --trail=${trail} \
      --training_nums=${training_nums} \
      --learning_rate=${lr} \
      --is_training=False #> ${log_dir}/${model_name}_${data}_L-${num_hidden_layers}_H-${num_attention_heads}-${embedding_dim}-last-${training_nums}-${lr}-${trail} &
      }
      done
    }
    done
  }
  done
}
done

#-------------------reber----------------#
:'
training_nums=1000
embedding_dim=256
lr=5e-4
for trail in 1
do
{
  for num_attention_heads in 8
  do
  {
   for t5_bucket in 32
   do
   {
     python train_cls.py \
                  --model_name=${model_name} \
                  --data=reber \
                  --num_epochs=500 \
                  --learning_rate=${lr} \
                  --transformer_ret_pooling=last \
                  --embedding_dim=${embedding_dim} \
                  --num_hidden_layers=1 \
                  --training_nums=${training_nums} \
                  --num_attention_heads=${num_attention_heads} \
                  --t5_bucket=${t5_bucket} \
                  --trail=${trail} \
                  --is_training=True #> ${log_dir}/${model_name}_reber_L-1_H-${num_attention_heads}-${training_nums}-${lr}-${t5_bucket}-${trail} &
   }
   done
  }
  done
}
done
'

#-------------Process-50-----------------------#
:'
transformer_ret_pooling='mean'
training_nums=5000
embedding_dim=256
num_hidden_layers=1
num_attention_heads=8

for trail in 1
do
{
  for t5_bucket in 64
  do
  {
    python train_cls.py \
                            --model_name=${model_name} \
                            --data=process_cls_50 \
                            --num_epochs=100 \
                            --learning_rate=1e-4 \
                            --embedding_dim=${embedding_dim} \
                            --num_hidden_layers=${num_hidden_layers} \
                            --training_nums=${training_nums} \
                            --num_attention_heads=${num_attention_heads} \
                            --t5_bucket=${t5_bucket} \
                            --trail=${trail}\
                            --transformer_ret_pooling=${transformer_ret_pooling}  #> ${log_dir}/${model_name}_process_cls_50_L-${num_hidden_layers}_H-${num_attention_heads}-${transformer_ret_pooling}-${t5_bucket}-${trail} &
  }
  done
}
done

#----------------------Adding-100-----------------#

embedding_dim=256
training_nums=1000
for trail in 1
do
{
  for num_attention_heads in 8
  do
  {
    for seq_lent in 500
    do
    {
     for t5_bucket in 32
     do
     {
       for  num_hidden_layers in 1
       do
       {
         python train_cls.py \
                         --model_name=${model_name} \
                         --data=adding_problem_${seq_lent} \
                         --num_epochs=300 \
                         --embedding_dim=${embedding_dim} \
                         --num_hidden_layers=1 \
                         --training_nums=${training_nums} \
                         --t5_bucket=${t5_bucket} \
                         --trainable=False\
                         --transformer_ret_pooling=last\
                         --num_attention_heads=${num_attention_heads} #> ${log_dir}/${model_name}_temporal_order_a_${seq_lent}_L-${num_hidden_layers}_H-${num_attention_heads}-${t5_bucket}-${training_nums}-${trail} &
       }
       done
     }
     done
    }
    done
  }
  done
}
done
'
