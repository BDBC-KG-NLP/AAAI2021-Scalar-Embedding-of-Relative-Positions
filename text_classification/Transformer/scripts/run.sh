#!/usr/bin/env bash


model_name='Non_PE'

log_dir=ret-3
if [ ! -d ${log_dir} ]
then 
  echo " ${log_dir} is not exist"
  mkdir -p ${log_dir}
fi

:'
training_nums=20000
embedding_dim=300
lr=2e-4
transformer_ret_pooling=last
for data in 'TREC'
do
{
  for num_attention_heads in 6
  do
  {
    for  num_hidden_layers in 5
    do
    {
      for trail in {1,2,3,4,5}
      do
      {
        python train.py \
                --model_name='Non_PE' \
                --transformer_ret_pooling=${transformer_ret_pooling} \
                --data=${data} \
                --num_epochs=60 \
                --num_hidden_layers=${num_hidden_layers} \
                --embedding_dim=${embedding_dim} \
                --num_attention_heads=${num_attention_heads} \
                --is_Embedding_Needed=True \
                --training_nums=${training_nums} \
                --trail=${trail} \
                --learning_rate=${lr}
      }
      done
    }
     done
  }
  done
}
done
'

model_name='Soft_T5_PE_NoB'
log_dir=ret-3
if [ ! -d ${log_dir} ]
then 
  echo " ${log_dir} is not exist"
  mkdir -p ${log_dir}
fi

:'
training_nums=20000
embedding_dim=300
num_hidden_layers=5
num_attention_heads=6
transformer_ret_pooling=last
lr=2e-4
for data in 'TREC'
do
{
  for bucket_slop_min in 1.0
  do
  {
    for bucket_slop_max in 10.0
    do
    {
      for trail in {1,2,3,4,5}
      do
      {
         python train.py \
      --model_name=${model_name} \
      --transformer_ret_pooling=${transformer_ret_pooling} \
      --data=${data} \
      --num_epochs=100 \
      --num_hidden_layers=${num_hidden_layers} \
      --embedding_dim=300 \
      --num_attention_heads=${num_attention_heads} \
      --is_Embedding_Needed=True \
      --trail=${trail} \
      --training_nums=${training_nums} \
      --learning_rate=${lr} \
      --bucket_slop_min=${bucket_slop_min} \
      --bucket_slop_max=${bucket_slop_max} \
      --is_training=True #> ${log_dir}/${model_name}_${data}_L-${num_hidden_layers}_H-${num_attention_heads}_W-${embedding_dim}-last-${training_nums}-${bucket_slop_min}-${bucket_slop_max}-${trail} &
      }
      done
    }
    done
  }
  done
}
done
'

model_name='Non_PE'

log_dir=ret-3
if [ ! -d ${log_dir} ]
then 
  echo " ${log_dir} is not exist"
  mkdir -p ${log_dir}
fi


training_nums=1000
embedding_dim=256
lr=5e-4
for num_attention_heads in 8
do
{
 for trail in 1
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
                       --trail=${trail} \
                       --is_training=True #> ${log_dir}/${model_name}_reber_L-1_H-${num_attention_heads}-${training_nums}-${lr}-${trail} &
 }
 done
}
done