#!/usr/bin/env bash
#{'TREC','mr','subj','cr','mpqa','sst2'}
model_name='Soft_T5_PE_NoB'
log_dir=ret-3
if [ ! -d ${log_dir} ]
then 
  echo " ${log_dir} is not exist"
  mkdir -p ${log_dir}
fi


training_nums=20000
embedding_dim=300
num_hidden_layers=5
num_attention_heads=6
transformer_ret_pooling=last
lr=2e-4
for data in {'mr','subj','cr','mpqa','TREC','sst2'}
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

#--------------reber-----------------#

training_nums=1000
embedding_dim=256
lr=5e-4

for bucket_slop_min in 0.1
do
{
  for bucket_slop_max in 10.0
  do
  {
    for trail in {1,2,3,4,5}
    do
    {
      for num_attention_heads in 8
      do
      {
         python train_cls.py \
                             --model_name=${model_name} \
                             --data=reber \
                             --num_epochs=1000 \
                             --transformer_ret_pooling=last \
                             --embedding_dim=${embedding_dim} \
                             --num_hidden_layers=1 \
                             --training_nums=${training_nums} \
                             --learning_rate=${lr} \
                             --num_attention_heads=${num_attention_heads} \
                             --bucket_slop_min=${bucket_slop_min}\
                             --bucket_slop_max=${bucket_slop_max}\
                             --trail=${trail}\
                             --is_training=True #> ${log_dir}/${model_name}_reber_L-1_H-${num_attention_heads}-${training_nums}-${lr}-${bucket_slop_min}-${bucket_slop_max}-${trail}&
      }
      done
    }
    done
  }
  done
}
done



#--------------process_cls_50---------------#
log_dir=ret-3

model_name='Soft_T5_PE'
transformer_ret_pooling='mean'
training_nums=5000
embedding_dim=256
num_hidden_layers=1

for trail in {2,3,4,5}
do
{
  for num_attention_heads in 8
  do
  {
    python train_cls.py \
                            --model_name=${model_name} \
                            --data=process_cls_50 \
                            --num_epochs=100 \
                            --learning_rate=2e-4 \
                            --embedding_dim=${embedding_dim} \
                            --num_hidden_layers=${num_hidden_layers} \
                            --training_nums=${training_nums} \
                            --num_attention_heads=${num_attention_heads} \
                            --trail=${trail}\
                            --transformer_ret_pooling=${transformer_ret_pooling}  #>${log_dir}/${model_name}_process_cls_50_L-${num_hidden_layers}_H-${num_attention_heads}-${transformer_ret_pooling}-${trail} &
  }
  done
}
done


#----------- Adding_problem----------------#

for trail in {1,2,3,4,5}
do
{
  for seq_lent in 100
  do
  {
   for num_attention_heads in 8
   do
   {
     for training_nums in 1000
     do
     {
      python train_cls.py \
                  --model_name=${model_name} \
                  --data=adding_problem_${seq_lent} \
                  --num_epochs=300 \
                  --embedding_dim=256 \
                  --num_hidden_layers=1 \
                  --trainable=False\
                  --training_nums=${training_nums} \
                  --num_attention_heads=${num_attention_heads}\
                  --transformer_ret_pooling=last\
                  --trail=${trail} #> ${log_dir}/${model_name}_adding_problem_${seq_lent}_L-1_H-${num_attention_heads}-${training_nums}-${trail}-last &
      }
      done
   }
   done
  }
  done
}
done

