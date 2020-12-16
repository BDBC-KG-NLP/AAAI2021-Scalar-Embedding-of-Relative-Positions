#!/bin/bash

export PYTHONPATH=`readlink -f .`:$PYTHONPATH

export CUDA_VISIBLE_DEVICES=0

stddev=0.02

for trail in {1,2,3,4,5}
do
{
  for bucket_slop_min in 1
  do
  {
    for bucket_slop_max in 15
    do
    {
      for att_drop in 0.1
      do
      {
        for l1_width in {16,32}
        do
        {
          for l2_width in 4
          do
          {
            output=ckpt-t5/soft-t5-${bucket_slop_min}-${bucket_slop_max}-${att_drop}-${l1_width}-${l2_width}-${stddev}-en2de_ckpt_dev-1_cycle-2-${trail}
                
            if [ ! -d ${output} ]
            then
              mkdir ${output}
            fi
                    
            nohup srun --exclude=dell-gpu-31 --gres=gpu:1 python thumt/bin/trainer.py\
                    --model transformer_raw_soft_t5 \
                    --input \
                        ./wmt17/corpus.tc.32k.en.shuf \
                        ./wmt17/corpus.tc.32k.de.shuf \
                    --validation \
                        ./wmt17/newstest2014.tc.32k.en\
                    --references \
                        ./wmt17/newstest2014.tc.de\
                    --output \
                        ${output} \
                    --vocabulary \
                        ./wmt17/vocab.32k.en.txt \
                        ./wmt17/vocab.32k.de.txt \
                    --parameters \
                        batch_size=12500,device_list=[0],update_cycle=2,eval_steps=5000,train_steps=250000,shared_embedding_and_softmax_weights=true,layer_preprocess=layer_norm,layer_postprocess=none,attention_dropout=${att_drop},relu_dropout=0.1,adam_beta2=0.98,bucket_slop_min=${bucket_slop_min},bucket_slop_max=${bucket_slop_max},l1_width=${l1_width},l2_width=${l2_width},stddev=${stddev} > ret-t5/soft-t5-en2de-en2de-${bucket_slop_min}-${bucket_slop_max}-${att_drop}-${l1_width}-${l2_width}-${stddev}_ckpt_dev-1_cycle-2-${trail}  &
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
}
done