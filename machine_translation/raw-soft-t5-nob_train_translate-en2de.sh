#!/bin/bash

export PYTHONPATH=`readlink -f .`:$PYTHONPATH


export CUDA_VISIBLE_DEVICES=0

for trail in {1,2,3,4,5}
do
{
  for att_drop in 0.1
  do
  {
    for soft_t5_width in 40
    do
    {
      output=ckpt-t5/soft-t5-nob-${soft_t5_width}-${att_drop}-en2de_ckpt_dev-1_cycle-2-${trail}
              
      if [ ! -d ${output} ]
      then
        mkdir ${output}
      fi
      
      python thumt/bin/trainer.py\
              --model transformer_raw_soft_t5_nob \
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
                  batch_size=12500,device_list=[0],update_cycle=2,eval_steps=5000,train_steps=250000,shared_embedding_and_softmax_weights=true,layer_preprocess=layer_norm,layer_postprocess=none,attention_dropout=${att_drop},relu_dropout=0.1,adam_beta2=0.98,soft_t5_width=${soft_t5_width} #> ret-t5/soft-t5-nob-en2de-${soft_t5_width}-${att_drop}-_ckpt_dev-1_cycle-2-${trail}  &
    }
    done
  }
  done
}
done