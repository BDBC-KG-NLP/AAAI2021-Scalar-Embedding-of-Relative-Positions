#!/bin/bash

export PYTHONPATH=`readlink -f .`:$PYTHONPATH

trail=${1}
num_buckets=64
max_distance=128
att_drop=0.1

output=ckpt-t5/t5-nob-en2de-${num_buckets}-${max_distance}-${att_drop}-ckpt_dev-1_cycle-2-${trail}


for year in {2014,2015,2016,2017}
do
{
  for decode_alpha in 0.6
  do
  {
    python thumt/bin/translator.py\
       --models transformer_raw_t5_nob \
       --input \
           ./wmt17/newstest${year}.tc.32k.en \
       --output \
           ./wmt17/newstest${year}.trans.en2de \
       --vocabulary \
           ./wmt17/vocab.32k.en.txt \
           ./wmt17/vocab.32k.de.txt \
       --checkpoints ${output}/eval \
       --parameters \
       "device_list=[0],shared_embedding_and_softmax_weights=true,layer_preprocess=layer_norm,layer_postprocess=none,attention_dropout=${att_drop},relu_dropout=0.1,adam_beta2=0.98,num_buckets=${num_buckets},max_distance=${max_distance}" \
    
    sed -r 's/(@@ )|(@@ ?$)//g' < ./wmt17/newstest${year}.trans.en2de > ./wmt17/newstest${year}.trans.en2de.norm
    
    perl multi-bleu.perl -lc ./wmt17/newstest${year}.tc.de < ./wmt17/newstest${year}.trans.en2de.norm 
  }
  done
}
done