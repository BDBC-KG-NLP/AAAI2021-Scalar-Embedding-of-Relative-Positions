#!/bin/bash

export PYTHONPATH=`readlink -f .`:$PYTHONPATH


#1 0.1 40
trail=$1
att_drop=$2
soft_t5_width=$3

output=ckpt-t5/soft-t5-nob-${soft_t5_width}-${att_drop}-en2de_ckpt_dev-1_cycle-2-${trail}

for year in 2014
do
{
  for decode_alpha in {2014,2015,2016,2017}
  do
  {
    python thumt/bin/translator.py\
       --models transformer_raw_soft_t5_nob \
       --input \
           ./wmt17/newstest${year}.tc.32k.en \
       --output \
           ./wmt17/newstest${year}.trans.en2de \
       --vocabulary \
           ./wmt17/vocab.32k.en.txt \
           ./wmt17/vocab.32k.de.txt \
       --checkpoints ${output}/eval \
       --parameters \
          device_list=[0],shared_embedding_and_softmax_weights=true,layer_preprocess=layer_norm,layer_postprocess=none,attention_dropout=${att_drop},relu_dropout=0.1,adam_beta2=0.98,soft_t5_width=${soft_t5_width} \
    
    sed -r 's/(@@ )|(@@ ?$)//g' < ./wmt17/newstest${year}.trans.en2de > ./wmt17/newstest${year}.trans.en2de.norm

    perl multi-bleu.perl -lc ./wmt17/newstest${year}.tc.de < ./wmt17/newstest${year}.trans.en2de.norm 
  }
  done
}
done