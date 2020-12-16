#!/bin/bash

export PYTHONPATH=`readlink -f .`:$PYTHONPATH

l1_width=$1
l2_width=4
stddev=0.02

bucket_slop_min=1
bucket_slop_max=15
trail=$2

output=ckpt-t5/soft-t5-${bucket_slop_min}-${bucket_slop_max}-0.1-${l1_width}-${l2_width}-${stddev}-en2de_ckpt_dev-1_cycle-2-${trail}

for year in $3
do
{
  for decode_alpha in 0.6
  do
  {
    python thumt/bin/translator.py\
       --models transformer_raw_soft_t5 \
       --input \
           ./wmt17/newstest${year}.tc.32k.en \
       --output \
           ./wmt17/newstest${year}.trans.en2de \
       --vocabulary \
           ./wmt17/vocab.32k.en.txt \
           ./wmt17/vocab.32k.de.txt \
       --checkpoints ${output}/eval \
       --parameters \
          device_list=[0],shared_embedding_and_softmax_weights=true,layer_preprocess=layer_norm,layer_postprocess=none,attention_dropout=0.1,relu_dropout=0.1,adam_beta2=0.98,decode_alpha=${decode_alpha},bucket_slop_min=${bucket_slop_min},bucket_slop_max=${bucket_slop_max},l1_width=${l1_width},l2_width=${l2_width},stddev=${stddev} \

    sed -r 's/(@@ )|(@@ ?$)//g' < ./wmt17/newstest${year}.trans.en2de > ./wmt17/newstest${year}.trans.en2de.norm

    perl multi-bleu.perl -lc ./wmt17/newstest${year}.tc.de < ./wmt17/newstest${year}.trans.en2de.norm 
  }
  done
}
done