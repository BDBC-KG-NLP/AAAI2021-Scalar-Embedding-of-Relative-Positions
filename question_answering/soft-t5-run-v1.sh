#2020/7/25
#{3.0,5.0,10.0}
#{10.0,15.0,20.0}
:'
for head in 8
do
{
  for fixed_c_maxlen in 400
  do
  {
    for learning_rate in 0.001
    do
    {
      for bucket_slop_min in {3.0,5.0,10.0}
      do
      {
        for bucket_slop_max in {10.0,15.0,20.0}
        do
        {
          for soft_t5_activation in relu
          do
          {
            for trail in 1
            do
            {
               nohup python config-v1.py "train" "Soft_T5" ${head} ${fixed_c_maxlen} ${learning_rate} ${bucket_slop_min} ${bucket_slop_max} ${soft_t5_activation} ${trail} > ret/train-soft-t5-poly_ln_h${head}_c-${fixed_c_maxlen}_lr-${learning_rate}-${bucket_slop_min}-${bucket_slop_max}-${soft_t5_activation}-${trail}-v1 &
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
}
done
'

l1_width=32
l2_width=4
stddev=0.02


for head in 8
do
{
  for fixed_c_maxlen in 400
  do
  {
    for learning_rate in 0.001
    do
    {
      for bucket_slop_min in 5.0
      do
      {
        for bucket_slop_max in 15.0
        do
        {
          for soft_t5_activation in relu
          do
          {
            for trail in {0,1}
            do
            {
              nohup srun --exclude=dell-gpu-31,dell-gpu-20 python config-v1.py "train" "Soft_T5" ${head} ${fixed_c_maxlen} ${learning_rate} ${bucket_slop_min} ${bucket_slop_max} ${l1_width} ${l2_width} ${stddev} ${soft_t5_activation} ${trail} > ret/train-soft-t5-poly_ln_h${head}_c-${fixed_c_maxlen}_lr-${learning_rate}-${bucket_slop_min}-${bucket_slop_max}-${l1_width}-${l2_width}-${soft_t5_activation}-${trail}-v1 &
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
}
done
'''
l1_width=32
l2_width=8
stddev=0.02


for head in 8
do
{
  for fixed_c_maxlen in 400
  do
  {
    for learning_rate in 0.001
    do
    {
      for bucket_slop_min in 5.0
      do
      {
        for bucket_slop_max in 15.0
        do
        {
          for soft_t5_activation in relu
          do
          {
            for trail in {0,1}
            do
            {
              nohup srun --exclude=dell-gpu-31,dell-gpu-20 python config-v1.py "train" "Soft_T5" ${head} ${fixed_c_maxlen} ${learning_rate} ${bucket_slop_min} ${bucket_slop_max} ${l1_width} ${l2_width} ${stddev} ${soft_t5_activation} ${trail} > ret/train-soft-t5-poly_ln_h${head}_c-${fixed_c_maxlen}_lr-${learning_rate}-${bucket_slop_min}-${bucket_slop_max}-${l1_width}-${l2_width}-${soft_t5_activation}-${trail}-v1 &
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
}
done

l1_width=64
l2_width=4
stddev=0.02


for head in 8
do
{
  for fixed_c_maxlen in 400
  do
  {
    for learning_rate in 0.001
    do
    {
      for bucket_slop_min in 5.0
      do
      {
        for bucket_slop_max in 15.0
        do
        {
          for soft_t5_activation in relu
          do
          {
            for trail in {0,1}
            do
            {
              nohup srun --exclude=dell-gpu-31,dell-gpu-20 python config-v1.py "train" "Soft_T5" ${head} ${fixed_c_maxlen} ${learning_rate} ${bucket_slop_min} ${bucket_slop_max} ${l1_width} ${l2_width} ${stddev} ${soft_t5_activation} ${trail} > ret/train-soft-t5-poly_ln_h${head}_c-${fixed_c_maxlen}_lr-${learning_rate}-${bucket_slop_min}-${bucket_slop_max}-${l1_width}-${l2_width}-${soft_t5_activation}-${trail}-v1 &
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
}
done


l1_width=64
l2_width=8
stddev=0.02


for head in 8
do
{
  for fixed_c_maxlen in 400
  do
  {
    for learning_rate in 0.001
    do
    {
      for bucket_slop_min in 5.0
      do
      {
        for bucket_slop_max in 15.0
        do
        {
          for soft_t5_activation in relu
          do
          {
            for trail in {0,1}
            do
            {
              nohup srun --exclude=dell-gpu-31,dell-gpu-20 python config-v1.py "train" "Soft_T5" ${head} ${fixed_c_maxlen} ${learning_rate} ${bucket_slop_min} ${bucket_slop_max} ${l1_width} ${l2_width} ${stddev} ${soft_t5_activation} ${trail} > ret/train-soft-t5-poly_ln_h${head}_c-${fixed_c_maxlen}_lr-${learning_rate}-${bucket_slop_min}-${bucket_slop_max}-${l1_width}-${l2_width}-${soft_t5_activation}-${trail}-v1 &
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
}
done
'''
:'
for head in 8
do
{
  for fixed_c_maxlen in 400
  do
  {
    for learning_rate in 0.001
    do
    {
      for bucket_slop_min in 5.0
      do
      {
        for bucket_slop_max in 15.0
        do
        {
          for soft_t5_activation in relu
          do
          {
            for trail in {2,3,4,5}
            do
            {
               nohup srun --gres=gpu:1 python config-v1.py "test" "Soft_T5" ${head} ${fixed_c_maxlen} ${learning_rate} ${bucket_slop_min} ${bucket_slop_max} ${soft_t5_activation} ${trail} > ret/test-soft-t5-poly_ln_h${head}_c-${fixed_c_maxlen}_lr-${learning_rate}-${bucket_slop_min}-${bucket_slop_max}-${soft_t5_activation}-${trail}-v1 &
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
}
done
'