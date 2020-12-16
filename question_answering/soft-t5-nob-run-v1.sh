#2020/8/29
#python config-v1.py "train" "Soft_T5_Nob" 8 400 0.001 relu 1

for head in 8
do
{
  for fixed_c_maxlen in 400
  do
  {
    for learning_rate in 0.001
    do
    {
      for soft_t5_activation in relu
      do
      {
        for trail in {1,2,3,4,5}
        do
        {
           python config-v1.py "train" "Soft_T5_Nob" ${head} ${fixed_c_maxlen} ${learning_rate} ${soft_t5_activation} ${trail} #> ret/train-soft-t5-Nob-h${head}_c-${fixed_c_maxlen}_lr-${learning_rate}-${soft_t5_activation}-${trail}-v1 &
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

for head in 8
do
{
  for fixed_c_maxlen in 400
  do
  {
    for learning_rate in 0.001
    do
    {
      for soft_t5_activation in relu
      do
      {
        for trail in {1,2,3,4,5}
        do
        {
           python config-v1.py "test" "Soft_T5_Nob" ${head} ${fixed_c_maxlen} ${learning_rate} ${soft_t5_activation} ${trail} #> ret/test-soft-t5-Nob-h${head}_c-${fixed_c_maxlen}_lr-${learning_rate}-${soft_t5_activation}-${trail}-v1 &
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