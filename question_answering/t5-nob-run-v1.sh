for head in 8
do
{
  for t5_num_buckets in 32
  do
  {
    for t5_max_distance in 128
    do
    {
      for trail in {1,2,3,4,5}
      do
      {
        python config-v1.py "train" "T5_Nob" ${head} ${t5_num_buckets} ${t5_max_distance} ${trail} #> ret/train-t5-nob-h${head}-${t5_num_buckets}-${t5_max_distance}-${trail}-v1 &
      }
      done
    }
    done
  }
  done
}
done



#python config-v1.py train T5_Nob 8 32 128 0
for head in 8
do
{
  for t5_num_buckets in 32
  do
  {
    for t5_max_distance in 128
    do
    {
      for trail in 1
      do
      {
        nohup python config-v1.py "test" "T5_Nob" ${head} ${t5_num_buckets} ${t5_max_distance} ${trail} > ret/test-t5-nob-h${head}-${t5_num_buckets}-${t5_max_distance}-${trail}-v1 &
      }
      done
    }
    done
  }
  done
}
done