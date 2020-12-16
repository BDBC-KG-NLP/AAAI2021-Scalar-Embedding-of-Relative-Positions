for head in 8
do
{
  for t5_num_buckets in 32
  do
  {
    for t5_max_distance in 128
    do
    {
      for trail in 3
      do
      {
         python config-v1.py "train" "T5" ${head} ${t5_num_buckets} ${t5_max_distance} ${trail}
      }
      done
    }
    done
  }
  done
}
done

:'
for head in 8
do
{
  for t5_num_buckets in 64
  do
  {
    for t5_max_distance in 256
    do
    {
      for trail in {2,3,4,5}
      do
      {
        nohup srun --gres=gpu:1 python config-v1.py "train" "T5" ${head} ${t5_num_buckets} ${t5_max_distance} ${trail}> ret/train-t5-h${head}-${t5_num_buckets}-${t5_max_distance}-${trail}-v1 &
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
  for t5_num_buckets in {64,128}
  do
  {
    for t5_max_distance in 512
    do
    {
      for trail in {2,3,4,5}
      do
      {
        nohup srun --gres=gpu:1 python config-v1.py "train" "T5" ${head} ${t5_num_buckets} ${t5_max_distance} ${trail} > ret/train-t5-h${head}-${t5_num_buckets}-${t5_max_distance}-${trail}-v1 &
      }
      done
    }
    done
  }
  done
}
done
'

:'
for head in 8
do
{
  for t5_num_buckets in 32
  do
  {
    for t5_max_distance in 128
    do
    {
      for trail in {2,3,4,5}
      do
      {
        nohup srun --gres=gpu:1 python config-v1.py "test" "T5" ${head} ${t5_num_buckets} ${t5_max_distance} ${trail} > ret/test-t5-h${head}-${t5_num_buckets}-${t5_max_distance}-${trail}-v1 &
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
  for t5_num_buckets in 64
  do
  {
    for t5_max_distance in 256
    do
    {
      for trail in {2,3,4,5}
      do
      {
        nohup srun --gres=gpu:1 python config-v1.py "test" "T5" ${head} ${t5_num_buckets} ${t5_max_distance} ${trail}> ret/test-t5-h${head}-${t5_num_buckets}-${t5_max_distance}-${trail}-v1 &
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
  for t5_num_buckets in {64,128}
  do
  {
    for t5_max_distance in 512
    do
    {
      for trail in {2,3,4,5}
      do
      {
        nohup srun --gres=gpu:1 python config-v1.py "test" "T5" ${head} ${t5_num_buckets} ${t5_max_distance} ${trail} > ret/test-t5-h${head}-${t5_num_buckets}-${t5_max_distance}-${trail}-v1 &
      }
      done
    }
    done
  }
  done
}
done
'