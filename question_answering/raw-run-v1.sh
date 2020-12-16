for head in 8
do
{
  for trail in {1,2,3,4,5}
  do
  {
    #nohup python config-v1.py "train" "Raw" ${head} ${trail} > ret/train-raw-h${head}-${trail}-v1 &
    python config-v1.py "train" "Raw" ${head} ${trail}
  }
  done
}
done

:'
for head in 8
do
{
  for trail in {1,2,3,4,5}
  do
  {
    nohup python config-v1.py "test" "Raw" ${head} ${trail} > ret/test-raw-h${head}-${trail}-v1 &
  }
  done
}
done
'