# Question Answering on SQuAD V1 

## Data Processing

- download SQuAD and Glove

```
bash download.sh
```

- preprocess the data

```
python raw_config.py --mode prepro
```

## train and test raw (TPE)

```
bash raw-run-v1.sh  

(python config-v1.py --mode test)
(python config-v1.py --mode train)
```

## train and test T5

```
bash t5-run-v1.sh
bash t5-nob-run-v1.sh
```

## to train and test AT5 (soft-t5)
```
bash soft-t5-run-v1.sh
bash soft-t5-nob-run-v1.sh
```
