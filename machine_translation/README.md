# Machine Translation

## Data Processing

-  download dataset

```
bash download.sh
```


- Run BPE

```
cd wmt17

git clone https://github.com/rsennrich/subword-nmt.git

python subword-nmt/learn_joint_bpe_and_vocab.py --input corpus.tc.de corpus.tc.en -s 32000 -o bpe32k --write-vocabulary vocab.de vocab.en

python subword-nmt/apply_bpe.py --vocabulary vocab.de --vocabulary-threshold 50 -c bpe32k < corpus.tc.de > corpus.tc.32k.de

python subword-nmt/apply_bpe.py --vocabulary vocab.en --vocabulary-threshold 50 -c bpe32k < corpus.tc.en > corpus.tc.32k.en

for year in {2014,2015,2016,2017}
do
{
  python subword-nmt/apply_bpe.py --vocabulary vocab.en
  --vocabulary-threshold 50 -c bpe32k < newstest${year}.tc.en >
  newstest${year}.tc.32k.en
  
  python subword-nmt/apply_bpe.py --vocabulary vocab.de
  --vocabulary-threshold 50 -c bpe32k < newstest${year}.tc.de >
  newstest${year}.tc.32k.de
}
done
```

- Suffling Training Set

```
python ../thumt/scripts/shuffle_corpus.py --corpus corpus.tc.32k.de corpus.tc.32k.en --suffix shuf
```

- Generating Vocabularies

```
python ../thumt/scripts/build_vocab.py corpus.tc.32k.de.shuf vocab.32k.de

python ../thumt/scripts/build_vocab.py corpus.tc.32k.en.shuf vocab.32k.en
```

## Training Models

- AT5

```
cd ../

bash raw-soft-t5_train_translate-en2de.sh

bash raw-soft-t5-nob_train_translate-en2de.sh
```

- T5

```
bash raw-t5_train_translate-en2de.sh

bash raw-t5-nob_train_translate-en2de.sh
```

## Test Models

```
bash test-soft-t5.sh

bash test-soft-t5-nob.sh

bash test-t5.sh

bash test-t5-nob.sh
```
