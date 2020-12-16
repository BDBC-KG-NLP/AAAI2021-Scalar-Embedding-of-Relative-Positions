# Artificial Tasks & Text Classification

## Generate datasets for artifical tasks


```
Download data from: https://drive.google.com/file/d/1eEFXccubyUyzppFZuN-IFSKowI8h31_w/view?usp=sharing

unzip data.zip

cd gen_data_utils
```

- generate Reber

```
python t1_reber.py
```

- generate Process-50

```
python t2_process.py
```

- generate adding problem

```
python t3_adding.py 100
```


- get embedding

```
GLOVE_DIR=embedding

mkdir -p $GLOVE_DIR

wget http://nlp.stanford.edu/data/glove.840B.300d.zip -O $GLOVE_DIR/glove.840B.300d.zip

unzip $GLOVE_DIR/glove.840B.300d.zip
```

## Model training

```
cd Transformer
```

- Due to extract the word embedding from Glove for text classificaiton, it may take a bit longer time when you first run.

- run Non-PE 

```
bash scripts/non-pe-run.sh
```

- run T5

```
bash scripts/t5-run.sh
bash scripts/t5-nb-run.sh
```

- run AT5 (soft-t5)

```
bash scripts/soft-t5-run.sh
bash scripts/soft-t5-nb-run.sh
```
