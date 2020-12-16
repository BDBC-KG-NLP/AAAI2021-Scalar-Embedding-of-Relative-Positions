OUTPUT_DIR_DATA=wmt17

cd ${OUTPUT_DIR_DATA}

echo "Downloading preprocess data. This may take a while..."

wget http://data.statmt.org/wmt17/translation-task/preprocessed/de-en/corpus.tc.de.gz

wget http://data.statmt.org/wmt17/translation-task/preprocessed/de-en/corpus.tc.en.gz

echo "Downloading preprocessed dev data..."
wget http://data.statmt.org/wmt17/translation-task/preprocessed/de-en/dev.tgz

echo "Downloading test data..."
wget http://data.statmt.org/wmt17/translation-task/test.tgz
    

echo "Downloading truecase model..."
wget http://data.statmt.org/wmt17/translation-task/preprocessed/de-en/true.tgz

echo "Extracting all files..."
gzip -d corpus.tc.de.gz
gzip -d corpus.tc.en.gz

tar -zxvf dev.tgz
tar -zxvf test.tgz
tar -zxvf true.tgz


# Convert newstest2017 data into raw text format
perl scripts/input-from-sgm.perl \
  < test/newstest2017-ende-src.en.sgm \
  > newstest2017.en

perl scripts/input-from-sgm.perl \
  < test/newstest2017-ende-ref.de.sgm \
  > newstest2017.de


cat newstest2017.de | \
   perl scripts/normalize-punctuation.perl -l de | \
   perl scripts/tokenizer.perl -a -q -l de | \
   perl scripts/truecase.perl -model truecase-model.de > newstest2017.tc.de

cat newstest2017.en | \
   perl scripts/normalize-punctuation.perl -l en | \
   perl scripts/tokenizer.perl -a -q -l en | \
   perl scripts/truecase.perl -model truecase-model.en > newstest2017.tc.en

cd ../