# README for the data folder

### `vectors`-subfolder

For experiments we use the GloVe-6B-300d pre-trained word vector:
```
cd data/
mkdir vectors
cd vectors
wget https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
```

### `vocabularies`-subfolder

Create the vocabulary file as follows:
```
cd python/utils/
python get_vocabulary.py -c <dataset>
```
Options for `<dataset>` are: `robust04`, `trec-covid`, and `msmarco-passage`