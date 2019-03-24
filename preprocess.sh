#!/bin/bash
cd ./raw_data

echo "download pretrained word embeddings: glove.42B.300d.zip"
wget https://nlp.stanford.edu/data/wordvecs/glove.42B.300d.zip

echo "unzip glove.42B.300d.zip"
unzip glove.42B.300d.zip

echo "add vocab size and embedding size to the head of glove.42B.300d.txt"
n=$(sed -n '$=' glove.42B.300d.txt)
sed -i "1i$n 300" glove.42B.300d.txt

cd ..
echo "process raw data"
python3 process_raw.py

echo "prepare training, valid, testing data"
if [ ! -d "./log" ]; then
    mkdir log
fi
python3 preprocess.py > log/preprocess.log
echo "Finished! preprocess.log is saved in log directory."
