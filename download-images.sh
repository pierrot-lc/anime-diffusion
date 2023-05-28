#!/bin/sh

mkdir -p datasets
cd datasets || exit

kaggle datasets download -d splcher/animefacedataset
unzip animefacedataset.zip
mv images anime-faces
rm animefacedataset.zip
