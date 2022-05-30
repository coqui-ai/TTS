# create venv
python3 -m venv env
source .env/bin/activate
pip install pip --upgrade

# download Thorsten_DE dataset
pip install gdown
gdown --id 1yKJM1LAOQpRVojKunD9r8WN_p5KzBxjc -O dataset.tgz
tar -xzf dataset.tgz

# create train-val splits
shuf LJSpeech-1.1/metadata.csv > LJSpeech-1.1/metadata_shuf.csv
head -n 20668 LJSpeech-1.1/metadata_shuf.csv > LJSpeech-1.1/metadata_train.csv
tail -n 2000 LJSpeech-1.1/metadata_shuf.csv > LJSpeech-1.1/metadata_val.csv

# rename dataset and remove archive
mv LJSpeech-1.1 thorsten-de
rm dataset.tgz

# destry venv
rm -rf env
