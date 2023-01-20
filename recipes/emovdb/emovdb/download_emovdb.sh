#!/bin/bash
# take the scripts's parent's directory to prefix all the output paths.
RUN_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo $RUN_DIR

# download transcriptions (already on repo)
#wget http://www.festvox.org/cmu_arctic/cmuarctic.data
#mv cmuarctic.data transcriptions.txt

# download dataset
wget https://www.openslr.org/resources/115/bea_Amused.tar.gz
wget https://www.openslr.org/resources/115/bea_Angry.tar.gz
wget https://www.openslr.org/resources/115/bea_Disgusted.tar.gz
wget https://www.openslr.org/resources/115/bea_Neutral.tar.gz
wget https://www.openslr.org/resources/115/bea_Sleepy.tar.gz
wget https://www.openslr.org/resources/115/jenie_Amused.tar.gz
wget https://www.openslr.org/resources/115/jenie_Angry.tar.gz
wget https://www.openslr.org/resources/115/jenie_Disgusted.tar.gz
wget https://www.openslr.org/resources/115/jenie_Neutral.tar.gz
wget https://www.openslr.org/resources/115/jenie_Sleepy.tar.gz
wget https://www.openslr.org/resources/115/josh_Amused.tar.gz
wget https://www.openslr.org/resources/115/josh_Neutral.tar.gz
wget https://www.openslr.org/resources/115/josh_Sleepy.tar.gz
wget https://www.openslr.org/resources/115/sam_Amused.tar.gz
wget https://www.openslr.org/resources/115/sam_Angry.tar.gz
wget https://www.openslr.org/resources/115/sam_Disgusted.tar.gz
wget https://www.openslr.org/resources/115/sam_Neutral.tar.gz
wget https://www.openslr.org/resources/115/sam_Sleepy.tar.gz

# extract
mkdir files
mkdir files/bea_Amused
mkdir files/bea_Angry
mkdir files/bea_Disgusted
mkdir files/bea_Neutral
mkdir files/bea_Sleepy
mkdir files/jenie_Amused
mkdir files/jenie_Angry
mkdir files/jenie_Disgusted
mkdir files/jenie_Neutral
mkdir files/jenie_Sleepy
mkdir files/josh_Amused
mkdir files/josh_Neutral
mkdir files/josh_Sleepy
mkdir files/sam_Amused
mkdir files/sam_Angry
mkdir files/sam_Disgusted
mkdir files/sam_Neutral
mkdir files/sam_Sleepy

tar -xf bea_Amused.tar.gz -C files/bea_Amused
tar -xf bea_Angry.tar.gz -C files/bea_Angry
tar -xf bea_Disgusted.tar.gz -C files/bea_Disgusted
tar -xf bea_Neutral.tar.gz -C files/bea_Neutral
tar -xf bea_Sleepy.tar.gz -C files/bea_Sleepy
tar -xf jenie_Amused.tar.gz -C files/jenie_Amused
tar -xf jenie_Angry.tar.gz -C files/jenie_Angry
tar -xf jenie_Disgusted.tar.gz -C files/jenie_Disgusted
tar -xf jenie_Neutral.tar.gz -C files/jenie_Neutral
tar -xf jenie_Sleepy.tar.gz -C files/jenie_Sleepy
tar -xf josh_Amused.tar.gz -C files/josh_Amused
tar -xf josh_Neutral.tar.gz -C files/josh_Neutral
tar -xf josh_Sleepy.tar.gz -C files/josh_Sleepy
tar -xf sam_Amused.tar.gz -C files/sam_Amused
tar -xf sam_Angry.tar.gz -C files/sam_Angry
tar -xf sam_Disgusted.tar.gz -C files/sam_Disgusted
tar -xf sam_Neutral.tar.gz -C files/sam_Neutral
tar -xf sam_Sleepy.tar.gz -C files/sam_Sleepy

# remove
rm bea_Amused.tar.gz 
rm bea_Angry.tar.gz
rm bea_Disgusted.tar.gz
rm bea_Neutral.tar.gz
rm bea_Sleepy.tar.gz
rm jenie_Amused.tar.gz
rm jenie_Angry.tar.gz
rm jenie_Disgusted.tar.gz
rm jenie_Neutral.tar.gz
rm jenie_Sleepy.tar.gz
rm josh_Amused.tar.gz
rm josh_Neutral.tar.gz
rm josh_Sleepy.tar.gz
rm sam_Amused.tar.gz
rm sam_Angry.tar.gz
rm sam_Disgusted.tar.gz
rm sam_Neutral.tar.gz
rm sam_Sleepy.tar.gz