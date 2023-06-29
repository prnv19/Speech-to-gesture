# Speech-driven Hand Gesture Generation Demo

## Requirements
* python 3
* ffmpeg (to visualize the results)

## Install dependencies
```
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage
```
./generate.sh  data/audio*.wav
```
Where in place of `audio*.wav` you can use any file from the folder `data`, which are chunks of the test sequences.
 Alternatively, you can download more audios for testing from [the Trinity Speech-Gesture dataset](https://trinityspeechgesture.scss.tcd.ie/Audio/).
(The recordings 'NaturalTalking_01.wav' and 'NaturalTalking_02.wav' were not used in training and were left them for testing)

