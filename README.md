# deepspeech.pytorch

Implementation of DeepSpeech2 using [Baidu Warp-CTC](https://github.com/baidu-research/warp-ctc).
Creates a network based on the [DeepSpeech2](http://arxiv.org/pdf/1512.02595v1.pdf) architecture, trained with the CTC activation function.

Forked from SeanNaren/deepspeech.pytorch

Testing with python 3.7 and Pytorch 1.0 (pre-release)

## Features

Sean Naren's excellent implementation contains a lot of functionality (distributed training, etc.) that is not (yet) included here, see WIP below.

* Train DeepSpeech, configurable RNN types with Pytorch 1.0 CTC Loss
* Multiple dataset downloaders, support for AN4, TED, Voxforge and Librispeech. Datasets can be merged, support for custom datasets included.

WIP:
- [ ] multi-gpu support.
- [ ] cpu support.
- [ ] Language model support using kenlm (WIP right now, currently no instructions to build a LM yet).
- [ ] Noise injection for online training to improve noise robustness.
- [ ] Audio augmentation to improve noise robustness.
- [ ]  Easy start/stop capabilities in the event of crash or hard stop during training.
- [ ] Visdom/Tensorboard support for visualizing training graphs.

## Installation

conda / ubuntu with typings and python 3.7

```
conda create -n deepspeech python=3.7
```

Install [PyTorch 1.0](https://pytorch.org/get-started/locally/) if you haven't already 
or, typically


```
pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cu92/torch_nightly.html
```

# Pytorch Audio

```
sudo apt-get install sox libsox-dev libsox-fmt-all
git clone https://github.com/pytorch/audio.git
cd audio
pip install cffi
python setup.py install
```


Finally clone this repo and run this within the repo:

```
pip install -r requirements.txt
```


# Usage

## Dataset

Currently supports AN4, TEDLIUM, Voxforge and LibriSpeech. Scripts will setup the dataset and create manifest files used in dataloading.

### AN4

To download and setup the an4 dataset run below command in the root folder of the repo:

```
cd data; python an4.py
```

### TEDLIUM

You have the option to download the raw dataset file manually or through the script (which will cache it).
The file is found [here](http://www.openslr.org/resources/19/TEDLIUM_release2.tar.gz).

To download and setup the TEDLIUM_V2 dataset run below command in the root folder of the repo:

```
cd data; python ted.py # Optionally if you have downloaded the raw dataset file, pass --tar_path /path/to/TEDLIUM_release2.tar.gz

```
### Voxforge

To download and setup the Voxforge dataset run the below command in the root folder of the repo:

```
cd data; python voxforge.py
```

Note that this dataset does not come with a validation dataset or test dataset.

### LibriSpeech

To download and setup the LibriSpeech dataset run the below command in the root folder of the repo:

```
cd data; python librispeech.py
```

You have the option to download the raw dataset files manually or through the script (which will cache them as well).
In order to do this you must create the following folder structure and put the corresponding tar files that you download from [here](http://www.openslr.org/12/).

```
cd data/
mkdir LibriSpeech/ # This can be anything as long as you specify the directory path as --target-dir when running the librispeech.py script
mkdir LibriSpeech/val/
mkdir LibriSpeech/test/
mkdir LibriSpeech/train/
```

Now put the `tar.gz` files in the correct folders. They will now be used in the data pre-processing for librispeech and be removed after
formatting the dataset.

Optionally you can specify the exact librispeech files you want if you don't want to add all of them. This can be done like below:

```
cd data/
python librispeech.py --files-to-use "train-clean-100.tar.gz, train-clean-360.tar.gz,train-other-500.tar.gz, dev-clean.tar.gz,dev-other.tar.gz, test-clean.tar.gz,test-other.tar.gz"
```

### Custom Dataset

To create a custom dataset you must create a CSV file containing the locations of the training data. This has to be in the format of:

```
/path/to/audio.wav,/path/to/text.txt
/path/to/audio2.wav,/path/to/text2.txt
...
```

The first path is to the audio file, and the second path is to a text file containing the transcript on one line. This can then be used as stated below.


### Merging multiple manifest files

To create bigger manifest files (to train/test on multiple datasets at once) we can merge manifest files together like below from a directory
containing all the manifests you want to merge. You can also prune short and long clips out of the new manifest.

```
cd data/
python merge_manifests.py --output-path merged_manifest.csv --merge-dir all-manifests/ --min-duration 1 --max-duration 15 # durations in seconds
```

## Training

```
python train.py --train-manifest data/train_manifest.csv --val-manifest data/val_manifest.csv
```

Use `python train.py --help` for more parameters and options.

There is also [Visdom](https://github.com/facebookresearch/visdom) support to visualize training. Once a server has been started, to use:


```
python test.py --model-path models/deepspeech.pth --test-manifest /path/to/test_manifest.csv --cuda
```

An example script to output a transcription has been provided:

```
python transcribe.py --model-path models/deepspeech.pth --audio-path /path/to/audio.wav
```


## Pre-trained models

Pre-trained models can be found under releases [here](https://github.com/SeanNaren/deepspeech.pytorch/releases).

## Acknowledgements

Thanks to [Egor](https://github.com/EgorLakomkin) and [Ryan](https://github.com/ryanleary) for their contributions!
