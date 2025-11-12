# ACN-TCN
Official PyTorch Implementation of "Lightweight Attentive ConvNeXt-TCN for Causal Target Sound Extraction"

We provide some audio samples here: [Audio samples](https://xiang-m-j.github.io/ACN-TCN-display/)

## Training and Evaluation

### Dataset

We use [Scaper](https://github.com/justinsalamon/scaper) toolkit to synthetically generate audio mixtures. Each audio mixture used in our experiments is described using a [`.jams`](https://jams.readthedocs.io/en/stable/) file . The `.jams` specifications are generated using [FSDKaggle2018](https://zenodo.org/record/2552860) and [TAU Urban Acoustic Scenes 2019](https://dcase.community/challenge2019/task-acoustic-scene-classification) datasets as sources for foreground and background sounds, respectively. Steps to create the dataset (obtained from https://github.com/vb000/Waveformer):

1. Go to the `data` directory:

        cd data

2. Download [FSDKaggle2018](https://zenodo.org/record/2552860), [TAU Urban Acoustic Scenes 2019, Development dataset](https://zenodo.org/record/2589280) and [TAU Urban Acoustic Scenes 2019, Evaluation dataset](https://zenodo.org/record/3063822) datasets using the `data/download.py` script:

        python download.py

3. Download and uncompress [FSDSoundScapes](https://targetsound.cs.washington.edu/files/FSDSoundScapes.zip) dataset:

        wget https://targetsound.cs.washington.edu/files/FSDSoundScapes.zip
        unzip FSDSoundScapes.zip

    This step creates the `data/FSDSoundScapes` directory. `FSDSoundScapes` would contain `.jams` specifications for training, validation and test samples used in the paper. Training and evaluation pipeline expect source samples (samples in `FSDKaggle2018` and `TAU Urban Acoustic Scenes 2019` datasets) at specific locations realtive to `FSDSoundScapes`. Following steps move source samples to appropriate locations.

4. Uncompress FSDKaggle2018 dataset and create scaper source:

        unzip FSDKaggle2018/\*.zip -d FSDKaggle2018
        python fsd_scaper_source_gen.py FSDKaggle2018 ./FSDSoundScapes/FSDKaggle2018 ./FSDSoundScapes/FSDKaggle2018

5. Uncompress TAU Urban Acoustic Scenes 2019 dataset to `FSDSoundScapes` directory:

        unzip TAU-acoustic-sounds/\*.zip -d FSDSoundScapes/TAU-acoustic-sounds/

### Preprocess

This step reads the `.jams` specifications, then generates the corresponding audio list and label list, and stores them in numpy format for easy loading during training.

```sh
python preprocess.py --TAU_path xxx

```
`--TAU_path` is used to specify the path where the `TAU-urban-acoustic-scenes-2019-development` and `TAU-urban-acoustic-scenes-2019-evaluation` folders are stored.


### Training

```python
python -W ignore -m src.training.train experiments/<Experiment dir with config.json> --use_cuda
```

### Evaluation

Run evaluation script:

```python
python -W ignore -m src.training.eval experiments/<Experiment dir with config.json and checkpoints> --use_cuda
```


### Reference

```
@misc{veluri2022realtime,
  title={Real-Time Target Sound Extraction}, 
  author={Bandhav Veluri and Justin Chan and Malek Itani and Tuochao Chen and Takuya Yoshioka and Shyamnath Gollakota},
  year={2022},
  eprint={2211.02250},
  archivePrefix={arXiv},
  primaryClass={cs.SD}
}
```
