# ACN-TCN


## Training and Evaluation

### Dataset

We use [Scaper](https://github.com/justinsalamon/scaper) toolkit to synthetically generate audio mixtures. Each audio mixture is generated on-the-fly, during training or evaluation, using Scaper's `generate_from_jams` function on a [`.jams`](https://jams.readthedocs.io/en/stable/) specification file. We provide (in the step 3 below) `.jams` specification files for all training, validation and evaluation samples used in our experiments. The `.jams` specifications are generated using [FSDKaggle2018](https://zenodo.org/record/2552860) and [TAU Urban Acoustic Scenes 2019](https://dcase.community/challenge2019/task-acoustic-scene-classification) datasets as sources for foreground and background sounds, respectively. Steps to create the dataset:

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

### Training

    python -W ignore -m src.training.train experiments/<Experiment dir with config.json> --use_cuda

### Evaluation

Pretrained checkpoints are available at [experiments.zip](https://targetsound.cs.washington.edu/files/experiments.zip). These can be downloaded and uncompressed to appropriate locations using:

    wget https://targetsound.cs.washington.edu/files/experiments.zip
    unzip -o experiments.zip -d experiments

Run evaluation script:

    python -W ignore -m src.training.eval experiments/<Experiment dir with config.json and checkpoints> --use_cuda

