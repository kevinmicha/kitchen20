# PROJET-3A-IMTA

Repository for the 3A Project - Reconnaissance de l'activité de vie quotidienne humaine à domicile à l'aide du son - IMT Atlantique - 2021/2022

### Authors:

<ul>
  <li>Mateo BENTURA</li>
  <li>Ezequiel CENTOFANTI</li>
  <li>Kevin MICHALEWICZ</li>
  <li>Oumaima TOUIL</li>
</ul>

### Environement 

In order to be able to execute the following steps, you will need to create a Python Environement.
This can be done by:

```bash
pip install -r requirements.txt
```

### Directory Structure

```

├── README.md                                   # This file
├── requirements.txt                            # Required packages
├── PythonAudioClassification.ipynb             # Classification implementation notebook, to train and test
├── AudioClassifier.py                          # Module to store defined functions and classes
├── test_audios.py                              # Simple testing script to classify a single audio file
├── .gitignore                                  
|
├── LL_AudioDB                                  # Sound database
|   ├── audio                                   # All audio files
|   ├── custom-audios                           # Our recorded audio files
│   ├──  metadata                                
|   │     ├── kitchen20b.csv                    # List of audio tracks 
|   │     └── kitchen20.csv                     # DB format
|   ├── add_audios.py                           # Script to add new audios to DB
|   └── categories.txt                          # List of categories used by 'add-audios.py'
|
├── Old-k20-model                               # First implementation of k20 model (unused)   |
└── checkpoints                                 # Trained model checkpoints
```

### Utilization guide
#### Python notebook
To train the model or bulk test audios we can use the notebook `PythonAudioClassification.ipynb`, whose use is straight-forward and requires simply running each command, with the option of loading the pre-trained model.

#### Test audios 
If you want to simply test one audio file, a testing script called `test_audios.py` is available. To used this, simply type the following command

```bash
python test_audios.py input_audio.wav
```

An exemplary output would be the following:

```bash
Testing audio 'input_audio.wav'
Predicted label is 'trash'
```

#### Add audios to database
This implementations uses a simple database, consisting of a directory and a `.csv` metadata file. In order to update this database and include new audio files, we provide a Python script called `add_audios.py`. This takes the `.wav` files from an input folder (they must indicate the label in the name) and places them in an ouput folder, updating the metadata file as well.

To call this script we use the following command in the corresponding directory:
```bash
python add_audios.py
```
This uses the default input and output directory names `custom-audios` and `audios`, we can use custom ones with the command:
```bash
python add_audios.py input-folder output-folder
```
The script prints the advancements and asks for a confirmation of this update.
