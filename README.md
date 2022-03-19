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
├── PythonAudioClassification.ipynb             # Classification implementation notebook
├── .gitignore                                  
|
├── LL_AudioDB                                  # Sound database
|   ├── audio                                   # All audio files
|   ├── custom-audios                           # Our recorded audio files
│   ├──  metadata                                
|   │     ├── kitchen20b.csv                    # List of audio tracks 
|   │     └── kitchen20.csv                     # DB format
|   ├── add-audios.py                           # Script to add new audios to DB
|   └── categories.txt                          # List of categories used by 'add-audios.py'
|
├── Old-k20-model                               # First implementation of k20 model (unused)   |
└── checkpoints                                 # Trained model checkpoints
```

### Execution 

In a Python Enviornment with the adequate project dependencies, only the following line has to be written:

```bash
complete
```
#### Add audios to database
This implementations uses a simple database, consisting of a directory and a `.csv` metadata file. In order to update this database and include new audio files, we provide a Python script called `add-audios.py`. This takes the `.wav` files from an input folder (they must indicate the label in the name) and places them in an ouput folder, updating the metadata file as well.

To call this script we use the following command in the corresponding directory:
```bash
python add_audios.py
```
This uses the default input and output directory names `custom-audios` and `audios`, we can use custom ones with the command:
```bash
python add_audios.py input-folder output-folder
```
The script prints the advancements and asks for a confirmation of this update.
