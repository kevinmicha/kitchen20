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
pip install requirements.txt
```

### Directory Structure

```

│   README.md                                   # this file
│   requirements.txt     
|
├── audio                                       # all audio filies
|   ├── 0-101477-B-3.wav
|   ├── ...
│   └── 9-LL-S-15.wav           
|
├── custom-audios                               # LivingLab recorded audio filies
|   ├── boiling-water  
|   ├── ...
│   └── plates  
|
├── k20-old-model       
│   ├── kitchen20                    
│   ├──                     
│   └──                               
|                          
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
python add-audios.py
```
This uses the default input and output directory names `custom-audios` and `audios`, we can use custom ones with the command:
```bash
python add-audios.py input-folder output-folder
```
The script prints the advancements and asks for a confirmation of this update.
