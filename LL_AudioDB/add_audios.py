import os, sys, re, random, shutil
import pandas as pd

if len(sys.argv) == 3:
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
elif len(sys.argv) == 1:
    input_folder = "custom-audios"
    output_folder = "audio"
else:
    sys.exit("Given "+str(len(sys.argv))+" arguments. Please, ONLY enter an input folder followed by an output folder.")

print("=============================================")
print("Running script to add audios to the database.")
print("=============================================")

print("Obtaining audios from folder '"+input_folder+"'")
print("Outputting audios to folder '"+output_folder+"'")

input_dir = os.path.join(os.getcwd(), input_folder)
output_dir = os.path.join(os.getcwd(), output_folder)

try:
    print("Creating output folder...")
    os.mkdir(output_folder)
except:
    print("Output folder exists.")

# Define target categories: modify according to the dataset
with open('categories.txt') as f:
    categories = [category for category in f.read().splitlines()]

print("Dataset containing "+str(len(categories))+" categories.")
category_codes = {category:str(i) for i, category in enumerate(categories)}

# Define the user who recorded the examples
usr_id = "LL" # IDEA add date to differentiate new recordings of the same category
# IDEA Prompt user in script to directly add id

df = pd.read_csv('metadata/kitchen20.csv').drop(columns=['Unnamed: 0','Unnamed: 0.1'])

if any(df['usr_id'] == usr_id):
    print("There exists recordings by user '"+usr_id+"' in the database.")
    ans = input("Do you wish to continue? [y/n] ")
    if ans != 'y' and ans != '':
        sys.exit("Exiting.")
    else:
        print("Continuing script.")

categories = sorted(os.listdir(input_dir), key=lambda item: int(category_codes[item]))

folds = [10]
id = len(df)

cols = list(df.keys())

for category in categories:
    category_dir = os.path.join(input_dir, category)
    print("Accessing '"+category+"', label=",category_codes[category])
    p = re.compile('\d+')
    for audio in sorted(os.listdir(category_dir)):
        audio_path = os.path.join(category_dir,audio)
        take = chr(int(p.findall(audio)[0])+64)
        fold = random.choice(folds)
        attributes = [fold-1,usr_id,take,category_codes[category]]
        audio_name = "".join([str(attr)+'-' for attr in attributes])[:-1]+'.wav'
        path = os.path.join(output_folder,audio_name)
        if any(df['path'] == path):
            print("There already exists a file with the same name.")
        else:
            copy_path = os.path.join(output_dir, audio_name)
            shutil.copyfile(audio_path, copy_path)

        line = pd.DataFrame([[  category,
                                fold,
                                id,
                                path,
                                take,
                                category_codes[category],
                                usr_id]],
                                columns=cols)
        df = df.append(line, ignore_index=True)
        id += 1

df.to_csv('metadata/kitchen20b.csv')
print("Audios added succesfully to database.")
