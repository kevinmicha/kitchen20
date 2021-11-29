import os, sys, re, random, shutil
import pandas as pd

if len(sys.argv) == 3:
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
if len(sys.argv) == 1:
    input_folder = "custom-audios"
    output_folder = "script-test"
else: sys.exit("Wrong number of argarguments. Please enter an input folder followed by an output folder.")

print("=============================================")
print("Running script to add audios to the database.")
print("=============================================")

print("Obtaining audios from folder '"+input_folder+"'")
print("Outputing audios to folder '"+output_folder+"'")

input_dir = os.path.join(os.getcwd(), input_folder)
output_dir = os.path.join(os.getcwd(), output_folder)

# Define target categories: modify according to the dataset

with open('categories.txt') as f:
    categories = [category for category in f.read().splitlines()]

print("Dataset containing "+str(len(categories))+" categories.")
category_codes = {category:str(i) for i, category in enumerate(categories)}

usr_id = "LL"

# TODO open existing csv file (kitchen20.csv)
# TODO check, with user_id if files are not already added

audio_categories = sorted(os.listdir(input_dir), key=lambda item: int(category_codes[item]))

folds = [10]
id = 0 # TODO begin id according to number of items in the csv

cols = ['category', 'fold', 'orig_idx', 'path', 'take', 'target', 'usr_id']
df = pd.DataFrame(columns=cols)

# TODO iterate over all categories
audio_category = audio_categories[0]
category_dir = os.path.join(input_dir, audio_category)
print("Accessing '"+audio_category+"', with label",category_codes[audio_category])
p = re.compile('\d+')
for audio in sorted(os.listdir(category_dir)):
    audio_path = os.path.join(category_dir,audio)
    take = chr(int(p.findall(audio)[0])+64)
    fold = random.choice(folds)
    audio_name = str(fold-1)+'-'+usr_id+'-'+take+'-'+category_codes[audio_category]+'.wav'
    print(audio_name)
    output_path = os.path.join(output_dir, audio_name)
    shutil.copyfile(audio_path, output_path)
    line = pd.DataFrame([[audio_category,fold,id,os.path.join(output_folder,audio_name),take,category_codes[audio_category],usr_id]], columns=cols)
    df = df.append(line, ignore_index=True)
    id += 1
print(df)


df.to_csv('test.csv')
# for audio_category in audio_categories:
#     category_dir = input_dir+audio_category
#     print("Accessing '"+audio_category+"' audios")
#     # audio_list =


# for root, audio_category, audio_files in os.walk(input_dir):
#     print("Accessing '"+audio_files+"' audios")
