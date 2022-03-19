import pandas as pd
from torch.utils.data import DataLoader
from AudioClassifier import *
import sys, os
import time

with open('LL_AudioDB/categories.txt') as f:
    categories = [category for category in f.read().splitlines()]

del categories[-1]

categories_id = {category:i for i, category in enumerate(categories)}

if len(sys.argv) == 1:
	print("Test audio missing")
	quit()
elif len(sys.argv) == 2:
	audio_path = sys.argv[1]
	print(f"Testing audio '{audio_path}'")
	dict_ = {'target':[0],'path':[audio_path]}
	df = pd.DataFrame(dict_)
	#df = pd.DataFrame(['0', audio_path], columns=['target','path'])

nb_classes = len(categories_id) # 26

test_ds = SoundDS(df)
test_dl = DataLoader(test_ds)

#Creating empty model

myModel = AudioClassifier(classes=nb_classes)

#Loading the pre-trained model

epochs = 350
model_out_path = "checkpoints/" + "model_" + "{}ep".format(epochs) 
myModel.load_checkpoint(model_out_path)

start = time.time()
myModel.inference(test_dl, classes=categories, with_prediction=True)
print(f"Time elapsed: {round(time.time()-start,4)}s")

targets, preds_1 = myModel.get_all_prediction(test_dl)
