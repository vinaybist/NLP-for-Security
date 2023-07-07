# FOR TESTING USE ONLY ******************

import csv
import pickle 
import tensorflow_hub as hub
import tensorflow as tf

# change this variable
DUMPED_PKL_FILE_NAME = "capec_dict.pkl" 

print("%%%%==> ")
#with open(DUMPED_PKL_FILE_NAME, 'rb') as f:
#    loaded_dict = pickle.load(f)
#    print(loaded_dict)
    
import os
#create the directory in which to cache the tensorflow universal sentence encoder.
#os.environ["TFHUB_CACHE_DIR"] = 'C:\\Users\\Vinay_Bist\\ML'
#download = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/4")    
loaded_obj = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
export_module_dir = os.path.join(os.getcwd(), "finetuned_model_export1")
print(export_module_dir)
tf.saved_model.save(loaded_obj, export_module_dir)
