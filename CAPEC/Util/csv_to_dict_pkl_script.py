#script to read csv of two column and create dictionary of col1 as key and col2 as value

import csv
import pickle 

# change this variable
FILE_NAME = "id_desc_opt.csv"
DUMPED_PKL_FILE_NAME = "capec_dict.pkl" 


_dict = {}


with open(FILE_NAME, encoding='utf-8') as f:
    # reading the CSV file
    csvFile = csv.reader(f)
    next(csvFile) # toss headers
    for col1, col2 in csvFile:  # col1 and key and col2 as value
        _dict.setdefault("CAPEC-"+col1, col2)
    
    print(_dict)

with open(DUMPED_PKL_FILE_NAME, 'wb') as f:
    pickle.dump(_dict, f)
