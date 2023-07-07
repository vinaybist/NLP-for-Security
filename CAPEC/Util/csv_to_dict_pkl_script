#script to read csv of two column and create dictionary of col1 as key and col2 as value

import csv
import pickle 
import pandas as pd

# change this variable
ORG_FILE_NAME = "allitems.csv"
SUBSET_FILE_NAME = "cve_desc.csv"
DUMPED_PKL_FILE_NAME = "cve_dict.pkl" 

_dict = {}

with open(SUBSET_FILE_NAME, encoding='utf-8') as f:
    # reading the CSV file
    csvFile = csv.reader(f)
    next(csvFile) # toss headers
    for col1, col2 in csvFile:  # col1 and key and col2 as value
        _dict.setdefault(col1, col2)
    
    print(_dict)

with open(DUMPED_PKL_FILE_NAME, 'wb') as f:
    pickle.dump(_dict, f)
