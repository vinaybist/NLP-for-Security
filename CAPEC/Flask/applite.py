from flask import *
import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import jsonpickle
import requests
from bs4 import BeautifulSoup
import csv
import pickle 
import time
from datetime import datetime
import sentencepiece as spm
import tensorflow.compat.v1 as tf


app = Flask(__name__)


#initialization of constants
CVE_SAMPLE = "D-Link DCS-825L devices with firmware 1.08 do not employ a suitable mechanism to prevent denial-of-service (DoS) attacks. An attacker can harm the device availability (i.e., live-online video/audio streaming) by using the hping3 tool to perform an IPv4 flood attack. Verified attacks includes SYN flooding, UDP flooding, ICMP flooding, and SYN-ACK flooding.";
DUMPED_CVE_FILE_NAME = "cve_dict.pkl" 
DUMPED_CAPEC_FILE_NAME = "capec_dict.pkl" 

# Lite calls requires below
tf.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()

def process_to_IDs_in_sparse_format(sp, sentences):
    # An utility method that processes sentences with the sentence piece processor
    # 'sp' and returns the results in tf.SparseTensor-similar format:
    # (values, indices, dense_shape)
    ids = [sp.EncodeAsIds(x) for x in sentences]
    max_len = max(len(x) for x in ids)
    dense_shape=(len(ids), max_len)
    values=[item for sublist in ids for item in sublist]
    indices=[[row,col] for row in range(len(ids)) for col in range(len(ids[row]))]
    return (values, indices, dense_shape)
    
    



def fetch_cve_details(cve_id):
    url = f"https://cve.mitre.o1rg/cgi-bin/cvename.cgi?name={cve_id}"
    try:
        print("Inside fetch_cve_details...")
        response = requests.get(url)
        print("1",response)
        response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            print("1",soup)
            # Extract relevant information from the parsed HTML
            table_container  = soup.find('div', id = "GeneratedTable")
            if isinstance(table_container, type(None)):
              return None
            else:
              table = table_container.find('table')
              # Extract table data
              table_data = []
              header_row = table.find('tr')
              headers = [header for header in header_row.find_all('th')]
              table_data.append(headers)
              rows = table.find_all('tr')
              for row in rows[1:]:
                  cells = [cell.text.strip() for cell in row.find_all('td')]
                  table_data.append(cells)

              cve_assigned_cna = ''.join(table_data[8])
              cve_description = ''.join(table_data[3])
              cve_details = {
                  "CVE ID": cve_id,
                  "Assigned CNA": cve_assigned_cna,
                  "Description": cve_description
              }
            print("3")
            return cve_details
        else:
            print(f"Failed to fetch CVE details for {cve_id}. Status Code: {response.status_code}")
            return None
            
    except Exception as e:
           print(f"Error Occured while connecting to CVE wesite- ,{e}")  
    

def fetch_cve_details_locally(cve):
    print("Inside fetch_cve_details_locally = ",cve)
    with open('./cve_dict.pkl', 'rb') as f:
        loaded_cve_dict = pickle.load(f)
        return loaded_cve_dict[cve]

def get_capec_dict():
    capec_data_uri = "./id_desc.csv"
    capec_data = pd.read_csv(capec_data_uri, encoding='utf-8')
    print(type(capec_data))
    capec_data = capec_data.dropna()
    capec_data = capec_data.sort_values(by=['ID'], ascending=True)
    capec_data = capec_data.reset_index(drop=True)
    print("Sorting the dataframe")
    capec_data['ID'] = "CAPEC-"+capec_data['ID'].apply(str)
    print("Creating dict of capecs")
    dict_of_capecs = capec_data.set_index('ID').to_dict()['Description']    
    return dict_of_capecs
    
def get_capec_dict_locally():
    capec_data_uri = "./id_desc.csv"
    capec_data = pd.read_csv(capec_data_uri, encoding='utf-8')
    print("Inside get_capec_dict_locally = ")
    with open('./capec_dict.pkl', 'rb') as f:
        loaded_capec_dict = pickle.load(f)
        return loaded_capec_dict

      


def cve_output(cve_details):
    if cve_details is not None:
        print("CVE Details - >>>")
        for key, value in cve_details.items():
            print(f"{key}: {value}")
            if key == "Description":
               return cve_details[key]
    else:
            print("Error Occured")
            return "Error";
            

def sent_embedings(input, model):
    return model(input)

def cve_processing_test(cve):
    list_sample = ["a","b","c","d"];
    return list_sample;



def cve_processing(cve):
    Input_CVE = CVE_SAMPLE
    #start = time.time()
    start = datetime.now()

    try:    
        print("Inside python logic of cosine similarity = ",cve)
        #capec_data_uri = "./id_desc.csv"
        #capec_data = pd.read_csv(capec_data_uri, encoding='utf-8')
        #print(type(capec_data))
        #capec_data = capec_data.dropna()
        #capec_data = capec_data.sort_values(by=['ID'], ascending=True)
        #capec_data = capec_data.reset_index(drop=True)
        #print("Sorting the dataframe")
        #capec_data['ID'] = "CAPEC-"+capec_data['ID'].apply(str)
        #print("Creating dict of capecs")
        #dict_of_capecs = capec_data.set_index('ID').to_dict()['Description']
        # get_capec_dict()
        dict_of_capecs = get_capec_dict_locally(); 
        #print("dict of capecs saved as jason file")
        #print("?????????????????????????????? ==> ")
        Input_CVE = fetch_cve_details_locally(cve)
        #print("?????????????????????????????? ==> ",x)
        #if cve != "CVE-2018-18442":
        #    Input_CVE = fetch_cve_details(cve)['Description'] 
        #Input_CVE = "D-Link DCS-825L devices with firmware 1.08 do not employ a suitable mechanism to prevent denial-of-service (DoS) attacks. An attacker can harm the device availability (i.e., live-online video/audio streaming) by using the hping3 tool to perform an IPv4 flood attack. Verified attacks includes SYN flooding, UDP flooding, ICMP flooding, and SYN-ACK flooding.";
        print("Input_CVE ===> ",Input_CVE)
        if Input_CVE is None:
            return "Error Occured while cve_processing - Input_CVE is None"
        Input_doc_dict = {'input_doc':Input_CVE}
        dict_of_corpus = {**dict_of_capecs, **Input_doc_dict}
        corpus_keys = list(dict_of_corpus.keys())
        end = datetime.now()
        difference = end - start
        print("Loading the corpus_keys time ==> ",difference.total_seconds()) # Loading the corpus_keys time ==>  0.43802523612976074

        print("Loading the model...")
        start = datetime.now()
        
        
        #Load the module from TF-Hub
        with tf.Session() as sess:
          module = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-lite/2")
          spm_path = sess.run(module(signature="spm_path"))


        sp = spm.SentencePieceProcessor()
        sp.Load(spm_path)


        input_placeholder = tf.compat.v1.sparse_placeholder(tf.int64, shape=[None, None])
        encodings = module(inputs=dict(values=input_placeholder.values, indices=input_placeholder.indices,dense_shape=input_placeholder.dense_shape))
            
        #Load SentencePiece model from the TF-Hub Module
        with tf.Session() as sess:
            spm_path = sess.run(module(signature="spm_path"))

        sp = spm.SentencePieceProcessor()
        with tf.io.gfile.GFile(spm_path, mode="rb") as f:
            sp.LoadFromSerializedProto(f.read())

        print("SentencePiece model loaded at {}.".format(spm_path))


        print("Loading the model - Done",module)
        
        
        end = datetime.now()
        difference = end - start
        print("Loading the model time ==> ",difference.total_seconds())   # Loading the model time ==>  7.744394

        start = datetime.now()
        sent_list_1 = list(dict_of_corpus.values())
        ls_dict_of_capecs_1 = list(dict_of_corpus.keys())
        #print(type(ls_dict_of_capecs_1))
        values, indices, dense_shape = process_to_IDs_in_sparse_format(sp, sent_list_1)        
        with tf.Session() as session:
          session.run([tf.global_variables_initializer(), tf.tables_initializer()])
          dicts_vec_1 = session.run(
              encodings,
              feed_dict={input_placeholder.values: values,
                        input_placeholder.indices: indices,
                        input_placeholder.dense_shape: dense_shape})
        
                        
                        
        #dicts_vec_1 = sent_embedings(sent_list_1, model)


        final_dict_vec_1 = {ls_dict_of_capecs_1[i]: dicts_vec_1[i] for i in range(len(ls_dict_of_capecs_1))}
        cs1_1 = cosine_similarity([final_dict_vec_1['input_doc']], list(final_dict_vec_1.values()))
        
        end = datetime.now()
        difference = end - start
        print("Got the similarity scores")
        print("Got the similarity scores time ==> ",difference.total_seconds())  # Got the similarity scores time ==>  0.7940115928649902
        
        df1_1 = pd.DataFrame(cs1_1, columns =corpus_keys)
        response_dict_1 = df1_1.to_dict('index')[0]
        sorted_CAPEC_by_maximum_similarity_1 = sorted(response_dict_1.items(), reverse=True, key=lambda x:x[1])
        sorted_CAPEC_by_maximum_similarity_1
        print("Sort the list - based on maximum similarity ")
        return_capec_list = [i[0] for i in sorted_CAPEC_by_maximum_similarity_1[1:10]]
        print("returning the ",type(return_capec_list))
        return return_capec_list

    except Exception as e:
        print(f"Error Occured while cve_processing- ,{e}")
        return "Error Occured while cve_processing";

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_CVE = request.form.get('name')
        #input_CVE_desc = request.form.get('desc')
        #print('input_CVE_desc',input_CVE_desc)
        print('Input CVE',input_CVE)
        result = cve_processing(input_CVE)
        print("=======> ",result)
        tbl_tag="";
        if "Error Occured while cve_processing" in result:
            tbl_tag = tbl_tag +"<tr><td>Error Occured, Please try again</td></tr>";
        else:
            print("No Error block!")
            for i in result:
                tbl_tag = tbl_tag +"<tr><td>"+i+"</td></tr>"
                print("inside == ",tbl_tag)        
        

        
        
        print("tbl_tag ==> ",tbl_tag)
        tbl_tag = "<tr><th bgcolor='#009879'>Related Attack Techniques&nbsp;&nbsp;</th></tr>"+tbl_tag
        return jsonpickle.encode(tbl_tag)




if __name__ == '__main__':
    app.run(debug=True)
