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
import chromadb
from tensorflow.python.types.doc_typealias import document

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


app = Flask(__name__)


#initialization of constants
app.json.sort_keys = False
CVE_SAMPLE = "D-Link DCS-825L devices with firmware 1.08 do not employ a suitable mechanism to prevent denial-of-service (DoS) attacks. An attacker can harm the device availability (i.e., live-online video/audio streaming) by using the hping3 tool to perform an IPv4 flood attack. Verified attacks includes SYN flooding, UDP flooding, ICMP flooding, and SYN-ACK flooding.";
DUMPED_CVE_FILE_NAME = "cve_dict.pkl" 
DUMPED_CAPEC_FILE_NAME = "capec_dict.pkl" 

path = '.'
client = chromadb.PersistentClient(path=path)


print("Loading the model...")
start = datetime.now()

#loaded_obj = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
export_module_dir = os.path.join(os.getcwd(), "finetuned_model_export")
#tf.saved_model.save(loaded_obj, export_module_dir)

model = tf.saved_model.load(export_module_dir)

print("Loading the model - Done",model)
end = datetime.now()
difference = end - start
print("Loading the model time ==> ",difference.total_seconds())   # Loading the model time ==>  7.744394



def get_or_create_Collection(client):
    print("Inside get_or_reate_collection...") 
    collection_db = client.get_or_create_collection("capec_embeddings")
    
    dbExists = len(collection_db.get('CAPEC-1').get('ids')) != 0

    if dbExists:
        print("====> DB already Exists, so fetch only mode")
        return collection_db
    else:
        print("====> Create and add all embeddings to DB")
        # get capac dicts
        docdict = get_capec_dict()  
        # get embeddings 
        model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        USE_capec_embeddings = model(list(docdict.values()))
        embeddings = USE_capec_embeddings.numpy().tolist()
        collection_db.add(
            documents=list(docdict.values()),
            embeddings=embeddings,
            ids =list(docdict.keys())
         )
        return collection_db


def get_or_create_CVE_Collection(client):
    print("Inside get_or_create_CVE_collection...")
    collection_db = client.get_or_create_collection("cve_embeddings")

    dbExists = len(collection_db.get()['ids']) != 0

    if dbExists:
        print("====> DB already Exists, so fetch only mode")
        return collection_db
    else:
        print("====> Create and add all embeddings to DB")
        # get cve dicts
        docdict = get_cve_dict_locally()
        print("get the dict-----",type(docdict))
        # load model and get embeddings
        model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        USE_cve_embeddings = model(list(docdict.values()))
        embeddings = USE_cve_embeddings.numpy().tolist()
        collection_db.add(
            documents=list(docdict.values()),
            embeddings=embeddings,
            ids =list(docdict.keys())
         )
        return collection_db

# get_or_store_Embeddings(dict_of_capecs)

def fetch_cve_details(cve_id):
    url = f"https://cve.mitre.org/cgi-bin/cvename.cgi?name={cve_id}"
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

def get_cve_dict_locally():
    print("Inside get_cve_dict_locally to read from pkl.... ")
    with open('./cve_dict.pkl', 'rb') as f:
        loaded_cve_dict = pickle.load(f)
        return loaded_cve_dict


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
        dict_of_capecs = get_capec_dict(); 
        #print("dict of capecs saved as jason file")
        #print("?????????????????????????????? ==> ")
        #Input_CVE = fetch_cve_details_locally(cve)
        Input_CVE = fetch_cve_details(cve)['Description']
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

        #print("Loading the model...")
        #start = datetime.now()
        
        #loaded_obj = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        #export_module_dir = os.path.join(os.getcwd(), "finetuned_model_export")
        #tf.saved_model.save(loaded_obj, export_module_dir)
        
        #model = tf.saved_model.load(export_module_dir)

        #model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        #downloaded_model = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/4")
        #saved_model_path = "./model"
        #tf.saved_model.save(model, saved_model_path)
        #Load the TF Hub model from custom folder path
        #model = hub.load(saved_model_path)
        #model = hub.Module('C:\\Users\\Vinay_Bist\\Documents\\Work\\AI-ML\\AWS\\model_cache\\063d866c06683311b44b4992fd46003be952409c');
    
        #print("Loading the model - Done",model)
        #end = datetime.now()
        #difference = end - start
        #print("Loading the model time ==> ",difference.total_seconds())   # Loading the model time ==>  7.744394

        start = datetime.now()
        sent_list_1 = list(dict_of_corpus.values())
        ls_dict_of_capecs_1 = list(dict_of_corpus.keys())
        #print(type(ls_dict_of_capecs_1))
        dicts_vec_1 = sent_embedings(sent_list_1, model)
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


def getcapecs_with_score(cve):
    Input_CVE = CVE_SAMPLE
    start = datetime.now()

    try:
        print("Inside python logic of cosine similarity = ",cve)
        dict_of_capecs = get_capec_dict();
        Input_CVE = fetch_cve_details(cve)['Description']
        print("Input_CVE ===> ",Input_CVE)
        if Input_CVE is None:
            return "Error Occured while cve_processing - Input_CVE is None"
        Input_doc_dict = {'input_doc':Input_CVE}
        dict_of_corpus = {**dict_of_capecs, **Input_doc_dict}
        corpus_keys = list(dict_of_corpus.keys())
        end = datetime.now()
        difference = end - start
        print("Loading the corpus_keys time ==> ",difference.total_seconds()) # Loading the corpus_keys time ==>  0.43802523612976074
        #print("Loading the model...")
        #start = datetime.now()

        #export_module_dir = os.path.join(os.getcwd(), "finetuned_model_export")

        #model = tf.saved_model.load(export_module_dir)


        #print("Loading the model - Done",model)
        #end = datetime.now()
        #difference = end - start
        #print("Loading the model time ==> ",difference.total_seconds())   # Loading the model time ==>  7.744394

        start = datetime.now()
        sent_list_1 = list(dict_of_corpus.values())
        ls_dict_of_capecs_1 = list(dict_of_corpus.keys())
        dicts_vec_1 = sent_embedings(sent_list_1, model)
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
        result_json_dict = dict(sorted_CAPEC_by_maximum_similarity_1[1:10])
        res = {key : round(result_json_dict[key], 2) for key in result_json_dict}
        return res

    except Exception as e:
        print(f"Error Occured while cve_processing- ,{e}")
        return "Error Occured while cve_processing";



def getcapecsv2(cve):
    Input_CVE = CVE_SAMPLE
    start = datetime.now()

    try:
        print("Inside getcapecV2 method  = ",cve)
        #dict_of_capecs = get_capec_dict();
        Input_CVE = fetch_cve_details(cve)['Description']
        print("Input_CVE ===> ",Input_CVE)
        if Input_CVE is None:
            return "Error Occured while cve_processing - Input_CVE is None"
        print("Loading the model...")
        #start = datetime.now()
        #export_module_dir = os.path.join(os.getcwd(), "finetuned_model_export")
        #model = tf.saved_model.load(export_module_dir)
        #model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        #model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        print("Loading the model - Done",model)
        #end = datetime.now()
        #difference = end - start
        #print("Loading the model time !!!!!!!!!!!!!!$$$$$$$$$$!!!!!!!!!!!!!==============> ",difference.total_seconds())   # Loading the model time ==>  7.744394
        query_text = Input_CVE
        query_embeddings = model([Input_CVE])
        query_embeddings = query_embeddings.numpy().tolist()
        print("query embedding done for input..")
        collection_db = get_or_create_Collection(client)
        query_result = collection_db.query(
            query_embeddings =query_embeddings,
            n_results=5,
             )
     
        res = dict(list(query_result.items())[:2]) 
        return res

    except Exception as e:
        print(f"Error Occured while getcapecsV2 - ,{e}")
        return "Error Occured while getcapecsV2";


def getQueryEmbedding(cve):
    # check if the query CVE exists in chroma collection if not then go to Thub to get embeddings
    collection_cve_db = get_or_create_CVE_Collection(client)

    cveExists = len(collection_cve_db.get(cve).get('ids')) != 0

    if cveExists:
        # fetch embedding from chromaDB
        embedding = collection_cve_db.get(ids=[cve],include=["embeddings"])
        embedding = embedding['embeddings'][0]
    else:
        #else fetch from THub
        start = datetime.now()
        #export_module_dir = os.path.join(os.getcwd(), "finetuned_model_export")
        #model = tf.saved_model.load(export_module_dir)
        #model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        print("Loading the model - Done",model)
        end = datetime.now()
        difference = end - start
        print("Loading the model time !!!!!!!!!!!!!!$$$$$$$$$$!!!!!!!!!!!!!========> ",difference.total_seconds())   # Loading the model time ==>  7.744394
        #query_text = cve
        query_embeddings = model([cve])
        embedding = query_embeddings.numpy().tolist()
        update_chromadb_cve_collection(cve)
   
    return embedding


def update_chromadb_cve_collection(cve):
    print("#### Updating the chromaDB CVE collection ######") 




def getcapecsv2_chroma(cve):
    Input_CVE = cve
    start = datetime.now()

    try:
        print("Inside getcapecV2 CHROMA method  = ",cve)
        print("Input_CVE ===> ",Input_CVE)
        if Input_CVE is None:
            return "Error Occured while cve_processing - Input_CVE is None"
        
        query_embeddings = getQueryEmbedding(cve) 

        print("query embedding done for input..now get capec collection and pass as query ")
        collection_db = get_or_create_Collection(client)
        query_result = collection_db.query(
            query_embeddings =query_embeddings,
            n_results=5,
             )

        res = dict(list(query_result.items())[:2])
        return res

    except Exception as e:
        print(f"Error Occured while getcapecsV2 CHROMA - ,{e}")
        return "Error Occured while getcapecsV2 CHROMA";




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



@app.route('/capecs/<string:name>/')
def get_capecs(name):
    if request.method == 'GET':
        result = getcapecs_with_score(name)
        print("=======> ",result)
        if "Error Occured while cve_processing" in result:
            result = "Error Occured, Please try again";
        else:
            print("No Error block!")
            return jsonify(result)      
    return jsonify(result)




@app.route('/capecs/v2/<string:name>/')
def get_capecs_v2(name):
    if request.method == 'GET':
        result = getcapecsv2(name)
        print("=======> ",result)
        if "Error Occured while cve_processing" in result:
            result = "Error Occured, Please try again";
        else:
            print("No Error block!")
            return jsonpickle.encode(result)
    return jsonpickle.encode(result)


@app.route('/capecs/v3/<string:name>/')
def get_capecs_v3(name):
    if request.method == 'GET':
        result = getcapecsv2_chroma(name)
        print("=======> ",result)
        if "Error Occured while cve_processing" in result:
            result = "Error Occured, Please try again";
        else:
            print("No Error block!")
            return jsonpickle.encode(result)
    return jsonpickle.encode(result)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8082)
