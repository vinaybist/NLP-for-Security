from flask import *
import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import jsonpickle

app = Flask(__name__)

def sent_embedings(input, model):
    return model(input)

def cve_processing(cve):

    print("Inside python logic of cosine similarity = ",cve)
    capec_data_uri = "./id_desc.csv"
    capec_data = pd.read_csv(capec_data_uri, encoding='utf-8')
    print(type(capec_data))
    capec_data = capec_data.dropna()
    capec_data = capec_data.sort_values(by=['ID'], ascending=True)
    capec_data = capec_data.reset_index(drop=True)
    print("Sorting the dataframe")
    capec_data['ID'] = "CAPEC-"+capec_data['ID'].apply(str)
    dict_of_capecs = capec_data.set_index('ID').to_dict()['Description']
    print("Creating dict of capecs")
    Input_CVE_2018_18442 = "D-Link DCS-825L devices with firmware 1.08 do not employ a suitable mechanism to prevent denial-of-service (DoS) attacks. An attacker can harm the device availability (i.e., live-online video/audio streaming) by using the hping3 tool to perform an IPv4 flood attack. Verified attacks includes SYN flooding, UDP flooding, ICMP flooding, and SYN-ACK flooding."
    Input_doc_dict = {'input_doc':Input_CVE_2018_18442}
    dict_of_corpus = {**dict_of_capecs, **Input_doc_dict}
    corpus_keys = list(dict_of_corpus.keys())
    print("Loading the model...")
    model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    print("Loading the model - Done")
    sent_list_1 = list(dict_of_corpus.values())
    ls_dict_of_capecs_1 = list(dict_of_corpus.keys())
    print(type(ls_dict_of_capecs_1))
    dicts_vec_1 = sent_embedings(sent_list_1, model)
    final_dict_vec_1 = {ls_dict_of_capecs_1[i]: dicts_vec_1[i] for i in range(len(ls_dict_of_capecs_1))}
    cs1_1 = cosine_similarity([final_dict_vec_1['input_doc']], list(final_dict_vec_1.values()))
    print("Got the similarity scores")
    df1_1 = pd.DataFrame(cs1_1, columns =corpus_keys)
    response_dict_1 = df1_1.to_dict('index')[0]
    sorted_CAPEC_by_maximum_similarity_1 = sorted(response_dict_1.items(), reverse=True, key=lambda x:x[1])
    sorted_CAPEC_by_maximum_similarity_1
    print("Sort the list - based on maximum similarity ")
    return_capec_list = [i[0] for i in sorted_CAPEC_by_maximum_similarity_1[1:10]]
    print("returning the ",type(return_capec_list))
    #return ', '.join(return_capec_list)
    return return_capec_list


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_CVE = request.form.get('name')
        print('Input CVE',input_CVE)
        result = cve_processing(input_CVE)
        
        return jsonpickle.encode(result)




if __name__ == '__main__':
    app.run(debug=True)
