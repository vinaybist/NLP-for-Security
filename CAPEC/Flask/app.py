from flask import *
import os
import numpy as np



app = Flask(__name__)

def cve_processing(cve):

    print("Inside pythin logic od cosine similarity = ",cve)
    
    return cve


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_CVE = request.form.get('name')
        print('Input CVE',input_CVE)
        result = cve_processing(input_CVE)
        
        return result



if __name__ == '__main__':
    app.run(debug=True)
