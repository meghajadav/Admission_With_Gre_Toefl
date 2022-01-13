import logging
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS, cross_origin
import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np

app = Flask(__name__)
# logging.basicConfig(filename='admission.log', filemode='w', level=logging.INFO)
# log = logging.getLogger(__name__)

@app.route('/', methods=["POST", "GET"])
def home_page():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def index():
    if request.method=="POST":
        try:
            gre_score = float(request.form['gre_score'])
            toefl_score = float(request.form['toefl_score'])
            univ_rating = float(request.form['univ_rating'])
            sop = float(request.form['sop'])
            lor = float(request.form['lor'])
            cgpa = float(request.form['cgpa'])
            isresearch = request.form['research']
            if isresearch=='yes':
                research=1
            else:
                research=0
            filename = 'final_lr_model.pkl'
            Standard_scaler_file = "Standard_Scaler_obj.pkl"
            sc = pickle.load(open(Standard_scaler_file,'rb'))
            model = pickle.load(open(filename,'rb'))
            arr = sc.transform([[gre_score, toefl_score, univ_rating, sop, lor, cgpa, research]])
            pred = model.predict(arr)
            print(pred)
            return render_template("results.html", prediction=float(np.round(100*pred,2)))
        except Exception as e:
            logging.info(e)
            return"Something went wrong"
        else:
            return render_template("index.html")

if __name__=="__main__":
    app.run()


