from flask import Flask,request,jsonify,render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# importing ridge regressor model and standard scaler pickle file
ridge_model = pickle.load(open("models/ridge.pkl","rb"))
standard_scaler = pickle.load(open("models/scaler.pkl","rb"))

@app.route("/")
def index():
    return render_template("index.html")


# Get -> retrieving 
# Post -> query/input/output based on some data input


@app.route("/predictdata",methods = ["GET","POST"]) # GET - displaying the home.html , POST - submitting the form
def predict_datapoint():
    if request.method == "POST":
        Temperature = float(request.form.get("Temperature"))
        RH = float(request.form.get("RH"))
        Ws = float(request.form.get("Ws"))
        Rain = float(request.form.get("Rain"))
        FFMC = float(request.form.get("FFMC"))
        DMC = float(request.form.get("DMC"))
        ISI = float(request.form.get("ISI"))
        Classes = float(request.form.get("Classes"))
        Region = float(request.form.get("Region"))

        new_data_scaled = standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result = ridge_model.predict(new_data_scaled) # result stores in list format
        return render_template("home.html",result = result[0])

    else:
        return render_template("home.html")

if __name__ == "__main__":
    app.run(host = "0.0.0.0",debug=True)