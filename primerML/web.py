from flask import Flask
from flask import request
from flask_cors import CORS
from flask import jsonify
import pickle
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)
CORS(app)

@app.route("/m1", methods=['post'])
def m1():
    #load the model from disk
    filename = 'models/model.pkl'
    model = pickle.load(open(filename, 'rb'))

    #load the scaler from disk
    filename = 'models/scaler.pkl'
    scaler = pickle.load(open(filename, 'rb'))
    a1=request.form.get("a1")
    a2=request.form.get("a2")
    a3=request.form.get("a3")
    a4=request.form.get("a4")
    a5=request.form.get("a5")
    a6=request.form.get("a6")
    a7=request.form.get("a7")
    a8=request.form.get("a8")
    a9=request.form.get("a9")
    a10=request.form.get("a10")
    a11=request.form.get("a11")
    a12=request.form.get("a12")
    a13=request.form.get("a13")
    a14=request.form.get("a14")
    a15=request.form.get("a15")
    a16=request.form.get("a16")
    a17=request.form.get("a17")
    a18=request.form.get("a18")
    a19=request.form.get("a19")
    a20=request.form.get("a20")
    a21=request.form.get("a21")
    a22=request.form.get("a22")
    a23=request.form.get("a23")
    a24=request.form.get("a24")
    a25=request.form.get("a25")
    a26=request.form.get("a26")
    a27=request.form.get("a27")
    a28=request.form.get("a28")
    a29=request.form.get("a29")
    a30=request.form.get("a30")
    a31=request.form.get("a31")
    a32=request.form.get("a32")
    a33=request.form.get("a33")
    a34=request.form.get("a34")
    a35=request.form.get("a35")
    a36=request.form.get("a36")
    a37=request.form.get("a37")
    a38=request.form.get("a38")
    a39=request.form.get("a39")
    a40=request.form.get("a40")
    a41=request.form.get("a41")
    a42=request.form.get("a42")
    a43=request.form.get("a43")
    a44=request.form.get("a44")
    a45=request.form.get("a45")
    a46=request.form.get("a46")
    a47=request.form.get("a47")
    a48=request.form.get("a48")
    a49=request.form.get("a49")
    a50=request.form.get("a50")
    a51=request.form.get("a51")
    a52=request.form.get("a52")
    a53=request.form.get("a53")
    a54=request.form.get("a54")
    a55=request.form.get("a55")
    a56=request.form.get("a56")
    a57=request.form.get("a57")
    a58=request.form.get("a58")
    a59=request.form.get("a59")
    a60=request.form.get("a60")
    
    data = [[a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31,a32,a33,a34,a35,a36,a37,a38,a39,a40,a41,a42,a43,a44,a45,a46,a47,a48,a49,a50,a51,a52,a53,a54,a55,a56,a57,a58,a59,a60]]
    
    rescaledValidationX = scaler.transform(data)#aqui
    predictions = model.predict(rescaledValidationX)
  
    

  
    
    return  jsonify({"predictions": predictions.tolist()})



@app.route("/m2", methods=['post'])
def m2():

    #load the model from disk
    filename = 'models/modelm2.pkl'
    modelo = pickle.load(open(filename, 'rb'))

    #load the scaler from disk
    filename = 'models/scalerm2.pkl'
    scalar = pickle.load(open(filename, 'rb'))


    # transform the validation dataset
    #data=[[0.00632,18,2.31,0,0.538,6.575,65.2,4.09,1,296,15.3,4.98]]

    a1=request.form.get("a1")
    a2=request.form.get("a2")
    a3=request.form.get("a3")
    a4=request.form.get("a4")
    a5=request.form.get("a5")
    a6=request.form.get("a6")
    a7=request.form.get("a7")
    a8=request.form.get("a8")
    a9=request.form.get("a9")
    a10=request.form.get("a10")
    a11=request.form.get("a11")
    a12=request.form.get("a12")
    
    data = [[a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12]] 


    data2 = scalar.transform(data) #aqui
    predictions = modelo.predict(data2)

    return jsonify({"predictions": predictions.tolist()})




@app.route("/m3", methods=['post'])
def m3():

    #load the model from disk
    filename = 'models/modelm3.pkl'
    model = pickle.load(open(filename, 'rb'))

    #load the scaler from disk
    filename = 'models/scalerm3.pkl'
    scaler = pickle.load(open(filename, 'rb'))


    # transform the validation dataset
    #data=[[0.00632,18,2.31,0,0.538,6.575,65.2,4.09,1,296,15.3,4.98]]

    a1=request.form.get("a1")
    a2=request.form.get("a2")
    a3=request.form.get("a3")
    a4=request.form.get("a4")
    
    
    data = [[a1,a2,a3,a4]] 


    data2 = scaler.transform(data) #aqui
    predictions = model.predict(data2)

    if (predictions.tolist()==[0]):
     predictions=['Iris-setosa']

    elif (predictions.tolist()==[1]):
        predictions=['Iris-versicolor']
    elif (predictions.tolist()==[2]):
        predictions=['Iris-virginica']

    return jsonify({"predictions": predictions})
    

@app.route("/")
def m():
    return 'No hay nada aqui :)'


if __name__=="__main__":
    app.run(debug=True)