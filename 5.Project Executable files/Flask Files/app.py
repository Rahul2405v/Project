from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)



loaded_model = pickle.load(open("CO2predection","rb"))


loaded_label_encoder1 =  pickle.load(open("CountryName","rb"))

loaded_label_encoder2 = pickle.load(open("CountryCode","rb"))


loaded_label_encoder3 =  pickle.load(open("IndicatorName","rb"))

@app.route('/')
def loadpage():
    return render_template("index.html")

@app.route('/predict', methods=["POST","GET"])
def loadValue():
    
        if request.method == 'POST':
            Indicator = request.form['indicator']
            year = request.form['year']
            countryName = request.form['countryName']
            countryCode = request.form['countryCode']
            transformed_Name = loaded_label_encoder1.transform([countryName])[0]
            tranformed_Code = loaded_label_encoder2.transform([countryCode])[0]
            transformed_Indicator = loaded_label_encoder3.transform([Indicator])[0]
            
            
            x_test = np.array([[int(transformed_Name), int(tranformed_Code), int(transformed_Indicator), int(year)]])
            print(x_test)  
    
            
            prediction = loaded_model.predict(x_test)
    
            
            return render_template("index.html", prediction_text=prediction[0])  
   
if __name__ == "__main__":
    app.run(debug=False)
