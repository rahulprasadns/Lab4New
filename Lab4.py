from flask import Flask, request, render_template,jsonify
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('home.html')
    
def predict_model (Length1,Length2, Length3, Height, Width):
    pickled_model = pickle.load(open('model.pkl', 'rb'))
    print(pd.DataFrame(np.array([[Length1,Length2, Length3, Height, Width]]), columns=['Length1', 'Length2', 'Length3', 'Height', 'Width']))
    print(pickled_model.predict(pd.DataFrame(np.array([[Length1,Length2, Length3, Height, Width]]), columns=['Length1', 'Length2', 'Length3', 'Height', 'Width']))[0])
    return pickled_model.predict(pd.DataFrame(np.array([[Length1,Length2, Length3, Height, Width]]), columns=['Length1', 'Length2', 'Length3', 'Height', 'Width']))[0]

@app.route('/Lab4', methods=['GET','POST'])
def my_form_post():
    print(request.form)
    Length1 = int(request.form['Length1'])
    Length2 = int(request.form['Length2'])
    Length3 = int(request.form['Length3'])
    Height = int(request.form['Height'])
    Width = int(request.form['Width'])
    weight = predict_model(Length1,Length2, Length3, Height, Width)
    result = {
        "output": weight
    }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)