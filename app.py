from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load('healthcare_model.pkl')
l1 = joblib.load('gender_le.pkl')
l2 = joblib.load('smoker_le.pkl')

# you are routing your web page to go through this app so that we can connect it with python

@app.route('/', methods = ['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/prediction', methods = ['POST'])
def prediction():
    age = request.form['age']
    gender = request.form['gender']
    bmi = request.form['bmi']
    smoker = request.form['smoker']
    # converting string form values to floating point
    age, bmi = float(age), float(bmi)
    gender = l1.transform([[gender]])[0]
    smoker = l2.transform([[smoker]])[0]
    result = model.predict([[age, gender, bmi, smoker]])
    string = 'The Healtcare Premium Charges are: ' + str(result[0])
    return render_template('index.html', prediction_text = string)

# running the app
if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 5000, threaded = False)