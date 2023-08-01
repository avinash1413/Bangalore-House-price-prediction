

# flask, scikit-learn, pandas, pickle-mixin, flask-cors
import pandas as pd
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
data = pd.read_csv('cleaned_data.csv')
pipe = pickle.load(open("ridgeModel.pkl", 'rb'))


@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)


@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk_str = request.form.get('bhk')
    bath_str = request.form.get('bath')
    sqft_str = request.form.get('total_sqft')

    if not bhk_str or not bath_str or not sqft_str:
        return "Please fill in all the required fields."

    try:
        bhk = float(bhk_str)
        bath = float(bath_str)
        sqft = float(sqft_str)
    except ValueError:
        return "Invalid input. Please enter valid numeric values for BHK, Bathrooms, and Square Feet."

    print(location, bhk, bath, sqft)
    input = pd.DataFrame([[location, sqft, bhk, bath]], columns=['location', 'total_sqft', 'bath', 'bhk'])
    prediction = pipe.predict(input)[0] * 100000

    return str(np.round(prediction, 2))


if __name__ == "__main__":
    app.run(debug=True, port=5001)
