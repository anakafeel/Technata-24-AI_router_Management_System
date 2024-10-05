from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the saved models
lr_model = joblib.load('linear-regression-model.pkl')
rf_model = joblib.load('random-forest-model.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    # Validate input
    if 'input' not in data or not isinstance(data['input'], list):
        return jsonify({'error': 'Invalid input. Input should be a list of features.'}), 400

    input_data = np.array(data['input']).reshape(1, -1)

    # Validate the number of features
    if input_data.shape[1] != 4:  # Assuming the model was trained on 4 features
        return jsonify({'error': f'Invalid number of features. Expected 4, got {input_data.shape[1]}.'}), 400

    # Check which model to use
    model_type = data.get('model', 'linear')  # default to linear regression

    if model_type == 'random_forest':
        prediction = rf_model.predict(input_data)
    else:
        prediction = lr_model.predict(input_data)

    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
