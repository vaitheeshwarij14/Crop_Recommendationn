from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
from flask_cors import CORS

# Initialize the Flask app 
app = Flask(__name__)

# Enable CORS for the Flask app
CORS(app, resources={r"/predict": {"origins": "*"}})

# Load the trained model
model = joblib.load('crop_prediction_model.pkl')

# Define the home route
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON data
        data = request.get_json()
        required_fields = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall']

        # Validate data
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f"Missing field: {field}"}), 400

        # Get the form data, matching exactly with the feature names used during training
        input_data = {
            'Nitrogen': [float(data['Nitrogen'])],
            'Phosphorus': [float(data['Phosphorus'])],
            'Potassium': [float(data['Potassium'])],
            'Temperature': [float(data['Temperature'])],
            'Humidity': [float(data['Humidity'])],
            'pH_Value': [float(data['pH_Value'])],
            'Rainfall': [float(data['Rainfall'])]
        }

        # Convert to DataFrame to match the model's expected input format
        input_df = pd.DataFrame(input_data)

        # Make the prediction
        predicted_crop = model.predict(input_df)

        # Return the result as JSON
        return jsonify({'crop': predicted_crop[0]})

    except KeyError as e:
        return jsonify({'error': f"KeyError: {str(e)}"}), 400
    except ValueError as e:
        return jsonify({'error': f"ValueError: {str(e)}"}), 400
    except Exception as e:
        app.logger.error(f"Exception occurred: {str(e)}")  # Log the exception
        return jsonify({'error': f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
