from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load your model and scaler
model = joblib.load("XGBC.pkl")
scaler = joblib.load("scaler.pkl")

# Iris species mapping
species = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

# Acceptable ranges for features (example ranges, adjust based on dataset specifics)
acceptable_ranges = {
    'sepal_length': (4.3, 7.9),
    'sepal_width': (2.0, 4.4),
    'petal_length': (1.0, 6.9),
    'petal_width': (0.1, 2.5)
}

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract and convert data from form
        input_features = [float(x) for x in request.form.values()]
        feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        
        # Check for outliers
        for i, feature in enumerate(feature_names):
            if not (acceptable_ranges[feature][0] <= input_features[i] <= acceptable_ranges[feature][1]):
                return render_template("result.html", prediction="Invalid input: value for {} out of allowed range".format(feature))
        
        # Prepare features for prediction
        features = np.array(input_features).reshape(1, -1)
        scaled_features = scaler.transform(features)
        
        # Predict using the model and get confidence
        prediction = model.predict(scaled_features)
        probabilities = model.predict_proba(scaled_features)[0]
        confidence = np.max(probabilities)
        
        # Convert numerical prediction to species name
        species_name = species[prediction[0]]
        confidence_percent = round(confidence * 100, 0)
        
        return render_template("result.html", prediction=species_name, confidence_percent=confidence_percent)
    
    except Exception as e:
        return render_template("result.html", prediction=str(e))

if __name__ == "__main__":
    app.run(debug=True)
