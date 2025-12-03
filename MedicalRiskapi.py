from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

# Charger le dictionnaire
data = joblib.load("rf_optimized.pkl")
model = data['model']
threshold = data['threshold']

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({
        "message": "API ML déployée sur Render",
        "status": "online",
        "endpoints": ["POST /predict"],
        "model": type(model).__name__,
        "threshold": float(threshold)
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Vérifier que l'API fonctionne"""
    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "threshold": float(threshold)
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        features = np.array(data["features"]).reshape(1, -1)

        pred_proba = model.predict_proba(features)[:, 1]
        prediction = int(pred_proba[0] >= threshold)

        return jsonify({"prediction": prediction})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Utiliser le port de Render ou 5000 par défaut
    port = int(os.environ.get('PORT', 5000))
    # '0.0.0.0' est IMPORTANT pour Render
    app.run(host='0.0.0.0', port=port, debug=False)
