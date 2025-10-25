from flask import Flask, request, jsonify
import joblib, os
import numpy as np

app = Flask(__name__)

MODEL_PATH = "sentiment_model.joblib"
VECT_PATH  = "vectorizer.joblib"

modelo = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECT_PATH)

if not hasattr(vectorizer, "idf_"):
    raise RuntimeError("El vectorizer.joblib no está ajustado (falta idf_). Vuelve a guardarlo tras fit.")

@app.route("/")
def home():
    return jsonify({
        "mensaje": "✅ API de Análisis de Sentimientos (:) 3 clases) funcionando correctamente",
        "uso": "Envía un POST a /predict con un JSON {'texto': 'tu frase aquí'}"
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True) or {}
        if "texto" not in data:
            return jsonify({"error": "Falta el campo 'texto'"}), 400

        texto = data["texto"]
        if not isinstance(texto, str) or not texto.strip():
            return jsonify({"error": "El campo 'texto' debe ser un string no vacío"}), 400

        X = vectorizer.transform([texto])

        # Predicción (etiqueta 'positivo'/'neutral'/'negativo')
        pred = modelo.predict(X)[0]

        confianza_pct = None
        probs_pct = None

        if hasattr(modelo, "predict_proba"):
            proba = modelo.predict_proba(X)[0]           # array de probs por clase
            cls = modelo.classes_                         # orden de clases
            # prob. de la clase predicha:
            conf = proba[np.argmax(proba)]
            confianza_pct = round(float(conf) * 100, 2)   # porcentaje
            # (opcional) todas las clases en porcentaje
            probs_pct = {str(c): round(float(p)*100, 2) for c, p in zip(cls, proba)}

        elif hasattr(modelo, "decision_function"):
            # Nota: no son probabilidades calibradas. Para porcentajes reales usa CalibratedClassifierCV
            score = modelo.decision_function(X)
            # manejo binario vs multiclase
            if np.ndim(score) == 1:
                # escala min-max para que al menos sea interpretable (no probabilístico)
                s = float(score[0])
                confianza_pct = "No disponible (usar predict_proba/calibración)"
            else:
                confianza_pct = "No disponible (usar predict_proba/calibración)"
        else:
            confianza_pct = "No disponible (modelo no soporta probabilidades)"

        return jsonify({
            "texto": texto,
            "sentimiento": pred,
            "confianza_pct": confianza_pct,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
