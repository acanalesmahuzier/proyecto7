# =========================================
# 🧠 API de Análisis de Sentimientos (3 clases)
# =========================================
from flask import Flask, request, jsonify
import joblib

# ==============================
# 1️⃣ Inicializar la aplicación
# ==============================
app = Flask(__name__)

# ==============================
# 2️⃣ Cargar modelo y vectorizador
# ==============================
modelo = joblib.load("modelo.pkl")
#vectorizer = joblib.load("vectorizer_tfidf.pkl")

# ==============================
# 3️⃣ Ruta de prueba (inicio)
# ==============================
@app.route('/')
def home():
    return jsonify({
        "mensaje": "✅ API de Análisis de Sentimientos (3 clases) funcionando correctamente",
        "uso": "Envía un POST a /predict con un JSON {'texto': 'tu frase aquí'}"
    })

# ==============================
# 4️⃣ Endpoint de predicción
# ==============================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Leer datos del cliente
        data = request.get_json()

        # Validar entrada
        if not data or 'texto' not in data:
            return jsonify({"error": "Falta el campo 'texto' en el JSON"}), 400

        texto = data['texto']

        # Transformar texto con el vectorizador
        #X = vectorizer.transform([texto])
        #Aca para limpiar el texto
        
        # Predecir con el modelo
        pred = int(modelo.predict(X)[0])

        # Asignar etiquetas según clase
        if pred == 0:
            sentimiento = "😞 Sentimiento NEGATIVO"
        elif pred == 1:
            sentimiento = "😐 Sentimiento NEUTRAL"
        elif pred == 2:
            sentimiento = "😊 Sentimiento POSITIVO"
        else:
            sentimiento = "🤔 Clase desconocida"

        # Respuesta final
        return jsonify({
            "texto": texto,
            "sentimiento": sentimiento,
            "valor": pred
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==============================
# 5️⃣ Ejecutar el servidor Flask
# ==============================
if __name__ == '__main__':
    app.run(debug=True, port=8000)