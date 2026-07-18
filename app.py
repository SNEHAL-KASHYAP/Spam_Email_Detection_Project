from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    message = request.form["message"]

    # Convert text to TF-IDF features
    data = vectorizer.transform([message])

    # Predict
    prediction = model.predict(data)

    # Prediction probability
    probability = model.predict_proba(data)

    if prediction[0] == 1:
        result = "🚨 Spam Email"
        color = "spam"
        confidence = round(probability[0][1] * 100, 2)
    else:
        result = "✅ Not Spam"
        color = "ham"
        confidence = round(probability[0][0] * 100, 2)

    return render_template(
        "index.html",
        prediction=result,
        message=message,
        confidence=confidence,
        color=color
    )


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)