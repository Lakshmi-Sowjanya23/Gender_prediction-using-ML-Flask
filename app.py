from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    confidence = ""

    if request.method == "POST":
        name = request.form["name"]

        name_vec = vectorizer.transform([name])

        prediction = model.predict(name_vec)[0]

        probability = model.predict_proba(name_vec)

        confidence = round(max(probability[0]) * 100, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence
    )

if __name__ == "__main__":
    app.run(debug=True)