from flask import Flask, render_template, request
import pickle
app = Flask(__name__)
with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    message = ""
    if request.method == "POST":
        message = request.form.get("message", "")
        data = vectorizer.transform([message])
        result = model.predict(data)[0]
        if result == 1:
            prediction = "Spam ❌"
        else:
            prediction = "Not Spam ✅"
    return render_template(
        "index.html",
        prediction=prediction,
        message=message
    )
if __name__ == "__main__":
   import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)