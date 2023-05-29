from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json

app = Flask(__name__)

@app.route("/classification", methods=["POST"])
def classification():
     if request.method == "POST":
        try:
            user_review = json.loads(request.data)["text"]
            
            model = load_model("app\modelo4.h5")
            maxlen = 256
            
            word_index = imdb.get_word_index()
            user_tokens = [
                word_index[word] if word in word_index else 0
                for word in user_review.split()
            ]
    
            user_padded = pad_sequences([user_tokens], maxlen=maxlen)
    
            prediction = model.predict(user_padded)[0][0]
            if prediction >= 0.5:
                print(f"Prediction: {prediction}")
                print("Positive review!")
                mensaje = "Positive review!"
                return mensaje
            else:
                print(f"Prediction: {prediction}")
                print("Negative review.")
                mensaje = "Negative review."
                return mensaje
        except Exception as e:
            error_message = "ERROR 500 OCURRED : " + str(e)
            return error_message, 500

   

if __name__ == "__main__":
    app.run(debug=True)
    #app.run(debug=True, host='0.0.0.0')
