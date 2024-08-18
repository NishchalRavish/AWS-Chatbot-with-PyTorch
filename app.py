from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from chat import get_response, bot_name

app = Flask(__name__)
CORS(app)

messages = []

USER = "Demo User"

@app.get('/')
def index_get():
    return render_template('base.html', messages=reversed(messages))

@app.post('/')
def index_post():
    if request.method =='POST':
        text = request.form['text']
        
        new_message = {"name": USER, "msg": text}
        messages.append(new_message)
        
        response = get_response(text)
        new_message = {"name": bot_name, "msg": response}
        messages.append(new_message)
        
        return index_get()
    
@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == "POST":
        print(request.get_json())
        
        text = request.get_json().get("message")
        
        response = get_response(text)
        message = {'answer': response}
        
        return jsonify(message)
    
if __name__ == "__main__":
    app.run(debug=True)