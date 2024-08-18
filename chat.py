import random
import json
import torch

from model import NeuralNet
from pipeline import bag_of_words,tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)
    
model_dict = "model.pth"
model_vals = torch.load(model_dict)

input_size = model_vals["input_size"]
hidden_size = model_vals["hidden_size"]
output_size = model_vals["output_size"]
all_words = model_vals['all_words']
tags = model_vals['tags']
model_state = model_vals["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = 'AWS Bot'

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    
    tag = tags[predicted.item()]
    
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent['tag']:
                return random.choice(intent['responses'])
            
    return "I do not know"