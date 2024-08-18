import numpy as np
import random
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader

from pipeline import bag_of_words, tokenize,stem
from model import NeuralNet

with open('intents.json', 'r') as f:
    intents = json.load(f)
    
all_words = []
tags = []
word_tag = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)

    for pattern in intent['patterns']:
        word = tokenize(pattern)
        all_words.extend(word)
        
        word_tag.append((word,tag))
        

ignore_words = ['?','.','!']

all_words = [stem(word) for word in all_words if word not in ignore_words]
all_words = sorted(set(all_words))

tags = sorted(set(tags))