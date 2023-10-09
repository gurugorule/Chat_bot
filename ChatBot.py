import random
import json
import torch

from model import NuralNetworks
from nltk_utils import tokenize , bag_of_words


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open("./data/symptoms.json")as file:
    intents = json.load(file)

FILE = 'data.pth'
data = torch.load(FILE)

input_size = data['input_size'] 
hidden_size = data['hidden_size']
num_classes =data['num_classes']
all_words = data['all_words']
tags = data['tags']
model_state = data['model_state'] 

model = NuralNetworks(input_size, hidden_size, num_classes).to(device)
model.load_state_dict(model_state)
model.eval()


CHAT_BOT_NAME = 'Guru'
# print('Hey My name is Guru How May i Help You? ')

def get_response(query):
    query =  tokenize(query)
    x = bag_of_words(query , all_words)
    x = x.reshape(1 , x.shape[0])
    x = torch.from_numpy(x)

    output = model(x)

    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return (f"{CHAT_BOT_NAME}: {random.choice(intent['responses'])}")
    else:
        return (f"{CHAT_BOT_NAME}: I Dont UnderStand!")

# while True:
#     query = input("You: ")
#     if query == 'exit':
#         break

#     query =  tokenize(query)
#     x = bag_of_words(query , all_words)
#     x = x.reshape(1 , x.shape[0])
#     x = torch.from_numpy(x)

#     output = model(x)

#     _, predicted = torch.max(output, dim=1)
#     tag = tags[predicted.item()]

#     probs = torch.softmax(output, dim=1)
#     prob = probs[0][predicted.item()]

#     if prob.item() > 0.75:
#         for intent in intents['intents']:
#             if tag == intent["tag"]:
#                 print(f"{CHAT_BOT_NAME}: {random.choice(intent['responses'])}")
#     else:
#         print(f"{CHAT_BOT_NAME}: I Dont UnderStand!")
