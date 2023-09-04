import json
import numpy as np
from nltk_utils import tokenize , stemming , bag_of_words

import torch
import torch.nn as nn
from torch.utils.data import Dataset , DataLoader 

from model import NuralNetworks

with open('.symptoms.json')as file:
    pattern = json.load(file)

all_words = []
tags = []
both_patterns_tags = []

for intent in pattern['intents']:
    tag = intent['tag']
    tags.append(tag)
    for patter in intent["patterns"]:
        w = tokenize(patter)
        all_words.extend(w)
        both_patterns_tags.append((w ,tag))


ignore_symbols = ['$', ',', '.', '!', '?']
all_words = [stemming(words) for words in all_words if words not in ignore_symbols ]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

x_train = []
y_train = []

for (patterns_sentance , tag) in both_patterns_tags:
    bag = bag_of_words(patterns_sentance , all_words)
    x_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)

x_train = np.array(x_train)
y_train = np.array(y_train)


class ChatDataSet(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index] , self.y_data[index]

    def __len__(self):
        return self.n_samples


batch_size = 8
input_size = len(x_train[0])
hidden_size = 8 
num_classes =  len(tags)
learning_rate = 0.001
num_epochs = 1000


dataset = ChatDataSet()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NuralNetworks(input_size, hidden_size, num_classes).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words ,labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        outputs = model(words)
        loss = criterion(outputs , labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if(epoch+1)%100 ==0:
        print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}')


print(f'Final Loss, loss={loss.item():.4f}') 

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"num_classes": num_classes,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')
