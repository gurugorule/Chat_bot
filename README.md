# Chat Bot
Demo chatbot using PyTorch. 

## Installation

### Create an python virtual environment and install required packages
```console
pip install -r requirements.txt
```
### May be you need to uncomment the 5th line from 'nltk_utils.py' and run that file it will Download `nltk punkt`:
 ```console
5   nltk.download('punkt')
```

## To Run The Chat Bot

### First Run 'train.py' to train the Model 
```console
python train.py
```
*NOTE*
```console
    If You find any Erron in Training the Model make sure you have set the right path for the symptoms.json data on line no. 11 -> *train.py* 
```


### Run 'app.py' to run the bot
```console
python app.py
```
