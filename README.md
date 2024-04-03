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

```markdown
# Chat Bot

This repository contains a demo chatbot implemented using PyTorch.

## Installation

### Create a Python Virtual Environment

To set up the required environment, you can create a Python virtual environment and install the necessary packages listed in the `requirements.txt` file. You can do this by running the following command:

```console
pip install -r requirements.txt
```

### Download NLTK Data

Before running the chatbot, you may need to download the NLTK data. NLTK (Natural Language Toolkit) is a leading platform for building Python programs to work with human language data. In this project, NLTK is used for tokenization. To download the NLTK punkt data, uncomment line 5 in the `nltk_utils.py` file and run the following command:

```console
python nltk_utils.py
```

This will download and install the necessary NLTK punkt data.

## Running the Chat Bot

### Train the Model

Before using the chatbot, you need to train the model. The training data is stored in the `symptoms.json` file. To train the model, run the following command:

```console
python train.py
```

If you encounter any errors during training, please ensure that you have set the correct path for the `symptoms.json` data. The path is specified in line 11 of the `train.py` file.

### Run the Chat Bot

Once the model is trained, you can run the chatbot using the following command:

```console
python app.py
```

This will start the chatbot application, allowing you to interact with it.

**Note:** Make sure you have completed the training step before running the chatbot. The trained model is required for the chatbot to function properly.

---

This README provides clear instructions for installing and running the chatbot. It also includes a note about troubleshooting during the training process.
```

This README provides clear instructions for installing and running the chatbot. It also includes a note about troubleshooting during the training process.
