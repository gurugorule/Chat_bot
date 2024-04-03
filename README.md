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
## Viewing the Chat Bot
- After running app.py, open localhost:8080 in your web browser to see the chat bot in action.
***
This will start the chatbot application, allowing you to interact with it.
This README provides clear instructions for installing and running the chatbot. It also includes a note about troubleshooting during the training process.

