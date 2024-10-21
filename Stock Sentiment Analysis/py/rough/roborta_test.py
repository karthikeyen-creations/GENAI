import os
import pickle
import shutil
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request


task='sentiment'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

folder = MODEL.split("/")[0]
if os.path.isdir(folder):
        shutil.rmtree(folder, onexc=lambda func, path, exc: os.chmod(path, 0o777))
        print(f'Deleted existing {folder}')

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.save_pretrained(MODEL)


def sentiment_analyse(text):
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    return scores
    
print(sentiment_analyse("I Love You So Much"))

# # Preprocess text (username and link placeholders)
# def preprocess(text):
#     new_text = []
 
 
#     for t in text.split(" "):
#         t = '@user' if t.startswith('@') and len(t) > 1 else t
#         t = 'http' if t.startswith('http') else t
#         new_text.append(t)
#     return " ".join(new_text)

# Tasks:
# emoji, emotion, hate, irony, offensive, sentiment
# stance/abortion, stance/atheism, stance/climate, stance/feminist, stance/hillary


# labels = ["negative","neutral","positive"]

# # download label mapping
# labels0=[]

# mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
# with urllib.request.urlopen(mapping_link) as f:
#     html = f.read().decode('utf-8').split("\n")
#     csvreader = csv.reader(html, delimiter='\t')
    
# labels0 = [row[1] for row in csvreader if len(row) > 1]

# # Save the list to a file
# with open('Stock Sentiment Analysis/files/labels.pkl', 'wb') as file:
#     pickle.dump(labels0, file)

# labels=[]

# with open('Stock Sentiment Analysis/files/labels.pkl', 'rb') as file:
#     labels = pickle.load(file)

# PT

# text = "I Love You So Much"
# text = preprocess(text)

# # TF
# model = TFAutoModelForSequenceClassification.from_pretrained(MODEL)
# model.save_pretrained(MODEL)

# text = "Good night 😊"
# encoded_input = tokenizer(text, return_tensors='tf')
# output = model(encoded_input)
# scores = output[0][0].numpy()
# scores = softmax(scores)

# ranking = np.argsort(scores)
# ranking = ranking[::-1]
# for i in range(scores.shape[0]):
#     l = labels[ranking[i]]
#     s = scores[ranking[i]]
#     print(f"{i+1}) {l} {np.round(float(s), 4)}")
