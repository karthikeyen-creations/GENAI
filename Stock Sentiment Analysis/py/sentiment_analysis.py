import os
import random
import shutil
from langchain.schema import Document
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from scipy.special import softmax

class SentimentAnalyzer:
    def __init__(self, task='sentiment', model_name=None):
        """
        Initialize the SentimentAnalyzer class.
        
        Args:
        task (str): The task for the model (default is 'sentiment').
        model_name (str): The name of the model to use. If None, uses the default model.
        """
        self.task = task
        if model_name is None:
            self.MODEL = f"cardiffnlp/twitter-roberta-base-{self.task}"
        else:
            self.MODEL = model_name
        
        self._setup_model()
        self._load_model()
        
    def _setup_model(self):
        """Remove existing model folder if it exists."""
        folder = self.MODEL.split("/")[0]
        if os.path.isdir(folder):
            shutil.rmtree(folder, onexc=lambda func, path, exc: os.chmod(path, 0o777))
            print(f'Deleted existing {folder}')

    def _load_model(self):
        """Load the tokenizer and model."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL)
        self.model.save_pretrained(self.MODEL)

    def analyze(self, text):
        """
        Analyze the sentiment of the given text.
        
        Args:
        text (str): The input text to analyze.
        
        Returns:
        numpy.ndarray: An array of sentiment scores (negative, neutral, positive).
        """
        encoded_input = self.tokenizer(text, return_tensors='pt')
        output = self.model(**encoded_input)
        scores = output[0][0].detach().numpy()
        return softmax(scores)

    def interpret_scores(self, scores):
        """
        Interpret the sentiment scores.
        
        Args:
        scores (numpy.ndarray): The sentiment scores from the analyze method.
        
        Returns:
        str: A string interpretation of the sentiment.
        """
        labels = ['Negative', 'Neutral', 'Positive']
        return labels[np.argmax(scores)]

def sentiment_analyse(ingested_data):
    analyzer = SentimentAnalyzer()
    analysed_data = []
    # for doc in random.sample(ingested_data,5):
    for doc in ingested_data:
        score = analyzer.analyze(doc.page_content)
         # Create a copy of the existing metadata
        updated_metadata = doc.metadata.copy()
        # Update each specified metadata field
        updated_metadata["negative"] = score[0]
        updated_metadata["neutral"] = score[1]
        updated_metadata["positive"] = score[2]
        # Create a new Document instance with the updated metadata
        updated_doc = Document(page_content=doc.page_content, metadata=updated_metadata)
        analysed_data.append(updated_doc)
    return analysed_data
    
# Example usage
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    
    test_text = "I Love You So Much"
    scores = analyzer.analyze(test_text)
    interpretation = analyzer.interpret_scores(scores)
    
    print(f"Text: {test_text}")
    print(f"Scores: {scores}")
    print(f"Interpretation: {interpretation}")