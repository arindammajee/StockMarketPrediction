import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
stopwords.words('english')
import re
import pandas as pd

class SentimentAnalysis:
    def __init__(self, path):
        self.data_path = path
        self.newsDataFrame = pd.read_csv(self.data_path)
        self.news = ""

    def text_cleaning(text):
        forbidden_words = set(stopwords.words('english'))
        if text:
            text = ' '.join(text.split('.'))
            text = re.sub('\/',' ',text)
            text = re.sub(r'\\',' ',text)
            text = re.sub(r'((http)\S+)','',text)
            text = re.sub(r'\s+', ' ', re.sub('[^A-Za-z]', ' ', text.strip().lower())).strip()
            text = re.sub(r'\W+', ' ', text.strip().lower()).strip()
            text = [word for word in text.split() if word not in forbidden_words]
            return text
        return []
    
    def cleanedData(self):
        self.newsDataFrame['Title'] = self.newsDataFrame['Title'].apply(lambda x: ' '.join(SentimentAnalysis.text_cleaning(x)))
        for title in self.newsDataFrame['Title']:
            self.news = self.news + '. ' + title

 
