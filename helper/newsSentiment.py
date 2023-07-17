import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
stopwords.words('english')
import re
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class SentimentAnalysis:
    def __init__(self, path):
        self.data_path = path
        self.newsDataFrame = pd.read_csv(self.data_path)

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
    
    def sentiment_scores(self):
        # Create a SentimentIntensityAnalyzer object.
        sid_obj = SentimentIntensityAnalyzer()
        avgNeg = 0
        avgNeu = 0
        avgPos = 0
        for title in self.newsDataFrame['Title']:
            sentiment_dict = sid_obj.polarity_scores(title)
            avgNeg = avgNeg + sentiment_dict['neg']
            avgNeu = avgNeu + sentiment_dict['neu']
            avgPos = avgPos + sentiment_dict['pos']
        
        if len(self.newsDataFrame['Title'])!=0:
            avgNeg = (1.0*avgNeg)/len(self.newsDataFrame['Title'])
            avgNeu = (1.0*avgNeu)/len(self.newsDataFrame['Title'])
            avgPos = (1.0*avgPos)/len(self.newsDataFrame['Title'])
        
        print("Negative Sentiment score {} % ".format(avgNeg*100))
        print("Neutral Sentimrnt score {} % ".format(avgNeu*100))
        print("Positive Sentiment score {} %".format(avgPos*100))
        
        Sentiment = [avgNeg, avgNeu, avgPos]
        if max(Sentiment) == avgNeg:
           return "be Downwards!. The news contain almost {} negative sentiment".format(avgNeg)
        elif max(Sentiment) == avgNeu:
           return "not change as much"
        else:
           return "be strongly upwards. The news contain almost {} positive sentiment".format(avgPos)

 
