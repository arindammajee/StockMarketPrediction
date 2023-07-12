import os
from helper.gnewsscraper import GoogleNews
import csv
from datetime import date

# Collect All Google News
class newsCollection:
    def __init__(self, stock='ADANIPOWER.NS'):
        self.stock = stock
        self.comp_dir = os.path.join(os.path.join(os.getcwd(),'companies'), self.stock)
        self.data_path = os.path.join(self.comp_dir, 'Data')
        self.log_path = os.path.join(self.comp_dir, 'log.txt')
        self.filepath = os.path.join(self.data_path, self.stock + '_news' + '.csv')
    
    def collectGoogleNews(self, addikey = [], country='IND', period='1d'):
        gn = GoogleNews(country)
        searchNews = []
        searchNews.append(gn.search(self.stock, when=period))
        for keyword in addikey:
            searchNews.append(gn.search(keyword, when=period))
        
        rows = []
        for newsObject in searchNews:
            for news in newsObject['entries']:
                splitTitle = news['title'].split('-')
                title = '-'.join(splitTitle[:-1])
                source = splitTitle[-1]
                time = news['published']
                rows.append([title, source, time])
        
        #print(gn.search(self.stock, when=period))    
        # Write the news in a csv file
        with open(self.filepath, 'w') as csvfile: 
            # creating a csv writer object 
            csvwriter = csv.writer(csvfile) 
            
            # writing the fields 
            csvwriter.writerow(['Title', 'News Source', 'Time']) 

            # writing the data rows 
            csvwriter.writerows(rows)

        