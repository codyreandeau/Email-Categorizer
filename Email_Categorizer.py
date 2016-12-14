#-------------------------------------------------------------------------------
# Name:        extract_email_text
# Purpose:
#
# Author:      Cody
#
# Created:     28/11/2016
# Copyright:   (c) Cody 2016
# Licence:     <your licence>
#-------------------------------------------------------------------------------

def main():
    pass

if __name__ == '__main__':
    main()

import nltk, re
from bs4 import BeautifulSoup
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

#Selected categories from 20newsgroups
categories = ['soc.religion.christian', 'misc.forsale',
'talk.politics.misc', 'comp.sys.ibm.pc.hardware', 'comp.graphics', 'rec.sport.hockey']

#Load the list of files
news = fetch_20newsgroups(categories=categories)

#Function that strips body from email
def extract_message(url):
    markup = open(url, encoding="utf8")
    soup = BeautifulSoup(markup, "html.parser")
    for script in soup(["script", "style"]):
        script.extract()
    text = soup.get_text()
    text = re.sub(r'[^\x00-\x7F]+',' ', text) #Eliminate Foreign Characters
    text = re.sub(r'\d+', ' ', text) #Eliminate Numbers
    text = re.sub(r'\W+', ' ', text) #Eliminate /n /t characters
    a = text.find('View')
    final_text = text[a:]
    return final_text

#File Names
names = 'gary gary2 jesus jesus2 jesus3 tech tech2 tech3 tech4 hockey hockey2 hockey3 hockey4 shop'.split()
docs_new = [extract_message("C:\\Users\\Cody\\Documents\\EmailCategorizer\\Emails\\%s.html" % name)
            for name in names]

#Build dictionary of features
count_vectorizor = CountVectorizer()
train_counts = count_vectorizor.fit_transform(news.data)

#Downscaling - tf-idf
tfidf_transformer = TfidfTransformer()
train_tfidf = tfidf_transformer.fit_transform(train_counts)
tf_transformer = TfidfTransformer(use_idf=False).fit(train_counts)
train_tf = tf_transformer.transform(train_counts)

#Train classifier
my_classifier = MultinomialNB().fit(train_tfidf, news.target)

#Extract feautures from emails
new_counts = count_vectorizor.transform(docs_new)
new_tfidf = tfidf_transformer.transform(new_counts)

#Predict the categories for each email
predicted_label = my_classifier.predict(new_tfidf)

#Store Files in a category
hockey_emails = []
computer_emails = []
politics_emails = []
tech_emails = []
religion_emails = []
forsale_emails = []

#Print out results and store each email in the appropritate category list
for name, category in zip(names, predicted_label):
    print('%r ---> %s' % (name, news.target_names[category]))
    if(news.target_names[category] == 'comp.sys.ibm.pc.hardware'
    or news.target_names[category] == 'comp.graphics'):
        computer_emails.append(name)
    if(news.target_names[category] == 'rec.sport.hockey'):
        hockey_emails.append(name)
    if(news.target_names[category] == 'talk.politics.misc'):
        politics_emails.append(name)
    if(news.target_names[category] == 'soc.religion.christian'):
        religion_emails.append(name)
    if(news.target_names[category] == 'misc.forsale'):
        forsale_emails.append(name)

print()
print('Religion Emails:')
print(religion_emails)
print('Hockey Emails:')
print(hockey_emails)
print('Politics Emails:')
print(politics_emails)
print('Computer Emails:')
print(computer_emails)
print('Emails Trying to Sell me Stuff:')
print(forsale_emails)
print()
print('EXAMPLE OF EXTRACTED EMAIL:')
print(docs_new[0])

