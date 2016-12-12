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

import nltk, re, string
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from nltk import word_tokenize
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

#Selected categories from 20newsgroups
categories = ['soc.religion.christian', 'misc.forsale',
'talk.politics.misc', 'comp.sys.ibm.pc.hardware', 'rec.sport.hockey']

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
names = 'gary gary2 jesus jesus2 shop tech hockey hockey2'.split()
docs_new = [extract_message("C:\\Users\\Cody\\Documents\\Emails\\%s.html" % name)
            for name in names]

#Print out class labels
print (news.target_names)
print()

#Build dictionary of features
count_vect = CountVectorizer()
x_train_counts = count_vect.fit_transform(news.data)

#Downscaling
tfidf_transformer = TfidfTransformer()
x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)
tf_transformer = TfidfTransformer(use_idf=False).fit(x_train_counts)
x_train_tf = tf_transformer.transform(x_train_counts)

#Train classifier
clf = MultinomialNB().fit(x_train_tfidf, news.target)

#Extract feautures from emails
x_new_counts = count_vect.transform(docs_new)
x_new_tfidf = tfidf_transformer.transform(x_new_counts)

#Predict the categories for each email
predicted = clf.predict(x_new_tfidf)

#Store Files in a category
hockey_emails = []
computer_emails = []
politics_emails = []
tech_emails = []
religion_emails = []
forsale_emails = []

#Print out results and store each email in the appropritate category list
for name, category in zip(names, predicted):
    print('%r ---> %s' % (name, news.target_names[category]))
    if(news.target_names[category] == 'comp.sys.ibm.pc.hardware'):
        computer_emails.append(name)
    if(news.target_names[category] == 'rec.sport.hockey'):
        hockey_emails.append(name)
    if(news.target_names[category] == 'talk.politics.misc'):
        politics_emails.append(name)
    if(news.target_names[category] == 'soc.religion.christian'):
        religion_emails.append(name)
    if(news.target_names[category] == 'misc.forsale'):
        forsale_emails.append(name)
    if(news.target_names[category] == 'comp.sys.ibm.pc.hardware'):
        computer_emails.append(name)

print()
print('Hockey Emails:')
print(hockey_emails)
print('Politics Emails:')
print(politics_emails)
