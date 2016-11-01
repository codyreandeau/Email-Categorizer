''' This method extracts an emails
contents, while eliminating stopwords, newline 
characters, /xa0 characters, then tokenizes
the words, which are stored in a set
'''

import nltk, re
from nltk.corpus import stopwords
from bs4 import BeautifulSoup

def extract_message(url):
	markup = open(url)
	soup = BeautifulSoup(markup, "html.parser")
	for script in soup(["script", "style"]):
		script.extract()
	text = soup.get_text()
	text_clean = re.sub(r"\n", " ", text)
	text_clean = text_clean.replace(u'\xa0', u' ')
	a = text_clean.find('From:')
	tokens = word_tokenize((text_clean[a:]))
	tokens_set = set(tokens)
	stopwords = nltk.corpus.stopwords.words('english')
	content = [w for w in tokens_set if w.lower() not in stopwords]
	return content
