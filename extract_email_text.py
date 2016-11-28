import nltk, re, string
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from nltk import word_tokenize

#Punctuation list
punct = [".",";",":","!","?","/","\\",",","#","@","$","&",")","...",
"''","<",">","(","\"!","@","#","$","%","^","&","*"]

def extract_message(url):
	markup = open(url)
	soup = BeautifulSoup(markup, "html.parser")
	for script in soup(["script", "style"]):
		script.extract()
	text = soup.get_text()
	text = re.sub(r'[^\x00-\x7F]+','', text) #Eliminate Foreign Characters
	text = re.sub(r'\d+', '', text) #Eliminate Numbers
	a = text.find('From:')
	tokens = word_tokenize((text[a:]))
	tokens_set = set(tokens)
	stopwords = nltk.corpus.stopwords.words('english')
	content = [w for w in tokens_set if w.lower() not in stopwords]
	content = [p for p in content if p not in punct]
	return content