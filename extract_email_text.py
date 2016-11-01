" " " This method extracts an emails
contents, while eliminating stopwords, newline 
characters, /xa0 characters, then tokenizes
the words, which are stored in a set
" " "

def extract_message(url):
	markup = open(url)
	soup = BeautifulSoup(markup, "html.parser")
	for script in soup(["script", "style"]):
		script.extract()
	text = soup.get_text()
	text_clean = re.sub(r"\n", " ", text)
	text_clean = text_clean.replace(u'\xa0', u' ')
	paras = text_clean.split("\n\n")
	formatted = "\n\n".join(textwrap.fill(p) for p in paras)
	a = formatted.find('From:')
	tokens = word_tokenize((formatted[a:]))
	tokens_set = set(tokens)
	stopwords = nltk.corpus.stopwords.words('english')
	content = [w for w in tokens_set if w.lower() not in stopwords]
	return content