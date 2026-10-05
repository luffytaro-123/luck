def p1() -> str:
    """Web Scraping"""
    return """
import requests
from bs4 import BeautifulSoup
import pandas as pd

# Scrape quotes
url = 'https://quotes.toscrape.com/'
soup = BeautifulSoup(requests.get(url).text, 'html.parser')

quotes = [q.text for q in soup.find_all('span', class_='text')]
authors = [a.text for a in soup.find_all('small', class_='author')]

# Save to CSV
df = pd.DataFrame({'Quote': quotes, 'Author': authors})
df.to_csv('scraped_quotes.csv', index=False)
display(df.head())
"""


def p2() -> str:
    """Sentiment Analysis"""
    return """
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

sentences = [
    "I absolutely love learning Natural Language Processing!",
    "The weather today is extremely gloomy and depressing.",
    "We had a normal day at the office."
]

for s in sentences:
    score = sia.polarity_scores(s)['compound']
    sentiment = "Positive" if score >= 0.05 else "Negative" if score <= -0.05 else "Neutral"
    print(f'"{s}" -> {sentiment} ({score})')
"""


def p3() -> str:
    """Text Preprocessing"""
    return """
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, gutenberg
from nltk.stem import PorterStemmer, WordNetLemmatizer
import demoji
import contractions

nltk.download(['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4', 'gutenberg'], quiet=True)
text = "Hello there! 👋 This is a test sentence. I love NLP! ❤️ It's so amazing. 😊 Don't you think so?"

print("Original Text:", text)
print("-" * 30)

# 1. Sentence Tokenization
sentences = sent_tokenize(text)
print("1. Sentence Tokenization:", sentences)
print("-" * 30)

# 2. Contraction Expansion
expanded_text = contractions.fix(text)
print("2. Contraction Expanded:", expanded_text)
print("-" * 30)

# 3. Emoji Removal
demoji_removed_text = demoji.replace(expanded_text, '')
print("3. Emoji Removed Text:", demoji_removed_text)
print("-" * 30)

# For subsequent steps, we'll use the text after contraction expansion and emoji removal
processed_text = demoji_removed_text

# 4. Lowercasing
lowered = processed_text.lower()
print("4. Lowercased:", lowered)
print("-" * 30)

# 5. Word Tokenization
tokens = [w for w in word_tokenize(lowered) if w.isalnum()]
print("5. Word Tokenization (alphanumeric only):", tokens)
print("-" * 30)

# 6. Stopwords Removal
stop_words = set(stopwords.words('english'))
filtered = [w for w in tokens if w not in stop_words]
print("6. Stop Words Removed:", filtered)
print("-" * 30)

# 7. Stemming & Lemmatization
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
print("7. Stemmed:", [stemmer.stem(w) for w in filtered])
print("8. Lemmatized:", [lemmatizer.lemmatize(w) for w in filtered])
print("-" * 30)

# 9. Gutenberg Corpus Demonstration
print("9. Gutenberg Corpus Demonstration:")
print("Available Gutenberg files (first 5):", gutenberg.fileids()[:5])

hamlet_sentences = gutenberg.sents('shakespeare-hamlet.txt')
print("First 3 sentences from Hamlet:")
for sent in hamlet_sentences[:3]:
    print(f"- {' '.join(sent)}")
"""


def p4() -> str:
    """Parser in NLP"""
    return """
import nltk
import spacy
from spacy import displacy

# 1. NLTK CFG Parsing
grammar = nltk.CFG.fromstring(""  -------- # TRIPLE QUOTES
  S -> NP VP
  VP -> V NP | V NP PP
  PP -> P NP
  V -> "saw"
  NP -> "Mary" | Det N | Det N PP
  Det -> "a" | "the"
  N -> "dog" | "park"
  P -> "in"
"") ------------- # TRIPLE QUOTES

sentence = "Mary saw a dog in the park".split()
parser = nltk.RecursiveDescentParser(grammar)
print("--- NLTK CFG Parser Trees ---")
for tree in parser.parse(sentence):
    tree.pretty_print()

# 2. SpaCy Dependency Parser Graph
print("\n--- SpaCy Dependency Parse Graph ---")
nlp = spacy.load("en_core_web_sm")
doc = nlp("Mary saw a dog in the park")
displacy.render(doc)
"""


def p5() -> str:
    """Feature Extraction Techniques"""
    return """
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd

corpus = [
    "Natural Language Processing is amazing.",
    "Feature extraction is a crucial step in NLP."
]

# Count Vectorizer
cv = CountVectorizer()
cv_df = pd.DataFrame(cv.fit_transform(corpus).toarray(), columns=cv.get_feature_names_out())
display("CountVectorizer:", cv_df)

# TF-IDF
tfidf = TfidfVectorizer()
tfidf_df = pd.DataFrame(tfidf.fit_transform(corpus).toarray(), columns=tfidf.get_feature_names_out())
display("TF-IDF:", tfidf_df)
"""


def p6() -> str:
    """One Hot Encoding"""
    return """
import numpy as np
import pandas as pd

words = "NLP models process text inputs".split()
unique = sorted(list(set(words)))

# Map each word to a one-hot vector
ohe_matrix = np.eye(len(unique))
df_ohe = pd.DataFrame(ohe_matrix, index=unique, columns=unique)
display(df_ohe.loc[words])
"""


def p7() -> str:
    """Bag-of-Words (BOW)."""
    return """
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "the quick brown fox",
    "jumped over the lazy dog"
]

# 1. BoW using Library (scikit-learn)
print("--- Bag of Words (using CountVectorizer Library) ---")
vectorizer = CountVectorizer()
bow_lib = vectorizer.fit_transform(documents).toarray()
df_lib = pd.DataFrame(bow_lib, columns=vectorizer.get_feature_names_out())
display(df_lib)

# 2. BoW from Scratch Logic
print("\n--- Bag of Words (from Scratch Custom Logic) ---")
vocab = sorted(list(set(" ".join(documents).split())))
bow_scratch = [[doc.split().count(word) for word in vocab] for doc in documents]
df_scratch = pd.DataFrame(bow_scratch, columns=vocab)
display(df_scratch)
"""


def p8() -> str:
    """N-Grams"""
    return """
def get_ngrams(text, n):
    words = text.split()
    return [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]

sample = "Natural Language Processing is a subset of AI"
print("Bigrams (N=2):", get_ngrams(sample, 2))
print("Trigrams (N=3):", get_ngrams(sample, 3))
print("Quadgrams (N=4):", get_ngrams(sample, 4))
"""


def p9() -> str:
    """Term Frequency-Inverse Document Frequency (TF-IDF)"""
    return """
import math
import pandas as pd

docs = ["nlp is great", "computers understand nlp", "great computers"]
unique_words = set(" ".join(docs).split())

# TF-IDF calculation
tfidf_data = []
for doc in docs:
    words = doc.split()
    scores = {}
    for word in unique_words:
        tf = words.count(word) / len(words)
        df = sum(1 for d in docs if word in d.split())
        idf = math.log(len(docs) / df)
        scores[word] = tf * idf
    tfidf_data.append(scores)

display(pd.DataFrame(tfidf_data))
"""


def p10() -> str:
    """Word Embedding in Natural Language Processing"""
    return """
from gensim.models import Word2Vec

data = [["natural", "language", "processing", "is", "fun"], ["nlp", "is", "fun"]]
model = Word2Vec(sentences=data, vector_size=10, window=2, min_count=1, workers=1)

print("Vector for 'nlp':", model.wv['nlp'][:5])
print("Most similar to 'nlp':", model.wv.most_similar('nlp', topn=2))
"""
