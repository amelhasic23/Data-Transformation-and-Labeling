import pandas as pd
import nltk
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import FastText

# Preuzmite potrebne resurse
nltk.download('punkt')
nltk.download('stopwords')

# Učitajte podatke iz CSV datoteke
data = pd.read_csv('data.csv')

# Prikaz svih naziva kolona
print("Nazivi kolona:", data.columns)

# Inicijalizujte spaCy model
nlp = spacy.load('en_core_web_sm')

# Tokenizacija
def tokenize(text):
    return word_tokenize(text)

# Koristimo naziv kolone sa tekstom
data['tokens'] = data['text'].apply(tokenize)

# Uklanjanje zaustavnih reči
stop_words = set(stopwords.words('english'))

def remove_stopwords(tokens):
    return [word for word in tokens if word.lower() not in stop_words]

data['tokens_no_stopwords'] = data['tokens'].apply(remove_stopwords)

# Lemmatizacija
def lemmatize(tokens):
    doc = nlp(' '.join(tokens))
    return [token.lemma_ for token in doc]

data['lemmatized_tokens'] = data['tokens_no_stopwords'].apply(lemmatize)

# Označavanje vrste reči (POS tagging)
def pos_tagging(tokens):
    doc = nlp(' '.join(tokens))
    return [(token.text, token.pos_) for token in doc]

data['pos_tags'] = data['tokens_no_stopwords'].apply(pos_tagging)

# Obuka FastText modela
# Ako već imate obučeni model, učitajte ga pomoću gensim.models.FastText.load('path_to_model')
model = FastText(sentences=data['lemmatized_tokens'], vector_size=100, window=5, min_count=1, workers=4)

# Pronalazak sličnih reči
def get_similar_words(word):
    return model.wv.most_similar(word)

# Pronalazak najmanje sličnih reči
def get_least_similar_words(word):
    similar_words = model.wv.similar_by_word(word)
    return sorted(similar_words, key=lambda x: x[1])[:2]

# Primer korišćenja za reč 'data'
word_to_check = 'data'
similar_words = get_similar_words(word_to_check)
least_similar_words = get_least_similar_words(word_to_check)

print("Najviše slične reči za '{}':".format(word_to_check))
for word, similarity in similar_words:
    print(f"{word}: {similarity}")

print("\nNajmanje slične reči za '{}':".format(word_to_check))
for word, similarity in least_similar_words:
    print(f"{word}: {similarity}")

# Čuvanje obrađenih podataka u novi CSV
data.to_csv('processed_data.csv', index=False)
