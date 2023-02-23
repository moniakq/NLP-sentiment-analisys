import re
import nltk
import spacy
import unicodedata
from unidecode import unidecode

from bs4 import BeautifulSoup
from contractions import CONTRACTION_MAP
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
nlp = spacy.load('en_core_web_sm')
porter = PorterStemmer()
tknzr = TweetTokenizer()


def remove_html_tags(text):
    text = BeautifulSoup(text, 'html.parser')
    text = text.get_text()
    return text


def stem_text(text):
    text2=[]
    for w in tokenizer.tokenize(text):
        text2.append(porter.stem(w))
    text = ' '.join(text2)
    return text


def lemmatize_text(text):
    doc = nlp(text)
    text=" ".join([token.lemma_ for token in doc])
    return text


def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    text2=[]
    for w in tknzr.tokenize(text):
        if w in CONTRACTION_MAP.keys():
            text2.append(CONTRACTION_MAP[w])
        else:
            text2.append(w)
    text = ' '.join(text2)
    return text


def remove_accented_chars(text):
    text = unidecode(text)
    return text


def remove_special_chars(text, remove_digits=False):
    text2=[]
    text = re.sub('#', "", text)
    if remove_digits == True:
        text = re.sub('[^A-Za-z ]+', "", text)
    else:
        text = re.sub('[^A-Za-z0-9 ]+', "", text)
    return text

# def remove_special_chars(text, remove_digits=False):
#     text2=[]
#     text = re.sub('#', "", text)
#     for w in tokenizer.tokenize(text):
#         if remove_digits == True:
#             w = re.sub('[^A-Za-z]+', "", w)
#         else:
#             w = re.sub('[^A-Za-z0-9]+', "", w)
#         if w: 
#             text2.append(w)
#     text = ' '.join(text2)
#     return text


def remove_stopwords(text, is_lower_case=False, stopwords=stopword_list):
    text2=[]
    for w in tokenizer.tokenize(text):
        if is_lower_case==True:
            if w not in stopword_list:
                text2.append(w)
        else:
            if w.lower() not in stopword_list:
                text2.append(w)
    text = ' '.join(text2)
    return text


def remove_extra_new_lines(text):
    text2=[]
    for w in tokenizer.tokenize(text):
        text2.append(w)
    text = ' '.join(text2)
    return text


def remove_extra_whitespace(text):
    text2=[]
    for w in tokenizer.tokenize(text):
        text2.append(w)
    text = ' '.join(text2)
    return text
    

def normalize_corpus(
    corpus,
    html_stripping=True,
    contraction_expansion=True,
    accented_char_removal=True,
    text_lower_case=True,
    text_stemming=False,
    text_lemmatization=False,
    special_char_removal=True,
    remove_digits=True,
    stopword_removal=True,
    stopwords=stopword_list
):
    
    normalized_corpus = []

    # Normalize each doc in the corpus
    for doc in corpus:
        # Remove HTML
        if html_stripping:
            doc = remove_html_tags(doc)
            
        # Remove extra newlines
        doc = remove_extra_new_lines(doc)
        
        # Remove accented chars
        if accented_char_removal:
            doc = remove_accented_chars(doc)
            
        # Expand contractions    
        if contraction_expansion:
            doc = expand_contractions(doc)
            
        # Lemmatize text
        if text_lemmatization:
            doc = lemmatize_text(doc)
            
        # Stemming text
        if text_stemming and not text_lemmatization:
            doc = stem_text(doc)
            
        # Remove special chars and\or digits    
        if special_char_removal:
            doc = remove_special_chars(
                doc,
                remove_digits=remove_digits
            )  

        # Remove extra whitespace
        doc = remove_extra_whitespace(doc)

         # Lowercase the text    
        if text_lower_case:
            doc = doc.lower()

        # Remove stopwords
        if stopword_removal:
            doc = remove_stopwords(
                doc,
                is_lower_case=text_lower_case,
                stopwords=stopwords
            )

        # Remove extra whitespace
        doc = remove_extra_whitespace(doc)
        doc = doc.strip()
            
        normalized_corpus.append(doc)
        
    return normalized_corpus
