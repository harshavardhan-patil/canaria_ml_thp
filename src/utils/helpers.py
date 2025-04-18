import ast
import json
import re
import pandas as pd
from bs4 import BeautifulSoup
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')

def parse_array_field(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            return json.loads(x)
        except:
            try:
                return ast.literal_eval(x)  # handles "['a', 'b']"
            except:
                return re.findall(r'[^,{}]+', x)  # handles '{a,b}' PostgreSQL format
    return []

def clean_html(html_text):
    """Remove HTML tags from text."""
    if pd.isna(html_text):
        return ""
    soup = BeautifulSoup(html_text, 'html.parser')
    return soup.get_text(separator=' ', strip=True)

def normalize_text(text):
    """Standardize casing and remove special characters."""
    if pd.isna(text):
        return ""
    # Keep only alphanumeric, whitespace and newlines
    text = re.sub(r'[^\w\s\n]', ' ', text)
    # Convert multiple spaces to a single space
    text = re.sub(r'\s+', ' ', text)
    # Lowercase everything
    text = text.lower().strip()
    return text

def lemmatize_text(text):
    """Convert words to their root forms."""
    if pd.isna(text) or text == "":
        return ""
    
    lemmatizer = WordNetLemmatizer()
    
    # Tokenize text
    word_tokens = word_tokenize(text)
    
    # Lemmatize and filter out stopwords
    lemmatized_words = [lemmatizer.lemmatize(word) for word in word_tokens]
    
    return ' '.join(lemmatized_words)

def standardize_companyname(text):
    """Standardize pre-lemmatized job titles"""
    if pd.isna(text) or text == "":
        return text
    # Replace common suffixes
    suffixes = [
        ' inc',
        ' inc.',
        ' corporation',
        ' corp',
        ' corp.',
        ' limited',
        ' ltd',
        ' ltd.',
        ' llc',
        ' l.l.c.',
        ' company',
        ' co',
        ' co.'
    ]

    for suffix in suffixes:
        text = text.replace(suffix, " ")
    
    return text
    
def standardize_title(text):
    # To handle cases like Data Science Intern - Graduate vs Graduate Data Science Intern
    sorted_title = ' '.join(sorted(text.replace(',', '').split()))
    return sorted_title