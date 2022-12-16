#!/usr/bin/env python
# coding: utf-8

# # Text Processing
# 
# ## Load Data
# 
# All data from the preceeding web scrape is loaded. 
# 
# `review_text` is converted to lower case.

# In[2]:


import pandas as pd
# load data
df = pd.read_csv('data/processed_review_data.csv')
df['review_text'] = df['review_text'].str.lower()


# ## Clean Text
# 
# Several functions to process review text are developed and applied. These include:
# 1. Contraction expansion
# 2. String formatting
# 3. Duplicate character and word removal
# 4. Spelling correction
# 5. Lemmatisation and tokenisation
# 
# ### Define Functions
# #### Contraction Expansion
# 
# A dictionary of contractions and associated expnasions are defined.
# 
# The Python [Regular Expression](https://docs.python.org/3/library/re.html) library is used to identify and replace all contractions with their expanded form.
# 

# In[3]:


import re
# define contractions dictionary
cList = {
    # A.
    "ain't": "am not","aren't": "are not",
    # C.
    "can't": "cannot","can't've": "cannot have","'cause": "because","could've": "could have","couldn't": "could not",
    "couldnt": "could not","couldn't've": "could not have",
    # D.
    "didn't": "did not","doesn't": "does not","don't": "do not",
    # H.
    "hadn't": "had not","hadn't've": "had not have","hasn't": "has not","haven't": "have not","he'd": "he would",
    "he'd've": "he would have","he'll": "he will","he'll've": "he will have","he's": "he is","how'd": "how did",
    "how'd'y": "how do you","how'll": "how will","how's": "how is",
    # I.
    "i'd": "i would","i'd've": "i would have","i'll": "i will","i'll've": "i will have","i'm": "i am","i've": "i have",
    "isn't": "is not","it'd": "it had","it'd've": "it would have","it'll": "it will","itll": "it will",
    "it'll've": "it will have","it's": "it is",
    # L.
    "let's": "let us",
    # M.
    "ma'am": "madam","mayn't": "may not","might've": "might have","mightn't": "might not","mightn't've": "might not have",
    "must've": "must have","mustn't": "must not","mustn't've": "must not have",
    # N.
    "needn't": "need not","needn't've": "need not have",
    # O.
    "o'clock": "of the clock","oughtn't": "ought not","oughtn't've": "ought not have",
    # S.
    "shan't": "shall not","sha'n't": "shall not","shan't've": "shall not have","she'd": "she would",
    "she'd've": "she would have","she'll": "she will","she'll've": "she will have","she's": "she is","should've": "should have",
    "shouldn't": "should not","shouldn't've": "should not have","so've": "so have","so's": "so is",
    # T.
    "that'd": "that would","that'd've": "that would have","that's": "that is","there'd": "there had",
    "there'd've": "there would have","there's": "there is","they'd": "they would","they'd've": "they would have",
    "they'll": "they will","they'll've": "they will have","they're": "they are","they've": "they have","to've": "to have",
    # V.
    "vr" : "virtual reality",
    # W.
    "wasn't": "was not","we'd": "we had","we'd've": "we would have","we'll": "we will","we'll've": "we will have",
    "we're": "we are","we've": "we have","weren't": "were not","what'll": "what will","what'll've": "what will have",
    "what're": "what are","what's": "what is","what've": "what have","when's": "when is","when've": "when have",
    "where'd": "where did","where's": "where is","where've": "where have","who'll": "who will","who'll've": "who will have",
    "who's": "who is","who've": "who have","why's": "why is","why've": "why have","will've": "will have","won't": "will not",
    "won't've": "will not have","would've": "would have","wouldn't": "would not","wouldn't've": "would not have",
    # Y.
    "y'all": "you all","y'alls": "you alls","y'all'd": "you all would","y'all'd've": "you all would have",
    "y'all're": "you all are","y'all've": "you all have","you'd": "you had","you'd've": "you would have",
    "you'll": "you you will","you'll've": "you you will have","you're": "you are","you've": "you have"
}
c_re = re.compile('(%s)' % '|'.join(cList.keys()))

# define expansion function
def expandContractions(text, cList=cList):
    def replace(match):
        return cList[match.group(0)]
    return c_re.sub(replace, text)


# #### String Formatting
# The following string formatting is performed:
# * All sentences are transformed to end in a '.'
# * All floating point numbers are removed.
# * Non alphanumeric characters are dropped.
# * Repeated fullstops and additional white space are dropped.

# In[7]:


# define str format function
def clean(text):
    text = re.sub(r'[?!:]', '.', text) # make all sentence ends with '.'
    text = re.sub('\d*\.\d+','', text) # remove all floats
    text = re.sub("[^a-zA-Z0-9. ]", '', text) # remove all not listed chars 
    text = re.sub('\.\.+', '. ',text) # remove repeat fullstops
    text = re.sub(' +',' ', text) # remove extra whitespace
    return text


# #### Duplicate Removal
# 
# Consecutive duplicate words are dropped and consecutive repeat characters are limited to a maximum of 2.

# In[9]:


# define duplicate removal function
from itertools import groupby

def consec_dup(text):
    text = " ".join([x[0] for x in groupby(text.split(" "))]) # remove repeat consecutive words
    text = re.sub(r'(.)\1+', r'\1\1',text) # replace >2 consecutive duplicate characters
    return text


# #### Spelling Correction
# 
# The `SymspellPy` library {cite}`symspell_2022` is used to perform spelling correction.

# In[12]:


import pkg_resources
from symspellpy import SymSpell, Verbosity

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt"
)

sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

def spell(text):
    sentences = text.split('.')
    corrected = " ".join([sym_spell.lookup_compound(x, max_edit_distance=2, ignore_non_words=True,ignore_term_with_digits=True)[0].term+'.' for x in sentences])
    return corrected


# #### Lemmatisation
# 
# The `spaCy` library {cite}`spacy_2020` is utilised to lemmatise the review text and transform all words to their base root form.

# In[14]:


import spacy
nlp = spacy.load("en_core_web_sm")

def lemma(text):
    doc = nlp(text)
    text = [token.lemma_ for token in doc]
    text = " ".join(text)
    return text


# ### Apply Functions
# 
# All functions are applied and the output saved to .csv for proceeding stages.

# In[24]:


for func in [expandContractions,clean,consec_dup,spell,lemma]:
    df.review_text = df.review_text.map(func,na_action='ignore')

df.dropna().to_csv('data/preTag_df1.csv',index=False)

df

