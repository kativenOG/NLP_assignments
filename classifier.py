import nltk
# import re 
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download("stopwords")
nltk.download("tagsets")
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))  # a set containing all the English stopWords

from nltk.stem import PorterStemmer, SnowballStemmer
stemmer = PorterStemmer() # Our stemmer 
stemmer = SnowballStemmer("english") # A better stemmer 

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Example: tokenizing by word and removing all the stopWords in example 
if __name__ == "__main__":
    worf_quote = "Sir, I protest. I am not a merry man!"
    real_worf = []
    worf_tokenized = word_tokenize(worf_quote)
    for word in worf_tokenized:
        # Regex 
        #if re.search("Not|not",word.casefold()):
        #elif re.search("I",word.casefold()):
        if word.casefold() not in stop_words:
            real_worf.append(word)
    # real_worf = [w for w in worf_quote if not w in stop_words]

    print("Tokenized worf: ",real_worf)
    stemmed_worf = [stemmer.stem(word) for word in real_worf]
    print("Stemmed worf: ",stemmed_worf)

    # Standard Lemmatizing 
    lemmatized_worf = [lemmatizer.lemmatize(word) for word in real_worf]
    print("Stemmed worf: ",stemmed_worf)

    # Lemmatizing after Tagging Parts of Speetch   
    # nltk.help.upenn_tagset()
    # worf_tagged_parted = nltk.pos_tag(real_worf)
    # lemmatized_worf = [lemmatizer.lemmatize(word[0],word[1]) for word in worf_tagged_parted]
    



