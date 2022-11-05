#************************ IMPORTS ************************#
import nltk
from nltk.tokenize import  PunktSentenceTokenizer# PunktSentenceTokenizer is tokenizer with a unsupervised machine learning program beneath it 
nltk.download("state_union")
nltk.download("stopwords")
nltk.download("tagsets")
# nltk.download("draw")
from nltk.corpus import stopwords

from nltk.stem import PorterStemmer, SnowballStemmer
stemmer = PorterStemmer() # Our stemmer 
stemmer = SnowballStemmer("english") # A better stemmer 

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
#********************************************************#

# CORPUS DATA 
from nltk.corpus import state_union
train_text = state_union.raw("2005-GWBush.txt")
testing_text = state_union.raw("2006-GWBush.txt")

worf_quote = "Sir, I protest. I am not a merry man!"

# TAP: Text Analysis Pipeline 
if __name__ == "__main__":
    # nltk.help.upenn_tagset()
    try:
        # Tokenization 
        custom_tokenizer = PunktSentenceTokenizer(train_text)
        tokenized = custom_tokenizer.tokenize(testing_text)

        # Chunking: 
        # The process of natural language processing used to identify parts of speech and short phrases present in a given sentence.
        for token in tokenized: 
            words = nltk.word_tokenize(token)
            tagged_POS= nltk.pos_tag(words) # First we need to tag parts of speech ( TAG POS )
            # r stands for Regex, RB is an adverb, VB verb, NNP proper noun 
            chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""  # RB followed by anything (includes all adverbs forms) then followd by a form of a verb , then a proper noun, then a noun 
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged_POS)
            print(chunked)
            chunked.draw()

        # Stop Words Elimination
        stopped = []
        stop_words = set(stopwords.words("english"))  # a set containing all the English stopWords
        for word in tokenized:
            if word.casefold() not in stop_words:
                stopped.append(word)
        # real_worf = [w for w in worf_quote if not w in stop_words]

        # Stemming:
        stemmed= [stemmer.stem(word) for word in stopped]
        # print("Stemmed data: ",stemmed)

        # Standard Lemmatizing 
        lemmatized = [lemmatizer.lemmatize(word) for word in stemmed]
        # print("Lemmatized data: ",lemmatized)

    except Exception as e:
        print(str(e))

