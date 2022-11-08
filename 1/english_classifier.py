#************************ IMPORTS ************************#
import nltk
import asyncio
import random
import math 
import collections
# from sklearn.feature_extraction.text import CountVectorizer 
# import pandas as pd  
#********************************************************#
nltk.download("europarl_raw")
nltk.download("stopwords")
# nltk.download("tagsets")
#********************************************************#
from nltk.corpus import europarl_raw
from nltk.corpus import stopwords
#********************************************************#
from nltk.metrics.scores import (precision, recall)
from nltk.tokenize import RegexpTokenizer, word_tokenize,sent_tokenize  #PunktSentenceTokenizer is tokenizer with a unsupervised machine learning program beneath it 
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer 
stemmer = PorterStemmer()  
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
#********************************************************#


# TAP: Text Analysis Pipeline 
async def tap_training(text):

    try:

        fdist = FreqDist()
        for token in text:
            # Tokenization 
            # tokenizer = RegexpTokenizer(r"\w+") # keeps only alphanumeric characters, removing all punctuation :^) 
            # words = tokenizer.tokenize(token) # we tokenize to words the alredy punktokenized sentences !

            # Stop Words Elimination
            stopped = []
            stop_words = set(stopwords.words("english"))  # a set containing all the English stopWords
            for word in token:
                if word.casefold() not in stop_words:
                    stopped.append(word)

            # Stemming:
            stemmed= [stemmer.stem(word) for word in stopped]
            # Standard Lemmatizing 
            lemmatized = [lemmatizer.lemmatize(word) for word in stemmed]

            for word in lemmatized:
                fdist[word.lower()] += 1

        print("Done with Tap, returning top 2000 words")
        return list(fdist)[:2000]
                    
    except Exception as e:
        print(str(e))
        return "err"



def feature_estractor(document,top_words):
    # document_words = set(document)
    # Ususal TAP pipeline (tokenization,stop words elimination, stemming and lemmatization )
    processed_document = set()
    for token in document:
        # Stop Words 
        stopped = []
        stop_words = set(stopwords.words("english"))  # a set containing all the English stopWords
        for word in token:
            if word.casefold() not in stop_words:
                stopped.append(word)

        # Stemming:
        stemmed= [stemmer.stem(word) for word in stopped]
        # Standard Lemmatizing 
        lemmatized = [lemmatizer.lemmatize(word) for word in stemmed]
        
        for word in lemmatized:
            processed_document.add(word)

    features = {}
    for word in top_words:
        features['contains({})'.format(word)] = (word in processed_document)
    return features


async def main():
    try: 
        print("Loading NLTK Data ")
        # CORPUS DATA 
        # for training the model is going to use the Europarlamentar english corpora
        # doing collage of Europarlamentar english and then Europarlamentar in french for non-english text
        eng_ids = [fileid for fileid in europarl_raw.english.fileids()]
        eng_ids = eng_ids[len(eng_ids)/3:]
        documents= [(list(europarl_raw.english.sents(fileid)), "English") # Maybe remove list() and use .words
            for fileid in europarl_raw.english.fileids()]
        for fileid in europarl_raw.french.fileids():
            documents.append((list(europarl_raw.french.sents(fileid)) , "NonEnglish"))
        random.shuffle(documents)
        print("DONE!")


        print("Starting BAT process for europarl_raw, to estract top 2000 from reduced V:")
        raw_data = [] 
        for [text,ids] in documents:
            for line in text:      
                raw_data.append(line)
        top_words = await tap_training(raw_data) # deve diventare tutte le parole sommate dei tue testi. 
        if top_words == "err": 
            print("ERROR!")
        else:
            print("DONE!")
            print("Creating Feature Sets: ")
            featuresets = [(feature_estractor(d,top_words), c) for (d,c) in documents]
            half_len = math.floor(len(featuresets) * 0.7 )
            train_set, test_set = featuresets[half_len:], featuresets[:half_len]
            print("DONE!")


            print("Training the Naive Bayes Classifier: ")
            classifier = nltk.NaiveBayesClassifier.train(train_set) 
            print("DONE!")

    
            print("Testing and Metrics: ")
            print(nltk.classify.accuracy(classifier, test_set))

            refsets =  collections.defaultdict(set)
            testsets = collections.defaultdict(set)
            i = 0  
            for [feats,label] in test_set:
                refsets[label].add(i)
                result = classifier.classify(feats)
                testsets[result].add(i)
                i+=1

            print( 'Precision:', precision(refsets['English'], testsets['English']) )
            print( 'Recall:', recall(refsets['English'], testsets['English']) )
            classifier.show_most_informative_features(35)

            while True: 
                print("Insert sentences, type quit when you done:")
                array = []
                sentence= ""
                while sentence!= "quit":
                    sentence = input()
                    array.append(word_tokenize(sentence))
                print(array) 
                lan = input("Language")
                pair = (array,lan) 
                print("Classificaton: ",classifier.classify(pair))
                print("Type yes to Exit")
                if input() == "yes":
                    break

    except Exception as e:
        print(str(e))
        return "err"

asyncio.run(main())

# Chunking: 
# Fai con catena di Markow 
# Contains parts of speech tagging 
# The process of natural language processing used to identify parts of speech and short phrases present in a given sentence. DIAMO UNA STRUTTURA GENERALE ALLA FRASE CHE VOGLIAMO 
# chunked = []
# for token in text: 
#     words = word_tokenize(token) # we tokenize to words !
#     nltk.help.upenn_tagset()
#     tagged_POS= nltk.pos_tag(words) # First we need to tag parts of speech ( TAG POS )
#     # r stands for Regex, RB is an adverb, VB verb, NNP proper noun 
#     grammar = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""  # RB followed by anything (includes all adverbs forms) then followd by a form of a verb , then a proper noun, then a noun 
#     chunkParser = nltk.RegexpParser(grammar)
#     chunked.append(chunkParser.parse(tagged_POS))
