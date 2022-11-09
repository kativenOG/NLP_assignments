#************************ IMPORTS ************************#
import nltk
import asyncio
import random
import math 
import collections
#********************************************************#
nltk.download("europarl_raw")
nltk.download("stopwords")
#********************************************************#
from nltk.corpus import europarl_raw
from nltk.corpus import stopwords
#********************************************************#
from nltk.metrics.scores import (precision, recall)
from nltk.tokenize import RegexpTokenizer, word_tokenize#,sent_tokenize,PunktSentenceTokenizer is tokenizer with a unsupervised machine learning program beneath it 
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer 
stemmer = PorterStemmer()  
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
#********************************************************#


# TAP: Text Analysis Pipeline 
async def tap_training(text):

    try:
        # BOW: bag of words, for calculating the top 2000 used words in the already greatly reduced Vocabulary
        fdist = FreqDist()
        for data in text: 

            # Tokenization 
            tokenizer = RegexpTokenizer(r"[A-zÀ-ú]+") 
            words = tokenizer.tokenize(data) # we tokenize to words the alredy punktokenized sentences !
            # r"[A-zÀ-ú ]+": rimuove numeri, caratteri speciali e tiene solo le lettere e spazi, anche accentate!  
            # r"\w+": keeps only alphanumeric characters, removing all punctuation :^) 

            # Stop Words Elimination
            stopped = []
            stop_words = set(stopwords.words("english"))  # a set containing all the English stopWords
            stop_words.add(word for word in stopwords.words("french"))
            stop_words.add(word for word in stopwords.words("dutch"))
            for word in words:
                if word.casefold() not in stop_words:
                    stopped.append(word)

            # MISSING CHUNKING !!!

            # Stemming:
            stemmed= [stemmer.stem(word) for word in stopped]

            # Standard Lemmatizing 
            lemmatized = [lemmatizer.lemmatize(word) for word in stemmed]

            # Counting the words that we already lemmatized in the BOW counter 
            for word in lemmatized:
                fdist[word.lower()] += 1
            # fdist= FreqDist(w.lower() for w in lemmatized)

        print("Done with Tap, returning top 2000 words")
        return list(fdist)[:2000]
                    
    except Exception as e:
        print(str(e))
        return "err"



def feature_estractor(document,top_words):
    # document_words = set(document)
    # Ususal TAP pipeline (tokenization,stop words elimination, stemming and lemmatization )
    processed_document = set()
    # Tokenization 
    tokenizer = RegexpTokenizer(r"[A-zÀ-ú ]+") 
    words = tokenizer.tokenize(document) # we tokenize to words the alredy punktokenized sentences !
    # Stop Words 
    stopped = []
    stop_words = set(stopwords.words("english"))  # a set containing all the English stopWords
    stop_words.add(word for word in stopwords.words("french"))
    stop_words.add(word for word in stopwords.words("dutch"))
    for word in words:
        if word.casefold() not in stop_words:
            stopped.append(word)
    # MISSING CHUNKING  !!!
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
        # doing collage of Europarlamentar english and then Europarlamentar in french and dutch for non-english text
        en_ids = [fileid for fileid in europarl_raw.english.fileids()]
        en_ids = en_ids[math.floor(len(en_ids)/3):]
        fr_ids = [fileid for fileid in europarl_raw.french.fileids()]
        fr_ids = fr_ids[math.floor(len(fr_ids)/3):]
        dutch_ids = [fileid for fileid in europarl_raw.dutch.fileids()]
        dutch_ids = dutch_ids[math.floor(len(dutch_ids)/3):]

        # Loading ENGLISH corpora and respective label 
        documents= [(europarl_raw.english.raw(fileid), "English") # Maybe remove list() and use .words
            for fileid in en_ids]
        # Loading FRENCH corpora and respective label 
        for fileid in fr_ids:
            documents.append((europarl_raw.french.raw(fileid) , "NonEnglish"))
        # Loading DUTCH corpora and respective label 
        for fileid in dutch_ids:
            documents.append((europarl_raw.dutch.raw(fileid) , "NonEnglish"))
        random.shuffle(documents)
        print("DONE!")


        print("Starting BAT process for europarl_raw, to estract top 2000 from reduced V:")
        raw_data = [] 
        for [text,ids] in documents:
            raw_data.append(text)
            # print(ids)
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
            print("Accuracy:",nltk.classify.accuracy(classifier, test_set))

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

    except Exception as e:
        print(str(e))
        return "err"

asyncio.run(main())
