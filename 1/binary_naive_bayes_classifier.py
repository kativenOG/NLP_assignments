# NEW FILE !
# Versione che fa un tap training per ogni documento 
# e poi passa i dati già modificati al feature estractor 

#************************ IMPORTS ************************#
import string
import nltk
import asyncio
import random
import math 
import collections
#********************************************************#
from tqdm import tqdm 
nltk.download("europarl_raw")
nltk.download("stopwords")
#********************************************************#
from nltk.corpus import europarl_raw
from nltk.corpus import stopwords
from nltk.metrics import ConfusionMatrix
#********************************************************#
from nltk.metrics.scores import (precision, recall)
from nltk.tokenize import RegexpTokenizer, sent_tokenize#,sent_tokenize,PunktSentenceTokenizer is tokenizer with a unsupervised machine learning program beneath it 
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer 
stemmer = PorterStemmer()  
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
#********************************************************#

def tap_training(text,fdist,chunker = None):

    try:
        # BOW: bag of words, for calculating the top 2000 used words in the already greatly reduced Vocabulary
        transformed_document = [] 
        for data in tqdm(text): 
            # Tokenization is sentences 
            sents = sent_tokenize(data)
            for sent in sents:

                # Tokenizing words 
                tokenizer = RegexpTokenizer(r"\w+") 
                words = tokenizer.tokenize(sent) # we tokenize to words the alredy punktokenized sentences !
                # Stemming:
                stemmed= [stemmer.stem(word) for word in words]
                # Standard Lemmatiziation
                lemmatized = [lemmatizer.lemmatize(word) for word in stemmed]
                # Counting the words that we already lemmatized for the BOW counter 
                for word in lemmatized:
                    transformed_document.append(word) # returning the document to 
                    fdist[word.lower()] += 1 # adding to the world count for BOW 

        print("\n")
        return list(fdist)[:2000], transformed_document

    except Exception as e:
        print(str(e))
        return "err"

def feature_estractor(document,top_words,chunker = None):

    # Ususal TAP pipeline (tokenization,stop words elimination, stemming and lemmatization )
    processed_document = set()
    # Tokenization in sentences
    sents = sent_tokenize(document)
    for sent in sents:

        # Tokenization in words 
        tokenizer = RegexpTokenizer(r"\w+") 
        words = tokenizer.tokenize(sent) # we tokenize to words the alredy punktokenized sentences !
        # Stemming:
        stemmed= [stemmer.stem(word) for word in words]
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
        fr_ids = [fileid for fileid in europarl_raw.french.fileids()]
        dutch_ids = [fileid for fileid in europarl_raw.dutch.fileids()]
        # fr_ids = fr_ids[math.floor(len(fr_ids)/3):]
        # en_ids = en_ids[math.floor(len(en_ids)/3):]
        # dutch_ids = dutch_ids[math.floor(len(dutch_ids)/3):]

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

        print("Starting BAT process for europarl_raw, to estract top 2000 from reduced V:")
        raw_data = [] 
        for [text,ids] in documents:
            raw_data.append(text)

        fdist = FreqDist() 

        top_words = tap_training(raw_data,fdist) # deve diventare tutte le parole sommate dei tue testi. 

        if top_words == "err": 
            print("ERROR!")
        else:
            print("Creating Feature Sets: ")
            featuresets = [(feature_estractor(d,top_words), c) for (d,c) in tqdm(documents)]
            half_len = math.floor(len(featuresets) * 0.5 )
            train_set, test_set = featuresets[:half_len], featuresets[half_len:]


            print("Training the Naive Bayes Classifier: ")
            classifier = nltk.NaiveBayesClassifier.train(train_set) 
    
            print("Testing and Metrics: ")
            refsets =  collections.defaultdict(set)
            testsets = collections.defaultdict(set)
            labels = []
            tests = []
            for i,(feats,label) in tqdm(enumerate(test_set)):
                refsets[label].add(i)
                result = classifier.classify(feats)
                testsets[result].add(i)
                labels.append(label)
                tests.append(result)
            cm = ConfusionMatrix(labels, tests)
            
            print("Accuracy:",nltk.classify.accuracy(classifier, test_set))
            print( 'Precision:', precision(refsets['English'], testsets['English']) )
            print( 'Recall:', recall(refsets['English'], testsets['English']) )
            print(cm)
            classifier.show_most_informative_features(35)

            print("Extra test: ")
            string1 = "Hello my name is brian, i want to test the result of test. Let's hope for a great result !"
            string2 = "Je m'appelle Angélica Summer, j'ai 12 ans et je suis canadienne. Il y a 5 ans, ma famille et moi avons déménagé dans le sud de la France. Mon père, Frank Summer, est mécanicien ; il adore les voitures anciennes et collectionne les voitures miniatures."
            string3 = "Pierre est un jeune garçon de 14 ans. Il vit à Paris avec ses parents et sa petite sœur Julie, âgée de 8 ans. Toute la petite famille habite dans un grand appartement au 3ème étage d'un immeuble situé près de la Tour Eiffel. Ainsi, Pierre a le privilège dadmirer chaque jour lun des monuments les plus visités au monde !Pour se rendre au collège, Pierre prend le métro à la station Ecole Militaire et sort au collège Claude Debussy. Le trajet ne dure que 20 minutes ! Le week-end, Pierre aime passer du temps en famille. Tous les quatre en profitent pour visiter les musées parisiens, aller au cinéma, faire du shopping, ou se balader dans l'un des nombreux parcs de la capitale. Les vacances sont les moments préférés de Pierre car il a pour habitude d'aller chez ses grands-parents qui vivent dans une ferme de la campagne normande. Toute la petite famille apprécie alors l'air pur, la nature et le calme pendant quelques semaines avant de rentrer à Paris."
            engfeatures = feature_estractor(string1,top_words)
            non_engfeatures = feature_estractor(string2,top_words)
            non_engfeatures1 = feature_estractor(string3,top_words)
            print("Text 1:",string1)
            print("Classification:", classifier.classify(engfeatures),"\n")
            print("Text 2:",string2)
            print("Classification:", classifier.classify(non_engfeatures),"\n")
            print("Text 3:",string3)
            print("Classification:", classifier.classify(non_engfeatures1),"\n")
    except Exception as e:
        print(str(e))
        return "err"


asyncio.run(main())

# Stop Words removal 
# stopped = []
# stop_words = set(stopwords.words("english"))  # a set containing all the English stopWords
# stop_words.add(word for word in stopwords.words("french"))
# stop_words.add(word for word in stopwords.words("dutch"))
# for word in words:
#     if word.casefold() not in stop_words:
#         stopped.append(word)

