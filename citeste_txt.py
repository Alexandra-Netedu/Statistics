#citeste_txt.py
#with open("corpus.txt", "r", encoding="utf-8") as f:
 #   continut = f.read()

#print("Conținutul fișierului este:\n")
#print(continut)

#import re

#with open("corpus.txt", "r", encoding="utf-8") as f:
 #for linie in f:
   #  linie = linie.strip()
        # caută: număr + punct, apoi text între ghilimele speciale + virgulă + clasa
#match = re.match(r'^\d+\.\s*[“"](.*?)[”"]\s*,\s*(\d+)$', linie)
#if not match:
 #           print(f"Format invalid: {linie}")
#else:
           # text = match.group(1)
          #  clasa = match.group(2)
           # print(f"Text: {text} | Clasa: {clasa}")
#import re
import nltk
#from nltk.tokenize import word_tokenize
#from nltk.classify import NaiveBayesClassifier
#from nltk.classify.util import accuracy

# Asigură-te că ai descărcat datele necesare
#nltk.download('punkt')

# 1. Citește și extrage text + clasa
#texte = []
#clase = []

#with open("corpus.txt", "r", encoding="utf-8") as f:
#    for linie in f:
#        linie = linie.strip()
#        match = re.match(r'^\s*[“"]?(.*?)[”"]?\s*,\s*(\d+)$', linie)
#        if match:
#            text = match.group(1)
#            clasa = match.group(2)
#            texte.append(text)
#            clase.append(clasa)

# 2. Transformare în caracteristici (feature extraction)
#def extrage_cuvinte(text):
#    tokens = word_tokenize(text.lower())
#    return {cuvant: True for cuvant in tokens}

# 3. Crearea setului de date
#date_features = [(extrage_cuvinte(text), label) for text, label in zip(texte, clase)]

# 4. Împărțire manuală în set de antrenare și test
#procent_train = 0.8
#dim_train = int(len(date_features) * procent_train)
#train_set = date_features[:dim_train]
#test_set = date_features[dim_train:]

# 5. Antrenare Naive Bayes
#clasificator = NaiveBayesClassifier.train(train_set)

# 6. Evaluare a acurateței
#print("Acuratețe:", accuracy(clasificator, test_set))



import nltk
#from nltk.corpus import stopwords
#from nltk.tokenize import word_tokenize
#from nltk.stem import WordNetLemmatizer
#import string
#import re

#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')

#stop_words = set(stopwords.words('english'))
#lemmatizer = WordNetLemmatizer()


#def preprocess(text):
     #Lowercase
     #text = text.lower()

     #Tokenize
     #tokens = word_tokenize(text)

     #Remove punctuation and stopwords, lemmatize
     #cleaned_tokens = []
     #for token in tokens:
      #  if token in string.punctuation:
       #     continue
        #if token in stop_words:
         #   continue
        #lemma = lemmatizer.lemmatize(token)
        #cleaned_tokens.append(lemma)
        #return cleaned_tokens


#def extract_features(words):
 #   return {word: True for word in words}


#Read corpus
#data = []
#with open('corpus.txt', 'r', encoding='utf-8') as f:
 #   for line in f:
  #      line = line.strip()
      #   Extract text and label from each line using regex
   #     match = re.match(r'^"(.*)",\s*(\d+)$', line)
    #    if match:
     #       text = match.group(1)
      #      label = match.group(2)
       #     tokens = preprocess(text)
        #    features = extract_features(tokens)
         #   data.append((features, label))

 #Example: Print first 3 preprocessed samples
#for sample in data[:3]:
 #   print(sample)

import nltk
#from nltk.tokenize import word_tokenize
#from nltk.classify import NaiveBayesClassifier
#from nltk.classify.util import accuracy

# exemplu simplu
#texte = [
#    "I feel so happy today!",
#    "I am very sad and depressed.",
 #   "I am angry about what happened.",
  #  "I feel anxious and worried."
#]
#clase = ['fericire', 'tristete', 'furie', 'anxietate']

#def extract_features(text):
#    tokens = word_tokenize(text.lower())
#    return {word: True for word in tokens}

#data = [(extract_features(text), label) for text, label in zip(texte, clase)]

#train_set = data[:int(len(data)*0.8)]
#test_set = data[int(len(data)*0.8):]

#classifier = NaiveBayesClassifier.train(train_set)

#print("Acuratețe:", accuracy(classifier, test_set))

#classifier.show_most_informative_features(20)

#import re
import nltk
#from nltk.tokenize import word_tokenize
#from nltk.classify import NaiveBayesClassifier
#from nltk.classify.util import accuracy

#nltk.download('punkt')

# 1. Citește fișierul și extrage text + clasă
#texte = []
#clase = []

#with open("corpus.txt", "r", encoding="utf-8") as f:
#    for linie in f:
#        linie = linie.strip()
#        match = re.match(r'^\s*[“"](.*?)[”"]\s*,\s*(\d+)$', linie)
#        if match:
#            text = match.group(1)
#            clasa = match.group(2)
#            texte.append(text)
#            clase.append(clasa)

# 2. Feature extraction
#def extrage_cuvinte(text):
#    tokens = word_tokenize(text.lower())
#    return {cuvant: True for cuvant in tokens}

# 3. Creează setul de date
#date_features = [(extrage_cuvinte(text), label) for text, label in zip(texte, clase)]

# 4. Împărțire train/test
#train_set = date_features[:int(0.8 * len(date_features))]
#test_set = date_features[int(0.8 * len(date_features)):]

# 5. Antrenare clasificator
#clasificator = NaiveBayesClassifier.train(train_set)

# 6. Acuratețe
#print("Acuratețe:", accuracy(clasificator, test_set))

# 7. Trăsături informative
#clasificator.show_most_informative_features(20)


#from nltk.corpus import stopwords
#from nltk.stem import WordNetLemmatizer

#stop_words = set(stopwords.words('english'))
#lemmatizer = WordNetLemmatizer()

#def extrage_cuvinte(text):
#    tokens = word_tokenize(text.lower())
#    tokens = [lemmatizer.lemmatize(t) for t in tokens if t.isalpha() and t not in stop_words]
#    return {cuvant: True for cuvant in tokens}
#import re
import nltk
#from nltk.tokenize import word_tokenize
#from nltk.classify import NaiveBayesClassifier
#from nltk.classify.util import accuracy
from nltk.corpus import stopwords
#nltk.download('punkt')
#nltk.download('stopwords')

# 1. Citește și extrage text + etichete
#texte = []
#clase = []

#with open("corpus.txt", "r", encoding="utf-8") as f:
 #   for linie in f:
  #      linie = linie.strip()
   #     match = re.match(r'^[“"](.*?)[”"]\s*,\s*(\d+)$', linie)
    #    if match:
     #       text = match.group(1)
      #      clasa = match.group(2)
       #     texte.append(text)
        #    clase.append(clasa)

# 2. Extrage caracteristici din text
#def extrage_cuvinte(text):
#    tokens = word_tokenize(text.lower())
#    return {cuvant: True for cuvant in tokens if cuvant.isalpha()}  # ignoră semnele de punctuație

# 3. Set de date cu trăsături și etichete
#date_features = [(extrage_cuvinte(text), label) for text, label in zip(texte, clase)]

# 4. Împărțire în set de antrenare și test
#procent_train = 0.8
#limita = int(len(date_features) * procent_train)
#train_set = date_features[:limita]
#test_set = date_features[limita:]

# 5. Antrenare model
#clasificator = NaiveBayesClassifier.train(train_set)

# 6. Evaluare
#print("Acuratețe:", accuracy(clasificator, test_set))
#clasificator.show_most_informative_features(20)


import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Descărcare resurse necesare
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')

# Funcție de extragere a trăsăturilor
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

from nltk.corpus import stopwords
import string

stop_words = set(stopwords.words('english'))

def extrage_cuvinte(text):
    tokens = word_tokenize(text.lower())
    tokens = [cuvant for cuvant in tokens if cuvant.isalpha() and cuvant not in stop_words]
    return {cuvant: True for cuvant in tokens}


# Citește corpusul
texte = []
clase = []

with open("corpus.txt", "r", encoding="utf-8") as f:
    for linie in f:
        linie = linie.strip()
        match = re.match(r'^\s*[“"]?(.*?)[”"]?\s*,\s*(\d+)\s*$', linie)
        if match:
            text = match.group(1)
            clasa = match.group(2)
            texte.append(text)
            clase.append(clasa)

# Pregătire seturi de antrenare/test
date_features = [(extrage_cuvinte(text), label) for text, label in zip(texte, clase)]
limita = int(len(date_features) * 0.8)
train_set = date_features[:limita]
test_set = date_features[limita:]

# Antrenare model
clasificator = NaiveBayesClassifier.train(train_set)


# Evaluare
print("Acuratețe:", accuracy(clasificator, test_set))
print("Most Informative Features")
clasificator.show_most_informative_features(20)

# Testare propoziție nouă
#propozitie_noua = input("\nIntrodu o propoziție pentru clasificare: ")
#features = extrage_cuvinte(propozitie_noua)
#eticheta = clasificator.classify(features)
#print(f"Eticheta prezisă: {eticheta} (1=Anxietate, 2=Fericire, 3=Furie, 4=Tristețe)")

#from sklearn.metrics import confusion_matrix, classification_report

#y_true = [label for (_, label) in test_set]
#y_pred = [clasificator.classify(features) for (features, _) in test_set]

#print(classification_report(y_true, y_pred))


# Set de testare – propoziții fără etichetă
#propozitii_test = [
#    "I can’t stop worrying about what might go wrong.",
#    "Just had the best weekend with friends ❤️",
#    "I’m constantly second-guessing every decision I make.",
#    "I finally achieved my goal and I’m so proud.",
#    "My heart races every time I think about tomorrow.",
#    "This is the happiest I’ve been in years 😊",
#    "I don’t know if I can handle this anymore 😟",
#    "It feels like everything is spiraling out of control.",
#    "Stop blaming me for your mistakes! 😤",
#    "I can’t stop smiling, life feels perfect right now.",
#    "Today has been absolutely amazing!",
#    "I’m so sick of being treated like I don’t matter.",
#    "I can’t believe they did that — I’m furious.",
#    "This is completely unacceptable and I won’t stand for it.",
#    "I feel like crying and I don’t know why.",
#    "There’s this constant sadness I can’t explain.",
#    "Everything feels pointless lately.",
#]

#print("\nClasificare automată pe setul de testare:\n")
#for idx, prop in enumerate(propozitii_test, start=1):
#    features = extrage_cuvinte(prop)
#    eticheta_prezisa = clasificator.classify(features)
#    print(f"{idx}. \"{prop}\" → Etichetă prezisă: {eticheta_prezisa} (1=Anxietate, 2=Fericire, 3=Furie, 4=Tristete)")

# Set de testare – propoziții cu etichete reale (1=Anxietate, 2=Fericire, 3=Furie)
propozitii_test_etichetate= [
    ("I can’t stop worrying about what might go wrong.", "1"),
    ("Just had the best weekend with friends ❤️", "2"),
    ("I’m constantly second-guessing every decision I make.", "1"),
    ("I finally achieved my goal and I’m so proud.", "2"),
    ("My heart races every time I think about tomorrow.", "1"),
    ("This is the happiest I’ve been in years 😊", "2"),
    ("I don’t know if I can handle this anymore 😟", "1"),
    ("It feels like everything is spiraling out of control.", "1"),
    ("Stop blaming me for your mistakes! 😤", "3"),
    ("I can’t stop smiling, life feels perfect right now.", "2"),
    ("Today has been absolutely amazing!", "2"),
    ("I’m so sick of being treated like I don’t matter.", "3"),
    ("I can’t believe they did that — I’m furious.", "3"),
    ("This is completely unacceptable and I won’t stand for it.", "3"),
    ("The clock kept ticking, but time stopped for me.", "4"),
    ("An empty chair still waits at the table.", "4"),
    ("The echo hurts more than the absence.", "4")
]

print("\nClasificare automată pe setul de testare:\n")
corecte = 0
for idx, (prop, eticheta_corecta) in enumerate(propozitii_test_etichetate, start=1):
    features = extrage_cuvinte(prop)
    eticheta_prezisa = clasificator.classify(features)
    corecte += eticheta_prezisa == eticheta_corecta
    print(f'{idx}. "{prop}" → Etichetă prezisă: {eticheta_prezisa}, Etichetă reală: {eticheta_corecta} '
          f'{"✅" if eticheta_prezisa == eticheta_corecta else "❌"}')

print(f"\nAcuratețea pe setul de testare manual: {corecte}/{len(propozitii_test_etichetate)} = {corecte / len(propozitii_test_etichetate):.2%}")

with open("rezultate_testare.txt", "w", encoding="utf-8") as f:
    f.write("Clasificare automată pe setul de testare:\n\n")
    corecte = 0
    for idx, (prop, eticheta_corecta) in enumerate(propozitii_test_etichetate, start=1):
        features = extrage_cuvinte(prop)
        eticheta_prezisa = clasificator.classify(features)
        corecte += eticheta_prezisa == eticheta_corecta
        rezultat = f'{idx}. "{prop}" → Etichetă prezisă: {eticheta_prezisa}, Etichetă reală: {eticheta_corecta} ' \
                   f'{"✅" if eticheta_prezisa == eticheta_corecta else "❌"}\n'
        f.write(rezultat)

    f.write(f"\nAcuratețea pe setul de testare manual: {corecte}/{len(propozitii_test_etichetate)} = "
            f"{corecte / len(propozitii_test_etichetate):.2%}\n")
