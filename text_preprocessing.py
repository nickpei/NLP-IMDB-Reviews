import os
import string
import nltk
import time
import json
import pandas as pd
from nltk.corpus import stopwords
def ReadOneFile(fileName):
    contents = []
    with open(fileName, 'r', encoding='UTF-8') as file:
        for line in file:
            contents.append(line.rstrip('\n').lower())
   # file.close()
    result = ''.join(contents)

    #remove punctuations
    special_char = ["‘", "’", "·", "–", "“", "”"]
    result = result.translate(str.maketrans('', '', string.punctuation)).translate({ord(c): 'special char' for c in special_char})
    result = result.split()

    #remove stopwords in result
    stop_words = stopwords.words('english')
    result = [w for w in result if w not in stop_words]
    
    return result

def ReadFiles(fileName):
    data = []
    directory_top = "C:/Users/nicho/Desktop/IMDB_Dataset/" + fileName + "/"
    for data_class in os.listdir(directory_top):
        directory_class = directory_top + data_class + "/"
        for file in os.listdir(directory_class):
            #print('Processing...:',data_class + file)
            words = ReadOneFile(directory_class + file)
            example = {x:words.count(x) for x in words}
            example['__FileID__'] = file
            example['__CLASS__'] = 1 if data_class[:3] == 'pos' else 0
            data.append(example)
    return data

start = time.time()
data_train = ReadFiles('train')
with open('data_train_wostopNpunc.txt','w', encoding='utf-8') as f:
    json.dump(data_train, f)
end = time.time()
print(end - start)

##with open('C:/Users/nicho/Desktop/data_train_wostopNpunc.txt','r', encoding='utf-8') as f:
##    data_train = json.load(f)
##
##df = pd.DataFrame(data_train).fillna(0)
##print(df.shape)

