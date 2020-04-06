# ! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from upravaDat import *
from predpriprava import *
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from bs4 import BeautifulSoup

import pickle, h5py, json
from os import path
from sklearn.preprocessing import Normalizer

jazyk = 'czech'
vstup = 'TrainDataCeleNahravky'
vstup = "MovieSum"
jazyk = 'english'
#vstup = 'Vstup3raw'
#vstup = 'Vstup3raw10NG'

np.random.seed(1234)
SCRIPT_DIR = path.dirname(path.realpath(__file__))

if jazyk == "czech":

    textyNahravky, temataNahravky, vystupVek, temaToCislo, cisloToTema = np.load(vstup + '/vstupCeleNahravkyATemata.npy', allow_pickle = True)
    retezceTextyNahravky = {}
    for id in textyNahravky:
        retezceTextyNahravky[id] = u' '.join(textyNahravky[id])
    vycisteneTexty, lemmaTexty, tagsTexty = UpravAVysictiTexty(retezceTextyNahravky, vstup, jazyk)

else:
    if vstup == "MovieSum":
        meta = pd.read_csv("MovieSummaries/movie.metadata.tsv", sep = '\t', header = None)
        meta.columns = ["movie_id",1,"movie_name",3,4,5,6,7,"genre"]
        genres = meta[["movie_id","movie_name","genre"]]
        plots = pd.read_csv("MovieSummaries/plot_summaries.txt", sep = '\t', header = None)
        plots.columns = ["movie_id", "plot"]
        genres['movie_id'] = genres['movie_id'].astype(str)
        plots['movie_id'] = plots['movie_id'].astype(str)
        movies = pd.merge(plots, genres, on = 'movie_id')
        genres_lists = []
        for i in movies['genre']:
            genres_lists.append(list(json.loads(i).values()))
        
        soubAtextyRaw, soubAslozky = {}, {}
        temaToCislo, cisloToTema, vystupVek, cis, temaPocet = {}, {}, {}, 0, {}
        for i in range(len(movies["movie_id"])):
            genress = genres_lists[i]
            for ge in genress:
                if ge not in temaToCislo:
                    temaToCislo[ge] = cis
                    cisloToTema[cis] = ge
                    temaPocet[cis] = 1
                    cis += 1
                else:
                    temaPocet[temaToCislo[ge]] = temaPocet[temaToCislo[ge]] + 1
            soubAtextyRaw[movies["movie_id"][i]] = movies["plot"][i]
            soubAslozky[movies["movie_id"][i]] = genress
        print(len(soubAtextyRaw))
        print(len(cisloToTema))
        vekkPocty = {}
        for soub in soubAslozky:
            temata = soubAslozky[soub]
            vekk = [0] * len(cisloToTema)
            for j in range(len(temata)):
                vekk[temaToCislo[temata[j]]] = 1
            vystupVek[soub] = vekk
        print("Původně souborů:" + str(len(soubAslozky)))
        """
        for soub in soubAslozky:
            vekk1 = vystupVek[soub]
            poc = 0
            for soub2 in soubAslozky:
                vekk2 = vystupVek[soub2]
                if vekk1 == vekk2:
                    poc += 1
            vekkPocty[soub] = poc
        soubAslozkyPom, soubAtextyRawPom = {}, {}
        for soub in soubAslozky:
            if vekkPocty[soub] >= 3:
                soubAslozkyPom[soub] = soubAslozky[soub]
                soubAtextyRawPom[soub] = soubAtextyRaw[soub]
        soubAslozky, soubAtextyRaw = soubAslozkyPom, soubAtextyRawPom
        print("Použitelných souborů:" + str(len(soubAslozky)))
        """
        vycisteneTexty, lemmaTexty, tagsTexty = UpravAVysictiTexty(soubAtextyRaw, vstup + "_" + str(len(soubAslozky)), jazyk)

    else:
        soubAtextyRaw, soubAslozky = NacteniRawVstupu(vstup)

        vycisteneTexty, lemmaTexty, tagsTexty = UpravAVysictiTexty(soubAtextyRaw, vstup, jazyk)

        temaToCislo, cisloToTema, vystupVek, cis = {}, {}, {}, 0
        for soub in soubAslozky:
            tema = soubAslozky[soub]
            if tema not in temaToCislo:
                temaToCislo[tema] = cis
                cisloToTema[cis] = tema
                cis += 1
        print(len(cisloToTema))
        for soub in soubAslozky:
            tema = soubAslozky[soub]
            vekk = [0] * len(cisloToTema)
            vekk[temaToCislo[tema]] = 1
            vystupVek[soub] = vekk
            print(vekk)

for vek in lemmaTexty:
    for word in lemmaTexty[vek]:
        print(word)
    break

pocbitriatdgr, posunOk = 1, 1
if pocbitriatdgr > 1:
    lemmaTexty = PredelejNaBiTriAtdGram(lemmaTexty, pocbitriatdgr, posunOk)
if pocbitriatdgr == 1:
    pripis = ''
else:
    pripis = '_' + str(pocbitriatdgr) + '_' + str(posunOk) + '_'
vstupPrac = vstup +'Lemma' + pripis
textyPracovni = lemmaTexty
velikost, okno, alphaa, minalphaa, minimalniPocetCetnostiSlov, Ncomponents = 5000, 0.01, 0.019, 0.019, 5, 500
velikostSlovniku = 10000
#slovnik, slovnikPole = VytvorVocab(vstupPrac, velikostSlovniku, textyPracovni)
if vstup == "MovieSum":
    slovnik, slovnikPole = VytvorVocabIDF(vstupPrac + '_IDF_' + str(len(soubAslozky)) + "_", velikostSlovniku, textyPracovni)
else:
    slovnik, slovnikPole = VytvorVocabIDF(vstupPrac + '_IDF_', velikostSlovniku, textyPracovni)
print((slovnikPole))


tfidf_vectorizer = TfidfVectorizer(max_df=1.0, max_features=None, min_df=0.0, vocabulary=slovnikPole,
                                           stop_words=None, use_idf=True, tokenizer=None, ngram_range=(1, 1),
                                           sublinear_tf=1)

nazvySoub = []
textyPol = []
for key in textyPracovni:
    textyPol.append(u' '.join(textyPracovni[key]))
    nazvySoub.append(key)
pozadVystup = []
for soub in nazvySoub:
    pozadVystup.append(vystupVek[soub])

tfidf_vectorizer.fit(textyPol)

'''
hf = h5py.File('TFIDFmat.h5', 'w')
hf.create_dataset('tfidf_vectorizer', data=tfidf_vectorizer)
hf.create_dataset('nazvySoub', data=nazvySoub)
hf.create_dataset('cisloToShl', data=cisloToShl)
hf.close()
'''
tfidfMat = tfidf_vectorizer.transform(textyPol)

np.save('TFIDFmat', [tfidf_vectorizer, nazvySoub, cisloToTema])

tfidfMat = tfidfMat.toarray()
#maticeDoc2VecVah = VytvorReprDoc2Vec(vstupPrac, textyPracovni, nazvySoub, velikost, okno, alphaa, minalphaa, minimalniPocetCetnostiSlov)
print(tfidfMat.shape)
"""
# část na výpočet výsledků s LSA
svd = TruncatedSVD(algorithm='randomized', n_components=Ncomponents, n_iter=10, random_state=None, tol=0.0)
normalizer = Normalizer(norm='l2', copy=False)
lsa = make_pipeline(svd, normalizer)
lsa.fit(tfidfMat)

np.save('TFIDFmatLSA', [lsa, svd, normalizer])

tfidfMatLSA = lsa.transform(tfidfMat)
"""
pocShluku = len(cisloToTema)
num_classes = pocShluku
print("Počet shluků: " + str(num_classes))
'''
# část na výpočet výsledků s LSA
svd = TruncatedSVD(algorithm='randomized', n_components=Ncomponents, n_iter=10, random_state=None, tol=0.0)
normalizer = Normalizer(norm='l2', copy=False)
lsa = make_pipeline(svd, normalizer)
maticeDoc2VecVahLSA = lsa.fit_transform(maticeDoc2VecVah)

maticeTFIDFD2V = np.append(tfidfMatLSA, maticeDoc2VecVahLSA, axis=1)
maticeCelaTFIDFaD2V = np.append(tfidfMat, maticeDoc2VecVah, axis=1)
'''

epochs = 50000
batch_size = 16
dropout = 0.3
dropout2 = 0.3
un = 512
un2 = 256
un3 = 128
un4 = 320

train_partition_name = "trainPart"
neuProvedAUloz(tfidfMat, pozadVystup, num_classes, train_partition_name, epochs, batch_size, dropout, dropout2, un, un2, un3, nazvySoub, vstup)
#neuProvedAUloz(tfidfMatLSA, pozadVystup, num_classes, train_partition_name, epochs, batch_size, dropout, dropout2, un, un2, un3, nazvySoub, vstup)