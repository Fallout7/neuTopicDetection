# -*- coding: utf-8 -*-
# coding: utf-8

from predpriprava import *
from evaluation import *
from sklearn import svm
import keras, h5py, random, os
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
from keras.models import load_model
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

def get_prah(y, ypred):
    prah, bestPrahACC, bestPrahF1, bestPrahHamm, bestScoreACC, bestScoreF1, bestScoreHamm = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0
    for i in range(1000):
        predicted = []
        for p in ypred:
            pPom = []
            for hod in p:
                if hod > prah:
                    pPom.append(1)
                else:
                    pPom.append(0)
            predicted.append(pPom)
        predicted = np.array(predicted)

        poc, pocShod = 0.0, 0.0
        if prah > 0.05:
            for j in range(len(predicted)):
                for k in range(len(predicted[j])):
                    if y[j][k] == 1:
                        poc += 1.0
                        if predicted[j][k] == 1:
                            pocShod += 1.0
                    #poc += 1.0
                    #if predicted[j][k] == y[j][k]:
                    #    pocShod += 1.0
            acc = pocShod / (poc / 100.0)

        proved = 1
        for vek in predicted:
            poc, pocKon = len(vek), 0
            for hod in vek:
                if hod == 1:
                    pocKon += 1
            if poc == pocKon:
                proved = 0
        if proved == 1:
            f1 = f1_score(y_true=y, y_pred=predicted, average='weighted')
            if f1 > bestScoreF1:
                bestScoreF1 = f1
                bestPrahF1 = prah
            hammL = hamming_loss(y, predicted)
            if hammL < bestScoreHamm:
                bestScoreHamm = hammL
                bestPrahHamm = prah
            if prah > 0.05:
                if acc > bestScoreACC:
                    bestScoreACC = acc
                    bestPrahACC = prah

        prah += 0.001
    return bestPrahACC, bestPrahF1, bestPrahHamm

def get_score(y, ypred, bestPrahACC, bestPrahF1, bestPrahHamm):
    predictedAcc = []
    for p in ypred:
        pPom = []
        for hod in p:
            if hod > bestPrahACC:
                pPom.append(1)
            else:
                pPom.append(0)
        predictedAcc.append(pPom)
    predictedAcc = np.array(predictedAcc)
    predictedf1 = []
    for p in ypred:
        pPom = []
        for hod in p:
            if hod > bestPrahF1:
                pPom.append(1)
            else:
                pPom.append(0)
        predictedf1.append(pPom)
    predictedf1 = np.array(predictedf1)
    predictedHamm = []
    for p in ypred:
        pPom = []
        for hod in p:
            if hod > bestPrahHamm:
                pPom.append(1)
            else:
                pPom.append(0)
        predictedHamm.append(pPom)
    predictedHamm = np.array(predictedHamm)
    poc, pocShod = 0.0, 0.0
    for j in range(len(predictedAcc)):
        for k in range(len(predictedAcc[j])):
            if y[j][k] == 1:
                poc += 1.0
                if predictedAcc[j][k] == 1:
                    pocShod += 1.0
            #poc += 1.0
            #if predictedAcc[j][k] == y[j][k]:
            #    pocShod += 1.0
    acc = pocShod / (poc / 100.0)
    f1 = f1_score(y_true=y, y_pred=predictedf1, average='weighted')
    hammmL = hamming_loss(y, predictedHamm)
    return acc, f1, hammmL, predictedAcc, predictedf1, predictedHamm

def PredelejNaBiTriAtdGram(texty, kolikGram, posunOk):
    textyKolikGram = {}
    for keyy in texty:
        slova = texty[keyy]
        slovakolikgram = []
        kolikGramSlova = []
        poz = 0
        while poz < len(slova):
            if poz + kolikGram < len(slova):
                for i in range(kolikGram):
                    kolikGramSlova.append(slova[poz + i])
            else:
                pocc = 0
                for i in range(kolikGram):
                    if poz + i < len(slova):
                        kolikGramSlova.append(slova[poz + i])
                    else:
                        kolikGramSlova.append(slova[pocc])
                        pocc += 1
            poz += posunOk
            slovakolikgram.append('_'.join(kolikGramSlova))
            kolikGramSlova = []
        textyKolikGram[keyy] = slovakolikgram
    return textyKolikGram

def neuProvedAUloz(tfidfTrain, pozadVystupTrain, num_classes, train_partition_name, epochs, batch_size, dropout, dropout2, un, un2, un3, soub, vstup):
    train_matrix_essays = tfidfTrain
    x_train = train_matrix_essays
    ypom = np.array(pozadVystupTrain)
    # jednoduché rozdělení
    vel = x_train.shape[0]
    velVal = int((vel / 100.0) * 25.0)
    pocTrain = vel - velVal
    X_trainKon, X_validKon, y_trainKon, y_validKon, pouzPoz, soubVal = [], [], [], [], {}, []
    while len(X_trainKon) + len(X_validKon) < vel:
        pozNah = random.randint(0, vel-1)
        if pozNah not in pouzPoz:
            pouzPoz[pozNah] = pozNah
            if len(X_trainKon) < pocTrain:
                X_trainKon.append(x_train[pozNah])
                y_trainKon.append(ypom[pozNah])
            else:
                X_validKon.append(x_train[pozNah])
                y_validKon.append(ypom[pozNah])
                soubVal.append(soub[pozNah])
    X_trainKon = np.array(X_trainKon)
    X_validKon = np.array(X_validKon)
    y_trainKon = np.array(y_trainKon)
    y_validKon = np.array(y_validKon)
    #X_trainKon, X_validKon = x_train[0:(vel - velVal)], x_train[(vel - velVal):vel]
    #y_trainKon, y_validKon = ypom[0:(vel - velVal)], ypom[(vel - velVal):vel]
    #soubVal = soub[(vel - velVal):vel]
    print("tady provedeno rozdělení")
    print(X_trainKon.shape, y_trainKon.shape, X_validKon.shape, y_validKon.shape)
    x_trainNew, y_trainNew, soubNew = [], [], []
    for i in range(len(X_trainKon)):
        poc = 0
        for hod in y_trainKon[i]:
            if hod == 1:
                poc += 1
        if poc == 1:
            x_trainNew.append(X_trainKon[i])
            y_trainNew.append(y_trainKon[i])
        else:
            for j in range(poc):
                x_trainNew.append(X_trainKon[i])
            for j in range(len(y_trainKon[i])):
                if y_trainKon[i][j] == 1:
                    vekpom = [0] * len(y_trainKon[i])
                    vekpom[j] = 1
                    y_trainNew.append(vekpom)
    x_train, ypom = np.array(x_trainNew), np.array(y_trainNew)
    print(x_train.shape, ypom.shape)

    """
    hodnota = 0.1
    y_train_pom = []
    for vek in y_train:
        vekPom = []
        poc1 = 0
        for hod in vek:
            if hod == 1:
                poc1 += 1
        for hod in vek:
            if not hod == 1:
                vekPom.append(hodnota + (poc1 * 0.05))
            else:
                vekPom.append(hod)
        y_train_pom.append(vekPom)
    y_train = np.array(y_train_pom)
    y_valid_pom = []
    for vek in y_valid:
        vekPom = []
        poc1 = 0
        for hod in vek:
            if hod == 1:
                poc1 += 1
        for hod in vek:
            if not hod == 1:
                vekPom.append(hodnota + (poc1 * 0.05))
            else:
                vekPom.append(hod)
        y_valid_pom.append(vekPom)
    y_validKonec = np.copy(y_valid)
    y_valid = np.array(y_valid_pom)
    """
    """
    #MultilabelStratifiedShuffleSplit rozdělení
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)

    for train_index, test_index in msss.split(x_train, ypom):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_valid = x_train[train_index], x_train[test_index]
        y_train, y_valid = ypom[train_index], ypom[test_index]

    """
    if vstup == "MovieSum":
        vel = x_train.shape[0]
        velVal = int((vel / 100.0) * 25.0)
        pocTrain = vel - velVal
        X_train, X_valid, y_train, y_valid, pouzPoz = [], [], [], [], {}
        while len(X_train) + len(X_valid) < vel:
            pozNah = random.randint(0, vel - 1)
            if pozNah not in pouzPoz:
                pouzPoz[pozNah] = pozNah
                if len(X_train) < pocTrain:
                    X_train.append(x_train[pozNah])
                    y_train.append(ypom[pozNah])
                else:
                    X_valid.append(x_train[pozNah])
                    y_valid.append(ypom[pozNah])
    else:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
        sss.get_n_splits(x_train, ypom)
        for train_index, test_index in sss.split(x_train, ypom):
            X_train, X_valid = x_train[train_index], x_train[test_index]
            y_train, y_valid = ypom[train_index], ypom[test_index]


    y_valid_pom = []
    for i in range(len(y_valid)):
        for j in range(len(y_valid[i])):
            if not y_valid[i][j] == 0:
                y_valid_pom.append(j)


    file_neu_name = vstup + '_neu_model_' + str(un) + "_" + str(batch_size) + '.h5'
    souboryPS, slozkyPS = ZiskejNazvySouboru("PomocneSouboryNeu/", file_neu_name)
    if len(souboryPS) == 0:
        model_path = ('PomocneSoubNeu/NNessey_model1_trigram50tis_{ep}epochs{ba}batch.h5')
        train_model_path = model_path.format(partition=train_partition_name, ep=epochs, ba=batch_size)

        x_train = x_train.astype('float32')

        model1 = Sequential()  # uz neni jina moznost modelu
        model1.add(Dense(units=un, activation='relu', input_dim=x_train.shape[1], name='dense_1')) #nejlepší druhé spuštění s tanh a softmax a rmsprop optimizerem
        model1.add(Dropout(dropout, name='dropout_1'))
        #model1.add(Dense(units=un2, activation='relu', name='dense_2'))  # nejlepší druhé spuštění s tanh a softmax a rmsprop optimizerem
        #model1.add(Dropout(dropout2, name='dropout_2'))
        #model1.add(Dense(units=un3, activation='relu', name='dense_3'))  # nejlepší druhé spuštění s tanh a softmax a rmsprop optimizerem
        #model1.add(Dropout(dropout, name='dropout_3'))
        #zkusit popřípadě batch normalizaci

        model1.add(Dense(num_classes, activation='sigmoid', name='dense_5'))

        sgd = keras.optimizers.SGD(lr=0.001, decay=0, momentum=0.9, nesterov=True)
        rmspropOpt = keras.optimizers.RMSprop(lr=0.001, epsilon=None, decay=0.0)
        model1.compile(loss="binary_crossentropy", optimizer=sgd,
                       metrics=['accuracy'])  # nstaveni ucici algoritmus

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.01, patience=50, verbose=1, mode='auto', cooldown=0, min_lr=0.001)#, restore_best_weights=True)
        early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=100, verbose=1, restore_best_weights=True)

        history1 = model1.fit(X_train, y_train,
                              batch_size=batch_size,
                              epochs=epochs,
                              verbose=2, callbacks=[reduce_lr, early_stop], validation_data=(X_valid, y_valid))  # natrenuj  .. v priade nevejde do mpameti ...  misto fit train_on_batch (nutne zabespecit nastaveni trenovani)

        model1.save("PomocneSouboryNeu/" + file_neu_name)


    else:
        model1 = load_model("PomocneSouboryNeu/" + file_neu_name)



    """
    predicted_probs = model1.predict(X_valid)
    predicted = np.argmax(predicted_probs, axis=1)
    pocShod = 0.0
    for i in range(len(predicted_probs)):
        if not predicted[i] == y_valid_pom[i]:
            max1 = np.max(predicted_probs[i])
            max2, shl2 = 0.0, 0
            for j in range(len(predicted_probs[i])):
                if not predicted_probs[i][j] == max1:
                    if max2 < predicted_probs[i][j]:
                        max2 = predicted_probs[i][j]
                        shl2 = j
            max3, shl3 = 0.0, 0
            for j in range(len(predicted_probs[i])):
                if not predicted_probs[i][j] == max1 and not predicted_probs[i][j] == max2:
                    if max3 < predicted_probs[i][j]:
                        max3 = predicted_probs[i][j]
                        shl3 = j
            if shl2 == y_valid_pom[i] or shl3 == y_valid_pom[i]:
                pocShod += 1
        else:
            pocShod += 1.0
    print "Spadá do 3 nejlepších acc: " + str(pocShod/X_valid.shape[0])
    """
    predicted_probs_Valid = model1.predict(X_valid)
    y_valid_prah = y_valid
    bestPrahACC, bestPrahF1, bestPrahHamm = get_prah(y_valid_prah, predicted_probs_Valid)

    file_name = "Vysledky/Vysledky_" + vstup
    if os.path.exists(file_name):
        file_work = open(file_name, "a+")
    else:
        file_work = open(file_name, "w+")

    #y_valid = np.copy(y_validKonec)
    print("Vstup: " + vstup + "; velikost relu vrstvy: " + str(un) + "; batch size: " + str(batch_size))
    print(X_valid.shape)
    file_work.write("Vstup: " + vstup + "; velikost relu vrstvy: " + str(un) + "; batch size: " + str(batch_size))
    file_work.write("\n")
    X_valid, y_valid = np.copy(X_validKon), np.copy(y_validKon)
    predicted_probs = model1.predict(X_valid)

    print(len(y_valid), len(predicted_probs))
    acc, f1, hammmL, predictedAcc, predictedf1, predictedHamm = get_score(y_valid, predicted_probs, bestPrahACC, bestPrahF1, bestPrahHamm)

    np.save("Prahy_neu", [bestPrahACC, bestPrahF1, bestPrahHamm])

    bestPrahACC, bestPrahF1, bestPrahHamm = np.load("Prahy_neu.npy")

    print('Výsledek NEU ACC: ' + str(acc) + ' s nastaveným prahem na: ' + str(bestPrahACC))
    print(f1_score(y_true=y_valid, y_pred=predictedAcc, average='weighted'), hamming_loss(y_valid, predictedAcc))
    file_work.write('    Výsledek NEU ACC: ' + str(acc) + ' s nastaveným prahem na: ' + str(bestPrahACC))
    file_work.write("\n")
    file_work.write('    ' + str(f1_score(y_true=y_valid, y_pred=predictedAcc, average='weighted')) + " " + str(hamming_loss(y_valid, predictedAcc)))
    file_work.write("\n")
    file_work.write("\n")
    for i in range(10):
        print(soubVal[i])
        print(list(predictedAcc[i]))
        print(list(y_valid[i]))
        print("")

    print('Výsledek NEU F1: ' + str(f1) + ' s nastaveným prahem na: ' + str(bestPrahF1))
    poc, pocShod = 0.0, 0.0
    for j in range(len(predictedf1)):
        for k in range(len(predictedf1[j])):
            if y_valid[j][k] == 1:
                poc += 1.0
                if predictedHamm[j][k] == 1:
                    pocShod += 1.0
    acc = pocShod / (poc / 100.0)
    print(acc, hamming_loss(y_valid, predictedf1))
    for i in range(10):
        print(soubVal[i])
        print(list(predictedf1[i]))
        print(list(y_valid[i]))
        print(list(predicted_probs[i]))
        print("")
    file_work.write('    Výsledek NEU F1: ' + str(f1) + ' s nastaveným prahem na: ' + str(bestPrahF1))
    file_work.write("\n")
    file_work.write('    ' + str(acc) + " " + str(hamming_loss(y_valid, predictedf1)))
    file_work.write("\n")
    file_work.write("\n")

    print('Výsledek NEU HammLoss: ' + str(hammmL) + ' s nastaveným prahem na: ' + str(bestPrahHamm))
    poc, pocShod = 0.0, 0.0
    for j in range(len(predictedHamm)):
        for k in range(len(predictedHamm[j])):
            #poc += 1.0
            if y_valid[j][k] == 1:
                poc += 1.0
                if predictedHamm[j][k] == 1:
                    pocShod += 1.0
            #if predictedHamm[j][k] == y_valid[j][k]:
            #   pocShod += 1.0
    acc = pocShod / (poc / 100.0)
    print(acc, f1_score(y_true=y_valid, y_pred=predictedHamm, average='weighted'))
    for i in range(10):
        print(soubVal[i])
        print(list(predictedHamm[i]))
        print(list(y_valid[i]))
        print("")
    file_work.write('    Výsledek NEU HammLoss: ' + str(hammmL) + ' s nastaveným prahem na: ' + str(bestPrahHamm))
    file_work.write("\n")
    file_work.write('    ' + str(acc) + " " + str(f1_score(y_true=y_valid, y_pred=predictedHamm, average='weighted')))
    file_work.write("\n")
    file_work.write("\n")
    file_work.write("\n")
    file_work.write("\n")
    file_work.close()
#nejlepší výsledky s velikostí LSA 500 a relu vrstvou 128 a sigmoid a sgd -- dole otevřené a uložená neu a matice v naki ujcskriptTest
# teď zkouška ještě s celou tfidf maticí, tady o velikosti 128 relu a v terminalu o vel 512 a znova tady o velikosti 1024