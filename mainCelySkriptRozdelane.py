# ! /usr/bin/python
# -*- coding: utf-8 -*-

from predpriprava import *
from sklearn import svm
import argparse
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten

# zjišťuje názvy složek a souborů v dané složce (vstupem je: složka kde má hledat a typ souborů co má hledat)
# vrací jak názvy podsložek tak názvy souborů
def ZiskejNazvySouboru(treeroot, pattern):
    results = []
    filess = []
    dirrs = []
    for base, dirs, files in os.walk(treeroot):
        goodfiles = fnmatch.filter(files, pattern)
        results.extend(os.path.join(base, f) for f in goodfiles)
    for i in range(len(results)):
        (dirname, filename) = os.path.split(results[i])
        filess.append(filename)
        dirrs.append(dirname)
    return filess, dirrs


def main():

    # logger.debug('input params')

    parser = argparse.ArgumentParser(
        description='Compute features on liver and other tissue.')
    #tady první parametr je tedy co se zadává při volání k tomuto spouštěcímu souboru, další je --nazevPromene (volá se pak args.nazevPromene), další je help při kterej se napíše při zavolání,
    #poslední je default proměné nebo defaultní cesta
    parser.add_argument('-trp', '--trainPath',
                        help='Input training files in folder according to theirs topic.', default="Train/")
    parser.add_argument('-clp', '--classPath', help='Input files that would be classified to specific topic.', default="ClassFiles/")

    parser.add_argument('-svm', '--svmSpustit',
                        help='For running LinearSVM classification set this parametr to 1.', default=0)
    parser.add_argument('-neu', '--neuSpustit',help='For running Neural Network classification set this parametr to 1.',
                        default=0)

    parser.add_argument('-svmT', '--svmReTrain', help='For reTrain LinearSVM classification set this parametr to 1 (For training data use folder Train/ , otherwise set different arg -trp).', default=0)
    parser.add_argument('-neuT', '--neuReTrain', help='For reTrain Neural Network classification set this parametr to 1 (For training data use folder Train/ , otherwise set different arg -trp).', default=0)

    parser.add_argument('-o', '--output', help='output file',
                        default="20130919_liver_statistics_results.pkl")
    parser.add_argument('-t', '--train', help='Training', default=False,
                        action='store_true')

    parser.add_argument('-fi', '--classFilesPath',
                        help='Path to trained SVM classificator and trained Nerual Network.',
                        default='SouboryKlasifikatoru/')
    args = parser.parse_args()

    if args.svmReTrain == 1:
        trainFolder = args.trainPath

        #Tady načíst vstup a upravid do matic, klasický postup. Samozřejmě z dané složky.
        soubAtextyRaw, soubAslozky = NacteniRawVstupu(trainFolder)
        vycisteneTexty, lemmaTexty, tagsTexty = UpravAVysictiTexty(soubAtextyRaw, trainFolder, 'czech')
        slovnik, slovnikPole = VytvorVocab(vstupPrac, velikostSlovniku, textyPracovni)
        tfidfMat, nazvySoub = vytvorTFIDF(vstupPrac, textyPracovni, slovnikPole)


        clf = svm.LinearSVC()
        clf.fit(tfidfTrain, pozadVystupTrain)


    if args.neuReTrain == 1:
        trainFolder = args.trainPath


    if args.svmSpustit == 1:
        hlSb = 'SVM.p'
        souboryPS, slozkyPS = ZiskejNazvySouboru(args.classFilesPath, hlSb)
        if len(souboryPS) == 0:
            clf = pickle.load(open(args.classFilesPath + hlSb, "rb"))

    if args.neuSpustit == 1:


# Ukládání výsledku do souboru
    output_file = os.path.join(path_to_script, args.output)
    misc.obj_to_file(result, output_file, filetype='pickle')

if __name__ == "__main__":
main()