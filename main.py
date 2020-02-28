# ! /usr/bin/python
# -*- coding: utf-8 -*-

from predpriprava import *
import argparse
from keras.models import load_model

def main():
    # logger.debug('input params')
    parser = argparse.ArgumentParser(
        description='Arguments for topic classification of input transcriptions from conversation between operator and caller.')
    #tady první parametr je tedy co se zadává při volání k tomuto spouštěcímu souboru, další je --nazevPromene (volá se pak args.nazevPromene), další je help při kterej se napíše při zavolání,
    #poslední je default proměné nebo defaultní cesta
    parser.add_argument('-clp', '--vstup', help='Input files that would be classified to specific topic.', default="Vstup/")
    parser.add_argument('-nt', '--numberOfTopics', help='Number of most probable topics to each record (default value is set to 3).', default=3)
    parser.add_argument('-o', '--output', help='Name of output file (default value is "results").', default="results")

    args = parser.parse_args()

    if os.path.isfile('TFIDFmat.npy'):
        tfidf_vectorizer, nazvySoub, cisloToShl = np.load('TFIDFmat.npy')
    else:
        print('Nutné dodat TF-IDF matici.')
        exit()

    if os.path.isfile('neu_model.h5'):
        model1 = load_model('neu_model.h5')
    else:
        print("Nutné natrénovat model neuronové sítě. ")
        exit()

    nacteneSoubory = NactiMLFStereoAuto2(args.vstup)
    retezceTextyNahravky = {}
    for id in nacteneSoubory:
        retezceTextyNahravky[id] = u' '.join(nacteneSoubory[id]["vse"]["slova"])
    vycisteneTexty, lemmaTexty, tagsTexty = UpravAVysictiTexty2(retezceTextyNahravky)

    results = {}
    for nahravka in lemmaTexty:
        vektor_TFIDF_nahravky = tfidf_vectorizer.transform([u' '.join(lemmaTexty[nahravka])])

        x_test = vektor_TFIDF_nahravky.astype('float32')
        predicted_probs = model1.predict(x_test)[0]
        topics = {}
        for i in range(int(args.numberOfTopics)):
            predicted = np.argmax(predicted_probs)
            prob = predicted_probs[predicted] * 100.0
            predicted_probs[predicted] = 0.0
            topics[cisloToShl[predicted].strip()] = prob
        results[nahravka] = topics

    #výsledky jsou dictionary kde key je vždy id nahrávky a obsahuje vždy další dictionary kde jsou klíče témata a hodnoty jsou pravděpodobnosti daných témat.
    np.save(args.output, results)

    #Jen vypsání výsledků

    mezeraPred = u'      '
    mezeraMezi = u'                                '
    file1 = open(args.output + u'.txt', 'w', encoding='utf8')
    file1.write('Výsledky ' + str(int(len(results))) + ' vstupních souborů jsou uloženy do souboru : ' + str(args.output))
    file1.write('\n')
    for nahravka in results:
        file1.write('Nahrávce s id ' + str(nahravka) + ' jsou přiřazeny následující témata: ')
        file1.write('\n')
        file1.write(mezeraPred + '        Téma' + mezeraMezi[0:len(mezeraMezi)-len('Téma        ')] + 'Pravděpodobnost')
        file1.write('\n')
        for tema in results[nahravka]:
            file1.write(mezeraPred + str(tema) + mezeraMezi[0:len(mezeraMezi)-len(tema)] + str(results[nahravka][tema]))
            file1.write('\n')
        file1.write('\n')
    file1.close()


if __name__ == "__main__":
    main()