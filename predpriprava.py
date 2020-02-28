# -*- coding: utf-8 -*-
# coding: utf-8
import os, fnmatch, time, re, pickle, codecs, math, sklearn, chardet
import numpy as np
from ufal.morphodita import *


class enc:
    def __init__(self, coding):
        self.coding = coding  # type of coding
        self.number = 0  # number of czech chars in the text


# nahrazuje  chybné znaky správnými (chybné znaky vznikají při špatném kódování)
def replace_nonsense_characters(text):
    text = re.sub(u'[”“]', u'"', text)
    text = re.sub(u"’", u"'", text)
    text = re.sub(u"[–—]", u"-", text)
    text = re.sub(u'[\x00-\x08\x0B\x0C\x0E-\x1F\xA0\u2009]+', u' ', text)
    text = re.sub(u'\t+', ' ', text)
    text = re.sub(u'  +', u' ', text)
    text = re.sub(u'( *[\r\n])+ *', u'\n', text)
    return text.strip()


# odhadne jak je daný string kódován, vrací tento odhad
def coding_guess(text):
    encodings = ('ascii', 'iso-8859-2', 'cp1250', 'cp1251', 'cp852', 'utf-8')
    czech_chars = u"áčďéěíňóřšťúůýžÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ"

    # first selection of the coding ... codings that change text to unicode without an error
    encoding = []
    for i in encodings:
        try:
            unicode(text, i)
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
        else:
            encoding.append(enc(i))  # insert into coding list

    # second selection of the coding ... this will choose coding with maximal number of czech chars
    max = -1
    for code in encoding:
        for char in unicode(text, code.coding):
            if char in czech_chars:
                code.number = code.number + 1
        if code.number > max:
            max = code.number
            max_code = code.coding
    return max_code


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



def display_topics(H, W, feature_names, documents, no_top_words):
    vysledkyLDA = []
    for topic_idx, topic in enumerate(H):
        '''
        print "Topic %d:" % (topic_idx)
        print " ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]])
        '''
        top_doc_indices = np.argsort( W[:,topic_idx] )[::-1][0:len(documents)]
        print(top_doc_indices)
        print(len(top_doc_indices))
        for doc_index in top_doc_indices:
            #print documents[doc_index]
            vysledkyLDA.append(topic_idx)
    return vysledkyLDA


def NacteniRawVstupu(kdeHledat):
    hlSb = kdeHledat + '.p'
    souboryPS, slozkyPS = ZiskejNazvySouboru('PomocneSoubory/', hlSb)
    if len(souboryPS) == 0:
        print('Načítání souborů, jejich názvů a originálních shluků složky ' + kdeHledat + ' a uložení. ')
        hlSb = '*'
        souboryPom, slozkyPom = ZiskejNazvySouboru(kdeHledat + '/', hlSb)

        print(len(souboryPom), len(slozkyPom))
        soubKon = {}
        konf = 0
        for sb in souboryPom:
            if sb in soubKon:
                konf = 1
                break
            else:
                soubKon[sb] = sb

        if konf == 1:
            print('Nastal konflikt s názvy souborů (stejné názvy), začíná proces přejmenování souborů aby se tento konflikt vyřešil. ')

            for i in range(len(souboryPom)):
                os.rename(slozkyPom[i] + '/' + souboryPom[i], slozkyPom[i] + '/' + str(00) + str(i))
            souboryPom, slozkyPom = ZiskejNazvySouboru(kdeHledat + '/', hlSb)

        soubory = {}
        souboryText = {}
        for i in range(len(slozkyPom)):
            sl = slozkyPom[i]
            poz = sl.rfind('/') + 1
            if kdeHledat == 'VstupREUTERS':
                patTitle = r"<TITLE>(.*?)</TITLE>"
                patID = r'NEWID="(.*?)">'
                patText = r'<BODY>(.*?)</BODY>'
                zacatek = u'<REUTERS'
                konec = u'</REUTERS>'
                clanekPP = u''
                parsuj = 0
                fileS = open(sl + '/' + souboryPom[i], "r")
                for radka in fileS:
                    radkaPom = (replace_nonsense_characters(unicode(radka.decode(coding_guess(radka)), 'utf-8' ))).strip()
                    #radkaPom = (replace_nonsense_characters(unicode(radka.decode("utf8")))).strip()
                    najdiZacatek = re.findall(zacatek,radkaPom)
                    najdiKonec = re.findall(konec,radkaPom)
                    if parsuj == 0 and len(najdiZacatek) > 0:
                        parsuj = 1
                    if parsuj == 1:
                        clanekPP += u' ' + radkaPom
                    if parsuj == 1 and len(najdiKonec) > 0:
                        parsuj = 0
                        #title = re.findall(patTitle, clanekPP)[0]
                        id = re.findall(patID, clanekPP)[0]
                        jeText = (re.findall(patText, clanekPP))
                        if not len(jeText) == 0:
                            text = jeText[0]
                            souboryText[id] = text.strip()
                            soubory[id] = souboryPom[i]
                        clanekPP = u''
                fileS.close()
            elif not kdeHledat.find('Vstup3raw') == -1:
                clanekPP = u''
                fileS = open(sl + '/' + souboryPom[i], "r")
                zacni = 0
                poprve = 0
                for radka in fileS:
                    radka = unicode(radka.decode(coding_guess(radka)))
                    if zacni == 1:
                        clanekPP += u' ' + (replace_nonsense_characters(radka))
                    if radka == u'' or radka == u' ' or radka == u'\n':
                        zacni = 1
                    if poprve == 0:
                        if not radka.find(u'Subject:') == -1:
                            clanekPP += u' ' + (replace_nonsense_characters(radka))
                            poprve = 1

                fileS.close()


            else:
                clanekPP = u''
                fileS = open(sl + '/' + souboryPom[i], "r")
                for radka in fileS:
                    clanekPP += u' ' + (replace_nonsense_characters(unicode(radka.decode("utf-8-sig"))))
                fileS.close()
                '''
                if len(clanekPP.strip().split()) < 20:
                    pocet += 1
                else:
                '''
                soubory[souboryPom[i]] = sl[poz:len(sl)]
                souboryText[souboryPom[i]] = clanekPP.strip()
        pickle.dump([souboryText, soubory], open('PomocneSoubory/' + kdeHledat + '.p', "wb"))
    else:
        print('Načítání souborů, jejich názvů a originálních shluků složky ' + kdeHledat + ' z již předem připraveného souboru. ')
        souboryText, soubory = pickle.load(open('PomocneSoubory/' + kdeHledat + '.p', "rb"))

    return souboryText, soubory


# odstraňuje interpunkci z textu
def odstran_interpunkci(text):
    #text = re.sub(r'\W+', u' ', text)
    text = text.replace(u'...', u'')
    text = text.replace(u'\u2026', u' ')
    text = text.replace(u"„", u"")
    text = text.replace(u"”", u"")
    text = text.replace(u'"', u'')
    text = text.replace(u'.', u' ')
    text = text.replace(u',', u'')
    text = text.replace(u'!', u'')
    text = text.replace(u'-', u'')
    text = text.replace(u'/', u'')
    text = text.replace(u'{', u'')
    text = text.replace(u'}', u'')
    text = text.replace(u'<', u'')
    text = text.replace(u'>', u'')
    text = text.replace(u':', u'')
    text = text.replace(u'?', u'')
    text = text.replace(u'(', u'')
    text = text.replace(u')', u'')
    text = text.replace(u'[', u'')
    text = text.replace(u']', u'')
    text = text.replace(u'´', u' ')
    text = text.replace(u'$', u' ')
    text = text.replace(u'%', u' ')
    text = text.replace(u'#', u' ')
    text = text.replace(u'@', u' ')
    text = text.replace(u'*', u' ')
    text = text.replace(u'+', u' ')
    text = text.replace(u"'", u'')
    text = text.replace(u"^", u' ')
    text = text.replace(u'   ', u' ')
    text = text.replace(u'  ', u' ')
    text = text.replace(u'roku1980', u'roku 1980')
    return text


def odstranNesmyslyZTextu(text):
    text = text.replace(u'\u010f', u'')
    text = text.replace(u'\u0165', u'')
    text = text.replace(u'\u017c', u'')
    return text


def encode_entities(text):
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')


def vytvorLemmaTagsTokens(textVstup, tagger, jazyk):
    forms = Forms()
    lemmas = TaggedLemmas()
    tokens = TokenRanges()
    tokenizer = tagger.newTokenizer()
    if jazyk == 'english':
        text = odstranNesmyslyZTextu(replace_nonsense_characters(textVstup)).replace(u'_', u' ')
    else:
        text = (replace_nonsense_characters(textVstup)).replace(u'_', u' ')
    # vycistenyText = odstran_interpunkci(text).lower()
    maxDel = 2000
    puvodniDelka = len(text)
    textCasti = []
    if puvodniDelka > maxDel:
        od = 0
        do = 0
        hledej = 0
        for i in range(len(text)):
            if not i == 0:
                if i % maxDel == 0:
                    hledej = 1
                if hledej == 1 or i == len(text) - 1:
                    if text[i] == u'.' or text[i] == u'?' or text[i] == u'!' or i == len(text) - 1:
                        do = i + 1
                        textCasti.append(text[od:do].strip())
                        od = i + 1
                        hledej = 0
        if len(textCasti) == 1 and len(textCasti[0]) > maxDel:
            textCasti = []
            od = 0
            do = 0
            hledej = 0
            for i in range(len(text)):
                if not i == 0:
                    if i % maxDel == 0:
                        hledej = 1
                    if hledej == 1 or i == len(text) - 1:
                        if text[i] == u' ' or i == len(text) - 1:
                            do = i + 1
                            textCasti.append(text[od:do].strip())
                            od = i + 1
                            hledej = 0
    else:
        textCasti.append(text)
    #print len(text), len(textCasti)
    lemmata, tagy, tokeny, lemmataJenSlova = [], [], [], []
    for textC in textCasti:
        tokenizer.setText(textC)
        t = 0
        vysll = u''
        while tokenizer.nextSentence(forms, tokens):
            tagger.tag(forms, lemmas)
            for i in range(len(lemmas)):
                lemma = lemmas[i]
                token = tokens[i]
                lemmata.append(encode_entities(lemma.lemma))
                lemmaCele = lemma.lemma
                if not lemmaCele.find(u'-') == -1:
                    lemmataJenSlova.append(lemmaCele[0:lemmaCele.find(u'-')])
                elif not lemmaCele.find(u'_') == -1:
                    lemmataJenSlova.append(lemmaCele[0:lemmaCele.find(u'_')])
                elif not lemmaCele.find(u'`') == -1:
                    lemmataJenSlova.append(lemmaCele[0:lemmaCele.find(u'`')])
                else:
                    lemmataJenSlova.append(lemmaCele)

                tagy.append(encode_entities(lemma.tag))
                tokeny.append(encode_entities(textC[token.start: token.start + token.length]))
                vysll += ('%s%s<token lemma="%s" tag="%s">%s</token>%s' % (
                    encode_entities(textC[t: token.start]),
                    "<sentence>" if i == 0 else "",
                    encode_entities(lemma.lemma),
                    encode_entities(lemma.tag),
                    encode_entities(textC[token.start: token.start + token.length]),
                    "</sentence>" if i + 1 == len(lemmas) else "",
                ))
                t = token.start + token.length
                vysll += encode_entities(textC[t:])
    '''
    lemjenslovPom = []
    for lemm in odstran_interpunkci(u' '.join(lemmataJenSlova).lower()).split():
        if lemm[0].isdigit():
            lemm = u'digit'
        lemjenslovPom.append(lemm)
    tokenyUprPom = []
    for lemm in odstran_interpunkci(u' '.join(tokeny).lower()).split():
        if lemm[0].isdigit():
            lemm = u'digit'
        tokenyUprPom.append(lemm)

    lemmataUpr = u' '.join(lemmata).lower().split()
    tokenyUpr = tokenyUprPom
    lammataJenSlovaKon = lemjenslovPom
    '''
    lemmataUpr = u' '.join(lemmata).lower().split()
    tokenyUpr = odstran_interpunkci(u' '.join(tokeny).lower()).split()
    lammataJenSlovaKon = odstran_interpunkci(u' '.join(lemmataJenSlova).lower()).split()

    return tokenyUpr, lemmataUpr, tagy, lammataJenSlovaKon

def vytvorLemmaTagsTokens2(textVstup, tagger):
    forms = Forms()
    lemmas = TaggedLemmas()
    tokens = TokenRanges()
    tokenizer = tagger.newTokenizer()
    text = (replace_nonsense_characters(textVstup)).replace(u'_', u' ')
    # vycistenyText = odstran_interpunkci(text).lower()
    maxDel = 2000
    puvodniDelka = len(text)
    textCasti = []
    if puvodniDelka > maxDel:
        od = 0
        do = 0
        hledej = 0
        for i in range(len(text)):
            if not i == 0:
                if i % maxDel == 0:
                    hledej = 1
                if hledej == 1 or i == len(text) - 1:
                    if text[i] == u'.' or text[i] == u'?' or text[i] == u'!' or i == len(text) - 1:
                        do = i + 1
                        textCasti.append(text[od:do].strip())
                        od = i + 1
                        hledej = 0
        if len(textCasti) == 1 and len(textCasti[0]) > maxDel:
            textCasti = []
            od = 0
            do = 0
            hledej = 0
            for i in range(len(text)):
                if not i == 0:
                    if i % maxDel == 0:
                        hledej = 1
                    if hledej == 1 or i == len(text) - 1:
                        if text[i] == u' ' or i == len(text) - 1:
                            do = i + 1
                            textCasti.append(text[od:do].strip())
                            od = i + 1
                            hledej = 0
    else:
        textCasti.append(text)
    #print len(text), len(textCasti)
    lemmata, tagy, tokeny, lemmataJenSlova = [], [], [], []
    for textC in textCasti:
        tokenizer.setText(textC)
        t = 0
        vysll = u''
        while tokenizer.nextSentence(forms, tokens):
            tagger.tag(forms, lemmas)
            for i in range(len(lemmas)):
                lemma = lemmas[i]
                token = tokens[i]
                lemmata.append(encode_entities(lemma.lemma))
                lemmaCele = lemma.lemma
                if not lemmaCele.find(u'-') == -1:
                    lemmataJenSlova.append(lemmaCele[0:lemmaCele.find(u'-')])
                elif not lemmaCele.find(u'_') == -1:
                    lemmataJenSlova.append(lemmaCele[0:lemmaCele.find(u'_')])
                elif not lemmaCele.find(u'`') == -1:
                    lemmataJenSlova.append(lemmaCele[0:lemmaCele.find(u'`')])
                else:
                    lemmataJenSlova.append(lemmaCele)

                tagy.append(encode_entities(lemma.tag))
                tokeny.append(encode_entities(textC[token.start: token.start + token.length]))
                vysll += ('%s%s<token lemma="%s" tag="%s">%s</token>%s' % (
                    encode_entities(textC[t: token.start]),
                    "<sentence>" if i == 0 else "",
                    encode_entities(lemma.lemma),
                    encode_entities(lemma.tag),
                    encode_entities(textC[token.start: token.start + token.length]),
                    "</sentence>" if i + 1 == len(lemmas) else "",
                ))
                t = token.start + token.length
                vysll += encode_entities(textC[t:])

    lemmataUpr = u' '.join(lemmata).lower().split()
    tokenyUpr = odstran_interpunkci(u' '.join(tokeny).lower()).split()
    lammataJenSlovaKon = odstran_interpunkci(u' '.join(lemmataJenSlova).lower()).split()

    return tokenyUpr, lemmataUpr, tagy, lammataJenSlovaKon

def OdstranNepotrebneZnaky2(text):
    text = text.replace(u"'", u'')
    text = text.replace(u'"', u'')
    text = text.replace(u'<', u'')
    text = text.replace(u'>', u'')
    text = text.replace(u'(', u'')
    text = text.replace(u')', u'')
    text = text.replace(u'[', u'')
    text = text.replace(u']', u'')
    text = text.replace(u'{', u'')
    text = text.replace(u'}', u'')
    text = text.replace(u'-', u'')
    text = text.replace(u'.', u'')
    text = text.replace(u'!', u'')
    text = text.replace(u'?', u'')
    text = text.replace(u'+', u'')
    text = text.replace(u';', u'')
    text = text.replace(u'/', u'')
    text = text.replace(u'|', u'')
    text = text.replace(u'`', u'')
    text = text.replace(u'   ', u' ')
    text = text.replace(u'  ', u' ')
    return text

def VycistiText(souboryText, vocabjazyku):
    souboryTextCiste = {}
    for keyy in souboryText:
        text = (souboryText[keyy])
        text = text.replace(u'&', u'and')
        text = re.sub('<.*?>', '', text)
        text = OdstranNepotrebneZnaky2(text)
        textPom = []
        for slovo in text.split():
            if slovo.find(u'@') == -1:
                if slovo[0].isdigit() or slovo[len(slovo)-1].isdigit():
                    textPom.append(u'number')
                else:
                    textPom.append(slovo)
        text = u' '.join(textPom)
        #text = text.replace(u'_', u' ')
        text = text.lower()
        if len(vocabjazyku) > 300000:
            textPom = []
            for slovo in text.split():
                if slovo in vocabjazyku:
                    textPom.append(slovo)
            text = u' '.join(textPom)
        souboryTextCiste[keyy] = text
    return souboryTextCiste


def VycistiText2(souboryText):
    souboryTextCiste = {}
    for keyy in souboryText:
        text = (souboryText[keyy])
        text = text.replace(u'&', u'and')
        text = re.sub('<.*?>', '', text)
        text = OdstranNepotrebneZnaky2(text)
        textPom = []
        for slovo in text.split():
            if slovo.find(u'@') == -1:
                if slovo[0].isdigit():
                    textPom.append(u'number')
                else:
                    textPom.append(slovo)
        text = u' '.join(textPom)
        #text = text.replace(u'_', u' ')
        text = text.lower()
        souboryTextCiste[keyy] = text
    return souboryTextCiste

def UpravAVysictiTexty(souboryTextVstup, vstup, jazyk):
    hlSb = vstup + 'VycisLemmTag.p'
    souboryPS, slozkyPS = ZiskejNazvySouboru('PomocneSoubory/', hlSb)
    if len(souboryPS) == 0:
        print('Probíhá čištění textu, vytváření lemmat a tagů, dále je vše uloženo pro ryhlejší načtení do souboru: ' + hlSb)
        if jazyk == "english":
            tagger = Tagger.load('english-morphium-wsj-140407.tagger')

            vocabjazyku = {}
        else:
            tagger = Tagger.load('czech-morfflex-pdt-161115.tagger')
            vocabjazyku = {}

        print('Počet načtených dat: ' + str(len(souboryTextVstup)))
        souboryText = VycistiText(souboryTextVstup, vocabjazyku)
        pocSpat = 0
        for keyy in souboryText:
            if not len(souboryText[keyy].split()) == len(souboryTextVstup[keyy].split()):
                pocSpat += 1
        vycistene, lemma, tags, lemmaJenSlov = {}, {}, {}, {}
        pocet = 0
        for keyy in souboryText:
            if pocet % 1000 == 0:
                print(pocet)
            vycistene[keyy], lemma[keyy], tags[keyy], lemmaJenSlov[keyy] = vytvorLemmaTagsTokens(souboryText[keyy], tagger, jazyk)
            #vycistene[keyy], lemma[keyy], tags[keyy], lemmaJenSlov[keyy] = vytvorLemmaTagsTokensBezMaloAHodneSeVyskytujicichSlovAKratkychSlov(souboryText[keyy], tagger, jazyk)
            pocet += 1
        pickle.dump([vycistene, lemma, tags, lemmaJenSlov, vocabjazyku], open('PomocneSoubory/' + hlSb, "wb"))
    else:
        print('Načítání vyčištěného textu, lemmat a tagů ze souboru: ' + hlSb)
        vycistene, lemma, tags, lemmaJenSlov, vocabjazyku = pickle.load(open('PomocneSoubory/' + hlSb, "rb"))
    print(len(souboryTextVstup), len(vycistene), len(lemma), len(tags))
    return vycistene, lemmaJenSlov, tags

def UpravAVysictiTexty2(souboryTextVstup):
    tagger = Tagger.load('czech-morfflex-pdt-161115.tagger')
    souboryText = VycistiText2(souboryTextVstup)
    pocSpat = 0
    for keyy in souboryText:
        if not len(souboryText[keyy].split()) == len(souboryTextVstup[keyy].split()):
            pocSpat += 1
    vycistene, lemma, tags, lemmaJenSlov = {}, {}, {}, {}
    pocet = 0
    for keyy in souboryText:
        vycistene[keyy], lemma[keyy], tags[keyy], lemmaJenSlov[keyy] = vytvorLemmaTagsTokens2(souboryText[keyy], tagger)
        #vycistene[keyy], lemma[keyy], tags[keyy], lemmaJenSlov[keyy] = vytvorLemmaTagsTokensBezMaloAHodneSeVyskytujicichSlovAKratkychSlov(souboryText[keyy], tagger, jazyk)
        pocet += 1

    return vycistene, lemmaJenSlov, tags


def VytvorVocab(vstup, pocetFeatures, texty):
    hlSb = vstup + 'Slovnik.p'
    souboryPS, slozkyPS = ZiskejNazvySouboru('PomocneSoubory/', hlSb)

    if len(souboryPS) == 0:
        print('Vytváření slovníku: ' + hlSb)
        termyVse = {}
        clankyVse = {}
        N = len(texty)
        for j in texty:
            termyCl = texty[j]
            termyClDict = {}
            for term in termyCl:
                termyClDict[term] = term
                if term not in termyVse:
                    termyVse[term] = term
            clankyVse[j] = termyClDict
        A = {}
        for key in texty:
            pole = texty[key]
            termyCl = {}
            for slo in pole:
                if slo not in termyCl:
                    termyCl[slo] = slo
                    if slo in A:
                        A[slo] = A[slo] + 1
                    else:
                        A[slo] = 1
        B = {}
        for term in termyVse:
            poc = 0.0
            for cl in clankyVse:
                if term not in clankyVse[cl]:
                    poc += 1.0
            B[term] = poc
        C = {}
        for cl in clankyVse:
            poc = 0.0
            for term in termyVse:
                if term not in clankyVse[cl]:
                    poc += 1.0
            C[cl] = poc
        print('vyber ' + str(pocetFeatures) + ' příznaků z celkového počtu ' + str(len(termyVse)) + ' termů.')
        maxx = [-10000000.0] * pocetFeatures
        termm = [''] * pocetFeatures
        print(len(maxx))
        termy = {}
        for cl in clankyVse:
            termyClanku = clankyVse[cl]
            #for term in termyVse:
            for term in termyClanku:
                if term not in termy and len(term) > 2:
                    hod1 = (A[term] * N)
                    hod2 = A[term] + B[term]
                    hod3 = A[term] + C[cl]
                    MIt = (math.log((hod1) / ((hod3) * (hod2))))
                    minimum = min(maxx)
                    poz = maxx.index(min(maxx))
                    if MIt > minimum:
                        maxx[poz] = MIt
                        termm[poz] = term
                        termy[term] = term
        # print termm
        slovnik = {}
        slovnikPole = []
        for k in range(len(termm)):
            slovnik[termm[k]] = termm[k]
            slovnikPole.append(termm[k])

        # případ extrahování stop slov
        '''
        idf = {}
        stopSlova = {}
        stopSlovaPole = []
        # slovnikPole = []
        for slo in A:
            hod = math.log(float(len(stemClankyTrain)) / A[slo])
            idf[slo] = hod
            if hod < mezStopSlov:
                stopSlova[slo] = slo
                stopSlovaPole.append(slo)
        '''
        pickle.dump([slovnik, slovnikPole], open("PomocneSoubory/" + hlSb, "wb"))
    else:
        print('Načítání slovníku: ' + hlSb)
        slovnik, slovnikPole = pickle.load(open("PomocneSoubory/" + hlSb, "rb"))
    return slovnik, slovnikPole

def VytvorVocabIDF(vstup, pocetFeatures, texty):
    hlSb = vstup + 'Slovnik_' + str(pocetFeatures) + '.p'
    souboryPS, slozkyPS = ZiskejNazvySouboru('PomocneSoubory/', hlSb)

    if len(souboryPS) == 0:
        print('Vytváření slovníku pomocí IDF: ' + hlSb)
        vyskytTermu = {}
        slovaTextu = {}
        idfSlov = {}
        D = len(texty)
        for keyy in texty:
            setSlov = {}
            for word in texty[keyy]:
                if word not in vyskytTermu:
                    vyskytTermu[word] = 0.0
                if word not in setSlov:
                    setSlov[word] = word
            slovaTextu[keyy] = setSlov
        for word in vyskytTermu:
            for keyy in texty:
                if word in slovaTextu:
                    vyskytTermu[word] = vyskytTermu[word] + 1.0
                else:
                    vyskytTermu[word] = 1.0

        print('vyber ' + str(pocetFeatures) + ' příznaků z celkového počtu ' + str(len(vyskytTermu)) + ' termů.')
        maxx = [-10000000.0] * pocetFeatures
        termm = [''] * pocetFeatures
        for word in vyskytTermu:
            idfSlov[word] = math.log(D / vyskytTermu[word])
            minimum = min(maxx)
            poz = maxx.index(min(maxx))
            if idfSlov[word] > minimum:
                maxx[poz] = idfSlov[word]
                termm[poz] = word
        # print termm
        slovnik = {}
        slovnikPole = []
        for k in range(len(termm)):
            slovnik[termm[k]] = termm[k]
            slovnikPole.append(termm[k])

        # případ extrahování stop slov
        '''
        idf = {}
        stopSlova = {}
        stopSlovaPole = []
        # slovnikPole = []
        for slo in A:
            hod = math.log(float(len(stemClankyTrain)) / A[slo])
            idf[slo] = hod
            if hod < mezStopSlov:
                stopSlova[slo] = slo
                stopSlovaPole.append(slo)
        '''
        pickle.dump([slovnik, slovnikPole], open("PomocneSoubory/" + hlSb, "wb"))
    else:
        print('Načítání slovníku IDF: ' + hlSb)
        slovnik, slovnikPole = pickle.load(open("PomocneSoubory/" + hlSb, "rb"))
    return slovnik, slovnikPole


def NactiVstupProUlozeniJenTextuMLF(slozka):
    #případ mlf souborů
    souboryIN, slozkyIN = ZiskejNazvySouboru(slozka + '/', "*in.mlf")
    souboryOUT, slozkyOUT = ZiskejNazvySouboru(slozka + '/', "*out.mlf")

    soubIn = {}
    for soubb in souboryIN:
        fileS = open(slozka + '/' + soubb, 'r')
        text, casyOd, casyDo = [], [], []
        prvni = 0
        for radka in fileS:
            radkaPrac = unicode(radka.decode(coding_guess(radka))).strip()
            kontr = re.findall(u'"*/', radkaPrac)
            if len(kontr) > 0 and prvni == 1:
                soubIn[idd2] = {'od': casyOd, 'do': casyDo, 'text': text}
                text, casyOd, casyDo = [], [], []
                idd2 = u' '.join(re.findall('"*/(.*?)"', radkaPrac))
                idd2 = idd2[0:idd2.find(u'-')]
                idd2 = idd2[idd2.find(u'_') + 1:len(idd2)]
            if len(kontr) > 0 and prvni == 0:
                idd2 = u' '.join(re.findall('"*/(.*?)"', radkaPrac))
                idd2 = idd2[0:idd2.find(u'-')]
                idd2 = idd2[idd2.find(u'_') + 1:len(idd2)]
                prvni = 1
            if not len(radkaPrac) == 0:
                if radkaPrac[0].isdigit():
                    radkaCasti = radkaPrac.split()
                    text.append(radkaCasti[2])
                    casyOd.append(radkaCasti[0])
                    casyDo.append(radkaCasti[1])
        fileS.close()
        soubIn[idd2] = {'od': casyOd, 'do': casyDo, 'text': text}
    soubOut = {}
    for soubb in souboryOUT:
        fileS = open(slozka + '/' + soubb, 'r')
        text, casyOd, casyDo = [], [], []
        prvni = 0
        for radka in fileS:
            radkaPrac = unicode(radka.decode(coding_guess(radka))).strip()
            kontr = re.findall(u'"*/', radkaPrac)
            if len(kontr) > 0 and prvni == 1:
                soubOut[idd2] = {'od': casyOd, 'do': casyDo, 'text': text}
                text, casyOd, casyDo = [], [], []
                idd2 = u' '.join(re.findall('"*/(.*?)"', radkaPrac))
                idd2 = idd2[0:idd2.find(u'-')]
                idd2 = idd2[idd2.find(u'_') + 1:len(idd2)]
            if len(kontr) > 0 and prvni == 0:
                idd2 = u' '.join(re.findall('"*/(.*?)"', radkaPrac))
                idd2 = idd2[0:idd2.find(u'-')]
                idd2 = idd2[idd2.find(u'_') + 1:len(idd2)]
                prvni = 1
            if not len(radkaPrac) == 0:
                if radkaPrac[0].isdigit():
                    radkaCasti = radkaPrac.split()
                    text.append(radkaCasti[2])
                    casyOd.append(radkaCasti[0])
                    casyDo.append(radkaCasti[1])
        fileS.close()
        soubOut[idd2] = {'od': casyOd, 'do': casyDo, 'text': text}

    zakaznik = {}
    operator = {}
    vse = {}
    if len(soubIn.keys()) >= len(soubOut.keys()):
        keeyys = soubIn.keys()
    else:
        keeyys = soubOut.keys()
    for keeyy in keeyys:
        if keeyy in soubIn and keeyy in soubOut:
            odZk = soubIn[keeyy]['od']
            odOp = soubOut[keeyy]['od']
            doZk = soubIn[keeyy]['do']
            doOp = soubOut[keeyy]['do']
            slovaZk = soubIn[keeyy]['text']
            slovaOp = soubOut[keeyy]['text']

            konec, pO, pZ, vseOd, vseDo, vseSl = 0, 0, 0, [], [], []
            while konec == 0:
                if len(odZk) - 1 <= pZ and len(odOp) - 1 <= pO:
                    konec = 1
                if len(odZk) == pZ:
                    if pO <= len(odOp) - 1:
                        vseOd.append(odOp[pO])
                        vseDo.append(doOp[pO])
                        vseSl.append(u'<operator> ' + slovaOp[pO])
                        pO += 1
                elif len(odOp) == pO:
                    if pZ <= len(odZk) - 1:
                        vseOd.append(odZk[pZ])
                        vseDo.append(doZk[pZ])
                        vseSl.append(u'<zakaznik> ' + slovaZk[pZ])
                        pZ += 1
                if len(odOp) - 1 >= pO and len(odZk) - 1 >= pZ:
                    if (odZk[pZ]) > (odOp[pO]):
                        vseOd.append(odOp[pO])
                        vseDo.append(doOp[pO])
                        vseSl.append(u'<operator> ' + slovaOp[pO])
                        pO += 1
                    else:
                        vseOd.append(odZk[pZ])
                        vseDo.append(doZk[pZ])
                        vseSl.append(u'<zakaznik> ' + slovaZk[pZ])
                        pZ += 1
            zakaznik[keeyy] = soubIn[keeyy]['text']
            operator[keeyy] = soubOut[keeyy]['text']
            vse[keeyy] = vseSl
        if keeyy not in soubIn:
            zakaznik[keeyy] = []
            operator[keeyy] = soubOut[keeyy]['text']
            vse[keeyy] = soubOut[keeyy]['text']
        if keeyy not in soubOut:
            zakaznik[keeyy] = soubIn[keeyy]['text']
            operator[keeyy] = []
            vse[keeyy] = soubIn[keeyy]['text']
    return zakaznik, operator, vse

def NactiMLFStereoAuto2(slozkaVstup):
    souboryOperatora1 = {}
    souboryVolajicich = {}
    souboryVLutf8, slozkyVLutf8 = ZiskejNazvySouboru(slozkaVstup + '/', "*-in.mlf")
    souboryOPutf8, slozkyOPutf8 = ZiskejNazvySouboru(slozkaVstup + '/', "*-out.mlf")

    for soubb in souboryVLutf8:
        zakaznik = {u'od': [], u'do':[], u'slova':[], u'cisKon': []} #hodnoty casu od (pole), casu do (pole), slov (pole).
        idd2 = soubb[0:soubb.find(u'-')]
        codingGuess = (chardet.detect(open(slozkaVstup + '/' + soubb, "rb").read()))
        fileS = open(slozkaVstup + '/' + soubb, 'r', encoding=codingGuess['encoding'])
        for radka in fileS:
            radkaPrac = radka.strip()
            if not len(radkaPrac) == 0:
                if radkaPrac[0].isdigit():
                    radkaCasti = radkaPrac.split()
                    odcasS = (float(radkaCasti[0]) / 10000000.0)
                    docasS = (float(radkaCasti[1]) / 10000000.0)
                    zakaznik[u'od'] = zakaznik[u'od'] + [odcasS]
                    zakaznik[u'do'] = zakaznik[u'do'] + [docasS]
                    zakaznik[u'slova'] = zakaznik[u'slova'] + [radkaCasti[2]]
                    zakaznik[u'cisKon'] = zakaznik[u'cisKon'] + [radkaCasti[3]]

        fileS.close()
        souboryVolajicich[idd2] = {u'zakaznik': zakaznik}
    for soubb in souboryOPutf8:
        operator = {u'od': [], u'do':[], u'slova':[], u'cisKon': []} #hodnoty casu od (pole), casu do (pole), slov (pole).
        idd2 = soubb[0:soubb.find(u'-')]
        codingGuess = (chardet.detect(open(slozkaVstup + '/' + soubb, "rb").read()))
        fileS = open(slozkaVstup + '/' + soubb, 'r', encoding=codingGuess['encoding'])
        for radka in fileS:
            radkaPrac = radka.strip()
            if not len(radkaPrac) == 0:
                if radkaPrac[0].isdigit():
                    radkaCasti = radkaPrac.split()
                    odcasS = (float(radkaCasti[0]) / 10000000.0)
                    docasS = (float(radkaCasti[1]) / 10000000.0)
                    operator[u'od'] = operator[u'od'] + [odcasS]
                    operator[u'do'] = operator[u'do'] + [docasS]
                    operator[u'slova'] = operator[u'slova'] + [radkaCasti[2]]
                    operator[u'cisKon'] = operator[u'cisKon'] + [radkaCasti[3]]

        fileS.close()
        souboryOperatora1[idd2] = {u'operator': operator}
    nacteneSoubory2 = {}
    for soubor in souboryVolajicich:
        vse = {u'od': [], u'do': [], u'slova': []}
        operator = {u'od': [], u'do': [], u'slova': []}  # hodnoty casu od (pole), casu do (pole), slov (pole)
        zakaznik = {u'od': [], u'do': [], u'slova': []}  # hodnoty casu od (pole), casu do (pole), slov (pole)
        zk = souboryVolajicich[soubor]
        pokracuj = 1
        if soubor in souboryOperatora1:
            op = souboryOperatora1[soubor]
        else:
            pokracuj = 0
        if pokracuj == 1:
            odZk = zk['zakaznik']['od']
            odOp = op['operator']['od']
            doZk = zk['zakaznik']['do']
            doOp = op['operator']['do']
            slovaZk = zk['zakaznik']['slova']
            slovaOp = op['operator']['slova']
            ckZK = zk['zakaznik']['cisKon']
            ckOp = op['operator']['cisKon']
            konec, pO, pZ, vseOd, vseDo, vseSl, cislaNaKonci = 0, 0, 0, [], [], [], []
            while konec == 0:
                if len(odZk) - 1 <= pZ and len(odOp) -1 <= pO:
                    konec = 1
                if len(odZk) == pZ:
                    if pO <= len(odOp) - 1:
                        vseOd.append(odOp[pO])
                        vseDo.append(doOp[pO])
                        vseSl.append(slovaOp[pO])
                        cislaNaKonci.append(ckOp[pO])
                        pO += 1
                elif len(odOp) == pO:
                    if pZ <= len(odZk) - 1:
                        vseOd.append(odZk[pZ])
                        vseDo.append(doZk[pZ])
                        vseSl.append(slovaZk[pZ])
                        cislaNaKonci.append(ckZK[pZ])
                        pZ += 1
                if len(odOp) -1 >= pO and len(odZk) -1 >= pZ:
                    if (odZk[pZ]) > (odOp[pO]):
                        vseOd.append(odOp[pO])
                        vseDo.append(doOp[pO])
                        vseSl.append(slovaOp[pO])
                        cislaNaKonci.append(ckOp[pO])
                        pO += 1
                    else:
                        vseOd.append(odZk[pZ])
                        vseDo.append(doZk[pZ])
                        vseSl.append(slovaZk[pZ])
                        cislaNaKonci.append(ckZK[pZ])
                        pZ += 1

            vse = {u'od': vseOd, u'do': vseDo, u'slova': vseSl, u'cislaKon': cislaNaKonci}
            if not len(vse[u'slova']) == 0:
                nacteneSoubory2[soubor] = {u'operator': op['operator'], u'zakaznik': zk['zakaznik'], u'vse': vse}

    return nacteneSoubory2