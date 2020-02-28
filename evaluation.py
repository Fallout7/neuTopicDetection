# -*- coding: utf-8 -*-
# coding: utf-8

from predpriprava import *


def evaluateResultsSoubory(slozkaVstup,slozkaVystup):
    coNacist = '*'
    vys, dirrsS = ZiskejNazvySouboru(slozkaVstup, coNacist)
    dirListRef = {}
    for dir in dirrsS:
        if not dirListRef.has_key(dir) and not dir == slozkaVstup:
            dirListRef[dir] = dir
    souboryRef = {}
    for dir in dirListRef:
        dirLis = {}
        for i in range(len(vys)):
            if dir == dirrsS[i]:
                dirLis[vys[i]] =vys[i]
        souboryRef[dir] = dirLis

    coNacist = '*.txt'
    vys, dirrsS = ZiskejNazvySouboru(slozkaVystup, coNacist)
    dirListVys = {}
    for dir in dirrsS:
        if not dirListVys.has_key(dir) and not dir == slozkaVstup:
            dirListVys[dir] = dir
    souboryVys = {}
    for dir in dirListVys:
        dirLis = {}
        for i in range(len(vys)):
            if dir == dirrsS[i]:
                dirLis[vys[i]] = vys[i]
        souboryVys[dir] = dirLis

    if len(dirListRef) > len(dirListVys):
        for ij in range(len(dirListRef)):
            ano = 0
            for dirrr in dirListVys:
                je = 0
                for ijij in range(len(dirrr)):
                    if dirrr[ijij] == '/':
                        je = 1
                        od = ijij + 1
                if je == 1:
                    dirrCis = int(dirrr[od:len(dirrr)])
                else:
                    dirrCis = int(dirrr)
                if ij == dirrCis:
                    ano = 1
            if ano == 0:
                dirListVys[str(ij)] = {}
                souboryVys[str(ij)] = {}

    #T bude množina složená z t clusterů, jsou to výstupní clustery, zatímco C je referenční množina c clusterů.
    alfa,beta,gama = [],[],[]
    pocShodSeShluky = {}
    for refc in dirListRef:
        souboryReff = souboryRef[refc]
        shluky = {}
        for vysc in dirListVys:
            souboryVyss = souboryVys[vysc]
            nAlfa = 0.0
            for soubor in souboryReff:
                if souboryVyss.has_key(soubor):
                    nAlfa += 1.0
            shluky[vysc] = nAlfa
        pocShodSeShluky[refc] = shluky

    dirListVysLabel = {}
    vysledShluky = {}
    zarazeneRef = {}
    zarazeneShl = {}
    for refc in dirListRef:
        maxHod = 0.0
        maxRef = ''
        maxShl = ''
        for refcc in dirListRef:
            if not zarazeneRef.has_key(refcc):
                shlukyCetnost = pocShodSeShluky[refcc]
                for shluk in shlukyCetnost:
                    if not zarazeneShl.has_key(shluk):
                        shodne = shlukyCetnost[shluk]
                        if maxHod < shodne+1.0:
                            maxHod = shodne
                            maxRef = refcc
                            maxShl = shluk

        dirListVysLabel[maxRef] = souboryVys[maxShl]
        vysledShluky[maxRef] = maxShl
        zarazeneRef[maxRef] = maxRef
        zarazeneShl[maxShl] = maxShl

    for refc in dirListRef:
        souboryReff = souboryRef[refc]
        souboryVyss = dirListVysLabel[refc]
        nAlfa = 0.0
        nBeta = 0.0
        for soub in souboryReff:
            if souboryVyss.has_key(soub):
                nAlfa += 1.0
            else:
                nBeta += 1.0

        alfa.append(nAlfa)
        beta.append(nBeta)
        nGamma = 0.0
        for refc2 in dirListRef:
            if not refc == refc2:
                souboryVyss = dirListVysLabel[refc2]
                odpovidajiciVyslednyShluk = vysledShluky[refc2]
                for soub in souboryReff:
                    if souboryVyss.has_key(soub):

                        fileS = file(refc + '/' + soub, "r")
                        dokumentRawRadky = u''
                        for radka in fileS:
                            codingg = coding_guess(radka)
                            r = replace_nonsense_characters(unicode(radka.decode(codingg)))
                            dokumentRawRadky += u'' + (r)
                        fileS.close()

                        fileS = file(odpovidajiciVyslednyShluk + '/' + soub, "r")
                        dokumentRawRadky2 = u''
                        for radka in fileS:
                            codingg = coding_guess(radka)
                            r = replace_nonsense_characters(unicode(radka.decode(codingg)))
                            dokumentRawRadky2 += u'' + (r)
                        fileS.close()
                        dokumentRawRadky = odstran_interpunkci(dokumentRawRadky)
                        dokumentRawRadky2 = odstran_interpunkci(dokumentRawRadky2)

                        if dokumentRawRadky == dokumentRawRadky2:
                            nGamma += 1.0
        gama.append(nGamma)
        #print nAlfa

    jmenP = 0.0
    jmenR = 0.0

    for ij in range(len(alfa)):
        jmenP += alfa[ij]+beta[ij]
        jmenR += alfa[ij]+gama[ij]
    P = (sum(alfa)/jmenP)*100.0
    R = (sum(alfa)/jmenR)*100.0
    return P,R

def evaluateResultsAcc(souboryVstup, vysledky, vysledkyNazvySoub):
    poc = 0.0
    for i in range(len(vysledky)):
        vys = vysledky[i]
        if vys == souboryVstup[vysledkyNazvySoub[i]]:
            poc += 1.0
    return (poc / (len(vysledky) / 100.0))

def evaluateResultsAcc2(souboryVstup, vysledky, vysledkyNazvySoub):
    slozkyNazvy = {}
    for soubb in souboryVstup:
        if not slozkyNazvy.has_key(souboryVstup[soubb]):
            slozkyNazvy[souboryVstup[soubb]] = [soubb]
        else:
            slozkyNazvy[souboryVstup[soubb]] = slozkyNazvy[souboryVstup[soubb]] + [soubb]

    slozkyVysNazvy = {}
    for ii in range(len(vysledkyNazvySoub)):
        if not slozkyVysNazvy.has_key(vysledky[ii]):
            slozkyVysNazvy[vysledky[ii]] = [vysledkyNazvySoub[ii]]
        else:
            slozkyVysNazvy[vysledky[ii]] = slozkyVysNazvy[vysledky[ii]] + [vysledkyNazvySoub[ii]]
    if not len(slozkyNazvy) == len(slozkyVysNazvy):
        for i in range(len(slozkyNazvy)):
            if not slozkyVysNazvy.has_key(i):
                slozkyVysNazvy[i] = []

    odpovidajicShl = {}
    pocetStejnychVShl = {}
    pouzOrig = {}
    pouzVys = {}
    for i in range(len(slozkyNazvy)):
        maxxStn = 0.0
        odpShlOrig = ""
        odpShlVys = ""
        for shlOrig in slozkyNazvy:
            soubOrig = slozkyNazvy[shlOrig]
            for shlVys in slozkyVysNazvy:
                soubVys = slozkyVysNazvy[shlVys]
                poccStn = 0.0
                for souborVys in soubVys:
                    for souborOrig in soubOrig:
                        if souborVys == souborOrig:
                            poccStn += 1.0
                if maxxStn < poccStn:
                    if not pouzOrig.has_key(shlOrig) and not pouzVys.has_key(shlVys):
                        maxxStn = poccStn
                        odpShlOrig = shlOrig
                        odpShlVys = shlVys
        pouzVys[odpShlVys] = odpShlVys
        pouzOrig[odpShlOrig] = odpShlOrig
        odpovidajicShl[odpShlVys] = odpShlOrig
        pocetStejnychVShl[odpShlVys] = maxxStn

    pocetStajnych = 0.0
    for shlOrig in slozkyNazvy:
        for shlVys in odpovidajicShl:
            if shlOrig == odpovidajicShl[shlVys]:
                pocetStajnych += pocetStejnychVShl[shlVys]

    return (pocetStajnych / (len(vysledky) / 100.0))

def evaluateResults(souboryVstup, vysledky, vysledkyNazvySoub):

    vstupShluky = {}
    for key in souboryVstup:
        if not vstupShluky.has_key(souboryVstup[key]):
            vstupShluky[souboryVstup[key]] = {key: key}
        else:
            dictPom = {}
            for keyy in vstupShluky[souboryVstup[key]]:
                dictPom[keyy] = keyy
            dictPom[key] = key
            vstupShluky[souboryVstup[key]] = dictPom

    vysledkyShluky = {}
    for i in range(len(vysledky)):
        if not vysledkyShluky.has_key(vysledky[i]):
            vysledkyShluky[vysledky[i]] = {vysledkyNazvySoub[i]: vysledkyNazvySoub[i]}
        else:
            dictPom = {}
            for key in vysledkyShluky[vysledky[i]]:
                dictPom[key] = key
            dictPom[vysledkyNazvySoub[i]] = vysledkyNazvySoub[i]
            vysledkyShluky[vysledky[i]] = dictPom


    '''
    if len(souboryRef) > len(souboryVys):
        for ij in range(len(dirListRef)):
            ano = 0
            for dirrr in dirListVys:
                je = 0
                for ijij in range(len(dirrr)):
                    if dirrr[ijij] == '/':
                        je = 1
                        od = ijij + 1
                if je == 1:
                    dirrCis = int(dirrr[od:len(dirrr)])
                else:
                    dirrCis = int(dirrr)
                if ij == dirrCis:
                    ano = 1
            if ano == 0:
                dirListVys[str(ij)] = {}
                souboryVys[str(ij)] = {}
    '''
    dirListRef = vstupShluky
    dirListVys = vysledkyShluky
    #T bude množina složená z t clusterů, jsou to výstupní clustery, zatímco C je referenční množina c clusterů.
    alfa,beta,gama = [],[],[]
    pocShodSeShluky = {}
    for refc in dirListRef:
        souboryReff = dirListRef[refc]
        shluky = {}
        for vysc in dirListVys:
            souboryVyss = dirListVys[vysc]
            nAlfa = 0.0
            for soubor in souboryReff:
                if souboryVyss.has_key(soubor):
                    nAlfa += 1.0
            shluky[vysc] = nAlfa
        pocShodSeShluky[refc] = shluky

    dirListVysLabel = {}
    vysledShluky = {}
    zarazeneRef = {}
    zarazeneShl = {}
    for refc in dirListRef:
        maxHod = 0.0
        maxRef = ''
        maxShl = ''
        for refcc in dirListRef:
            if not zarazeneRef.has_key(refcc):
                shlukyCetnost = pocShodSeShluky[refcc]
                for shluk in shlukyCetnost:
                    if not zarazeneShl.has_key(shluk):
                        shodne = shlukyCetnost[shluk]
                        if maxHod < shodne+1.0:
                            maxHod = shodne
                            maxRef = refcc
                            maxShl = shluk

        if maxRef == '':
            for refcc in dirListRef:
                if not zarazeneRef.has_key(refcc):
                    maxRef = refcc
                    for shluk in shlukyCetnost:
                        if not zarazeneShl.has_key(shluk):
                            maxShl = shluk
            if maxShl == '':
                hodMin = 0.0
                for shluk in shlukyCetnost:
                    if shlukyCetnost[shluk] > hodMin:
                        hodMin = shlukyCetnost[shluk]
                        maxShl = shluk

            print(maxRef, maxShl)
            dirListVysLabel[maxRef] = dirListVys[maxShl]
            vysledShluky[maxRef] = maxShl
        else:
            dirListVysLabel[maxRef] = dirListVys[maxShl]
            vysledShluky[maxRef] = maxShl
            zarazeneRef[maxRef] = maxRef
            zarazeneShl[maxShl] = maxShl

    for refc in dirListRef:
        souboryReff = dirListRef[refc]
        if dirListVysLabel.has_key(refc):
            souboryVyss = dirListVysLabel[refc]
            nAlfa = 0.0
            nBeta = 0.0
            for soub in souboryReff:
                if souboryVyss.has_key(soub):
                    nAlfa += 1.0
                else:
                    nBeta += 1.0

            alfa.append(nAlfa)
            beta.append(nBeta)
            nGamma = 0.0
            for refc2 in dirListRef:
                if not refc == refc2:
                    if dirListVysLabel.has_key(refc2):
                        souboryVyss = dirListVysLabel[refc2]
                        odpovidajiciVyslednyShluk = vysledShluky[refc2]
                        for soub in souboryReff:
                            if souboryVyss.has_key(soub):
                                '''
                                fileS = file(refc + '/' + soub, "r")
                                dokumentRawRadky = u''
                                for radka in fileS:
                                    codingg = coding_guess(radka)
                                    r = replace_nonsense_characters(unicode(radka.decode(codingg)))
                                    dokumentRawRadky += u'' + (r)
                                fileS.close()
        
                                fileS = file(odpovidajiciVyslednyShluk + '/' + soub, "r")
                                dokumentRawRadky2 = u''
                                for radka in fileS:
                                    codingg = coding_guess(radka)
                                    r = replace_nonsense_characters(unicode(radka.decode(codingg)))
                                    dokumentRawRadky2 += u'' + (r)
                                fileS.close()
                                dokumentRawRadky = odstran_interpunkci(dokumentRawRadky)
                                dokumentRawRadky2 = odstran_interpunkci(dokumentRawRadky2)
        
                                if dokumentRawRadky == dokumentRawRadky2:
                                    nGamma += 1.0
                                '''
                                nGamma += 1.0
            gama.append(nGamma)
        #print nAlfa

    jmenP = 0.0
    jmenR = 0.0

    for ij in range(len(alfa)):
        jmenP += alfa[ij]+beta[ij]
        jmenR += alfa[ij]+gama[ij]
    P = (sum(alfa)/jmenP)*100.0
    R = (sum(alfa)/jmenR)*100.0
    return P,R


def evaluateResultsCRVAL(souboryVstup, vysledkyVs, vysledkyNazvySoub):

    vysledky = {}
    for i in range(len(vysledkyNazvySoub)):
        vysledky[vysledkyNazvySoub[i]] = vysledkyVs[i]
    refData = {}
    for vs in vysledky:
        refData[vs] = souboryVstup[vs]

    shlukyPouziteACetnosti = {}
    for vys in vysledky:
        shluk = vysledky[vys]
        if shlukyPouziteACetnosti.has_key(shluk):
            shlukyPouziteACetnosti[shluk] = shlukyPouziteACetnosti[shluk] + 1.0
        else:
            shlukyPouziteACetnosti[shluk] = 1.0

    shlukyVstup = {}
    for soub in souboryVstup:
        if not shlukyVstup.has_key(souboryVstup[soub]):
            shlukyVstup[souboryVstup[soub]] = souboryVstup[soub]

    print(len(shlukyVstup), len(shlukyPouziteACetnosti))


    vstupShluky = {}
    for key in souboryVstup:
        if not vstupShluky.has_key(souboryVstup[key]):
            vstupShluky[souboryVstup[key]] = {key: key}
        else:
            dictPom = {}
            for keyy in vstupShluky[souboryVstup[key]]:
                dictPom[keyy] = keyy
            dictPom[key] = key
            vstupShluky[souboryVstup[key]] = dictPom

    vysledkyShluky = {}
    for i in range(len(vysledky)):
        if not vysledkyShluky.has_key(vysledky[i]):
            vysledkyShluky[vysledky[i]] = {vysledkyNazvySoub[i]: vysledkyNazvySoub[i]}
        else:
            dictPom = {}
            for key in vysledkyShluky[vysledky[i]]:
                dictPom[key] = key
            dictPom[vysledkyNazvySoub[i]] = vysledkyNazvySoub[i]
            vysledkyShluky[vysledky[i]] = dictPom


    dirListRef = vstupShluky
    dirListVys = vysledkyShluky
    #T bude množina složená z t clusterů, jsou to výstupní clustery, zatímco C je referenční množina c clusterů.
    alfa,beta,gama = [],[],[]
    pocShodSeShluky = {}
    for refc in dirListRef:
        souboryReff = dirListRef[refc]
        shluky = {}
        for vysc in dirListVys:
            souboryVyss = dirListVys[vysc]
            nAlfa = 0.0
            for soubor in souboryReff:
                if souboryVyss.has_key(soubor):
                    nAlfa += 1.0
            shluky[vysc] = nAlfa
        pocShodSeShluky[refc] = shluky

    dirListVysLabel = {}
    vysledShluky = {}
    zarazeneRef = {}
    zarazeneShl = {}
    pouziteSh = {}
    pouzeteRef = {}
    for refc in dirListRef:
        maxHod = 0.0
        maxRef = ''
        maxShl = ''
        for refcc in dirListRef:
            if not zarazeneRef.has_key(refcc):
                shlukyCetnost = pocShodSeShluky[refcc]
                for shluk in shlukyCetnost:
                    if not zarazeneShl.has_key(shluk):
                        shodne = shlukyCetnost[shluk]
                        if maxHod < shodne+1.0:
                            maxHod = shodne
                            maxRef = refcc
                            maxShl = shluk

        if not maxShl == u'':
            dirListVysLabel[maxRef] = dirListVys[maxShl]
            vysledShluky[maxRef] = maxShl
            zarazeneRef[maxRef] = maxRef
            zarazeneShl[maxShl] = maxShl
            if not pouziteSh.has_key(maxShl):
                pouziteSh[maxShl] = maxShl
            if not pouzeteRef.has_key(maxRef):
                pouzeteRef[maxRef] = maxRef

    for refc in dirListRef:
        for refcc in dirListRef:
            shlukyCetnost = pocShodSeShluky[refcc]
            for shluk in shlukyCetnost:
                if not pouzeteRef.has_key(refcc) and not pouziteSh.has_key(shluk):
                    maxRefNAHRADNI = refcc
                    maxShlNAHRADNI = shluk
        dirListVysLabel[maxRefNAHRADNI] = dirListVys[maxShlNAHRADNI]
        vysledShluky[maxRefNAHRADNI] = maxShlNAHRADNI
        zarazeneRef[maxRefNAHRADNI] = maxRefNAHRADNI
        zarazeneShl[maxShlNAHRADNI] = maxShlNAHRADNI


    for refc in dirListRef:
        souboryReff = dirListRef[refc]
        souboryVyss = dirListVysLabel[refc]
        nAlfa = 0.0
        nBeta = 0.0
        for soub in souboryReff:
            if souboryVyss.has_key(soub):
                nAlfa += 1.0
            else:
                nBeta += 1.0

        alfa.append(nAlfa)
        beta.append(nBeta)
        nGamma = 0.0
        for refc2 in dirListRef:
            if not refc == refc2:
                souboryVyss = dirListVysLabel[refc2]
                odpovidajiciVyslednyShluk = vysledShluky[refc2]
                for soub in souboryReff:
                    if souboryVyss.has_key(soub):

                        nGamma += 1.0
        gama.append(nGamma)
        #print nAlfa

    jmenP = 0.0
    jmenR = 0.0

    for ij in range(len(alfa)):
        jmenP += alfa[ij]+beta[ij]
        jmenR += alfa[ij]+gama[ij]
    P = (sum(alfa)/jmenP)*100.0
    R = (sum(alfa)/jmenR)*100.0
    return P,R