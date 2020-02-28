runTest.sh  -- obsahuje jednoduchý skript, jen spustí a když vyjde výsledek stejně jako porovnávaný soubor vypíše se že soubory jsou identické

main.py 	-- hlavní soubor s argumenty:
					-clp   	--string s cestou ke vstupním souborům, tedy k souborům které chci klasifikovat (defaultně na 'Vstup/')
					-nt 	--počet témat do kolika má být nahrávka přiřazena (defaultně nastaveno na 3)
					-o 		--název výstupního souboru (defaultně na 'results')
			-- 	výstupem je jak soubor (defaultně results).npy s dictionary kde keys jsou id nahrávek a obsahuje pak další dictionary, 					kde keys jsou témata a pod tím jsou pravděpodobnosti s jakou do daných témat patří. 

				Také je další výstupní soubor (defaultně results).txt kde jsou výsledky vypsány. 

mainNew.py 	-- hlavní soubor s argumenty:
					-clp   	--string s cestou ke vstupním souborům, tedy k souborům které chci klasifikovat (defaultně na 'Vstup/')
					-o 		--název výstupního souboru (defaultně na 'results')
			-- 	výstupem je jak soubor (defaultně results).npy s dictionary kde keys jsou id nahrávek a obsahuje pak další dictionary, 					kde keys jsou témata a pod tím jsou pravděpodobnosti s jakou do daných témat patří. 

				Také je další výstupní soubor (defaultně results).txt kde jsou výsledky vypsány. 
				Počet témat do kterých je daná nahrávka zařazena je určen podle získaného prahu z natrénované neuronové sítě na validačních datech. 

Zjistil jsem, že používám balík z ufal na taggování a vytváření lemmat. Nevím jestli se dá stáhnout přes repositáře, ale popřípadě by se měl dát sehnat tady: http://ufal.mff.cuni.cz/morphodita

		
