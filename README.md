# INFO 180 – Introduksjon til kunstig intelligens

## Oblig-oppgåve 2 – Maskinlæring - OPPGAVEBESKRIVELSE

Maskinlæring er eit stort felt innan kunstig intelligens, og er eit felt som har massevis av ressursar på veven. Det fins òg programpakkar som har ferdig implementerte versjonar av maskinlæringsalgoritmar. Algoritmane er altså ferdig implementert, ein treng berre å tilpasse data, ein del viktige læringsparametrar og køyre algoritmane og evaluere dei.

I denne oppgåva skal de jobbe med eit datasett med eigenskapar for studentar ved UiB og i kor stor grad dei vil vere ok å ha med på fest. Datasettet er tilgjengeleg på Mitt UiB. Ein kan tenke seg at dataa er samla inn av organisasjonen NoPartyKillers som registrerer på veven erfaringar folk har hatt med ulike festdeltakarar (sjølvsagt heilt ulovleg, men … ). De skal eigentleg sjekke kor gode tre maskinlærings-algoritmar er på dette datasettet. Oppgåva vert denne gongen i stor grad å finne ut av maskinlærings-verktøy og -metodar ved å bruke ressursar på veven.

### Datasettet har følgande kolonnar:
- Gender: male eller female
- Age: < 20, 20-24, 24-30, > 30
- Study: socsci, mathsci, med, hum – kva fakultet studenten kjem frå
- Activity: favorittfritidsaktivitet blant 5 mulige: outdoor life, gaming, sport, music, cooking.
- Music: musikkpreferanse blant 6 mulige: rock, soul/rb, hiphop, jazz, classic, folk
- Is dancer: dancer eller not dancer. Om personen dansar på festar
- Ok guest: ok eller not ok - om gjesten fungerte ok på festen. Det er denne de skal predikere

Dei to algoritmane «k-næraste naboar» og «logistisk regresjon» er gjennomgått på forelesinga. De skal og prøve ut algoritmen «avgjerdstre» (decision tree).

### Steg-for-steg:

1. **Installering av nødvendige pakkar**  
   Før du kan gjere noko må du installere dei rette programpakkane i dykkar python-oppsett. Du treng i alle fall pandas, numpy, og sklearn. Innlesing av data kan for eksempel gjerast med pandas sin `read_csv()`-metode. 

   Nyttige ressursar finn du her:  
   https://machinelearningmastery.com/machine-learning-in-python-step-by-step/

2. **Preprosessering av data**  
   Alle data her er kategoriske, så til kvar av algoritmane må det gjerast nokre førebuande steg (preprosessering) på datasettet:

    - **K-næraste nabo:**  
      Algoritmane i sklearn kan ikkje ta tekstverdiar, så du må bruke indikatorvariable (one-hot-encoding) for kvar kategori med `get_dummies()` i pandas. Etter dette må du erstatte alle verdiar i datasettet med 0 eller 1, for eksempel ved hjelp av `OrdinalEncoder` i sklearn. Bruk sklearn sin K-nearest neighbour-klasse:  
      https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html  
      Prøv versjonar der k = 3, 5, 11 eller 17.

    - **Logistisk regresjon:**  
      For logistisk regresjon må du passe på å unngå multikollinearitet ved å bruke `drop_first=True` i `get_dummies()`. Dette fjernar ein av indikatorvariablane. Bruk `LogisticRegression` frå sklearn:  
      https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html  
      Prøv også å justere `penalty`-parameteren (eks: `penalty=None`).

    - **Avgjerdstre:**  
      Bruk same datasettversjon som for k-næraste nabo. Du skal prøve både `gini` og `entropy` som kriterium for trebygging i `DecisionTreeClassifier`:  
      https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

3. **Datasplitting:**  
   Før du køyrer maskinlæringsalgoritmane må du dele det preprosesserte datasettet i eit treningssett (80%) og eit testsett (20%) med `train_test_split`:  
   https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

4. **Evaluering av modellane:**  
   For kvar algoritme (4 for k-næraste nabo, 2 for logistisk regresjon, 2 for avgjerdstre) skal du måle korrektheit (accuracy) og vise forvekslingsmatrise på testsettet. Print også forskjellen i korrektheit mellom trenings- og testsett for å sjekke for overtilpassing. 

   Skriv ut presisjon for klassifisering av gjester som ‘ok’.

   Sjå meir informasjon i seksjon 6 her:  
   https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
