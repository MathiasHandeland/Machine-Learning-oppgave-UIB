import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# les inn data fra csv filen 
data = pd.read_csv('party_data.csv', sep=',')

# K-nærmeste nabo algoritme
def knn():
    # get_dummies: konverterer kategoriske variabler til dummy/indikatorvariabler (0 eller 1). Kalles også one-hot encoding.
    # argumentet data er dataene vi skal konvertere, columns er kolonnene vi skal konvertere, drop_first=False vil si at vi ikke dropper en av kolonnene
    # One-hot encoding må gjøres fordi K-nearest neighbour algoritmen ikke kan håndtere kategoriske variabler.
    data_knn = pd.get_dummies(data, columns=['gender', 'age', 'study', 'activity', 'music', 'is dancer'])
    
    features = data_knn.drop('ok guest', axis=1) # features representerer alle kolonner uten 'ok guest', altså alle kolonner som skal brukes til å predikere
    labels = data_knn['ok guest'] # labels representerer kolonnen vi skal predikere, altså om en person er en ok gjest eller ikke

    # Deler data i trenings- og testsett. bruker 80% av data til trening og 20% til testing. 
    # Argumentet features er dataene vi skal dele, labels er kolonnen vi skal predikere. 
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Bruker OrdinalEncoder for å gå gjennom hele datasettet og erstatte alle verdier med 0 eller 1.
    encoder = OrdinalEncoder()
    features_train = encoder.fit_transform(features_train) # transformerer treningsdata til tallverdier
    features_test = encoder.transform(features_test) # transformerer testdata til tallverdier

    # Nå kan vi bruke K-nearest neighbour-klasse til læringa: 
    results = []
    k_values = [3,5,11,17] # Vi tester med ulike verdier for k (naboer)
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k) 
        knn.fit(features_train, labels_train)  # fit modellen

        # Predikere trenings- og testdata
        train_predictions = knn.predict(features_train) # predikerer treningsdata. 
        test_predictions = knn.predict(features_test) # predikerer testdata.

        train_accuracy = accuracy_score(labels_train, train_predictions) # bruker korrekthet som mål for å evaluere modellen
        test_accuracy = accuracy_score(labels_test, test_predictions) # bruker korrekthet som mål for å evaluere modellen
        cm = confusion_matrix(labels_test, test_predictions) # Forvekslingsmatrise (viser antall True Positives, True Negatives, False Positives og False Negatives)
        precision = precision_score(labels_test, test_predictions, pos_label='ok') # Presisjon: andelen av de som er klassifisert som "ok guest" som faktisk er "ok guest"
        
        results.append({
            'Algorithm': f'KNN (k={k})',
            'Train Accuracy': train_accuracy,
            'Test Accuracy': test_accuracy,
            'Precision': precision,
            'Confusion Matrix': cm
        })
        
    return results # returnerer resultatene fra KNN algoritmen slik at vi kan bruke de i show_results metoden


# Logistisk regresjon algoritme
def logistic_regression():

    data_lr = pd.get_dummies(data, columns=['gender', 'age', 'study', 'activity', 'music', 'is dancer'], drop_first=True) # drop_first=True dropper en av kolonnene, unngår multikollinearitet, som er når to eller flere prediktorer er sterkt korrelerte.
    features = data_lr.drop('ok guest', axis=1) # features representerer alle kolonner uten 'ok guest', altså alle kolonner som skal brukes til å predikere
    labels = data_lr['ok guest'] # labels representerer kolonnen vi skal predikere, altså om en person er en ok gjest eller ikke
    
    # Deler data i trenings- og testsett. 
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Bruker OrdinalEncoder for å gå gjennom hele datasettet og erstatte alle verdier med 0 eller 1.
    encoder = OrdinalEncoder()
    features_train = encoder.fit_transform(features_train)
    features_test = encoder.transform(features_test)
    
    # Nå kan vi bruke logistisk regresjons-klasse til læringa:
    results = []
    penalty = ['l2', None] # Vi tester med ulike verdier for penalty, penalty er en straff for å ha for mange parametere i modellen
    
    for p in penalty:
        
        lr = LogisticRegression(penalty=p)
        lr.fit(features_train, labels_train)
        
        train_predictions = lr.predict(features_train)
        test_predictions = lr.predict(features_test)

        train_accuracy = accuracy_score(labels_train, train_predictions)
        test_accuracy = accuracy_score(labels_test, test_predictions)
        cm = confusion_matrix(labels_test, test_predictions) # Forvekslingsmatrise
        precision = precision_score(labels_test, test_predictions, pos_label='ok') # Presisjon: andelen av de som er klassifisert som "ok guest" som faktisk er "ok guest"
    
        results.append({
            'Algorithm': f'Logistic Regression (penalty={p})',
            'Train Accuracy': train_accuracy,
            'Test Accuracy': test_accuracy,
            'Precision': precision,
            'Confusion Matrix': cm
        })
    
    return results # returnerer resultatene fra logistisk regresjons algoritmen slik at vi kan bruke de i show_results metoden
    

def decision_tree():
    
    data_dt = pd.get_dummies(data, columns=['gender', 'age', 'study', 'activity', 'music', 'is dancer'])
    features = data_dt.drop('ok guest', axis=1) # features representerer alle kolonner uten 'ok guest', altså alle kolonner som skal brukes til å predikere
    labels = data_dt['ok guest'] # labels representerer kolonnen vi skal predikere, altså om en person er en ok gjest eller ikke
    
    # Deler data i trenings- og testsett.
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # Ordinal encoding for kategorier. Det vil si at vi gir hver kategori en tallverdi.
    encoder = OrdinalEncoder()
    features_train = encoder.fit_transform(features_train)
    features_test = encoder.transform(features_test)
    
    # Nå kan vi bruke Decision Tree-klasse til læringa:
    results = []
    criterion = ['gini', 'entropy'] # Vi tester med ulike verdier for criterion
    
    for c in criterion:
        dt = DecisionTreeClassifier(criterion=c)
        dt.fit(features_train, labels_train)
        
        train_predictions = dt.predict(features_train)
        test_predictions = dt.predict(features_test)

        train_accuracy = accuracy_score(labels_train, train_predictions)
        test_accuracy = accuracy_score(labels_test, test_predictions) 
        cm = confusion_matrix(labels_test, test_predictions) # Forvekslingsmatrise
        precision = precision_score(labels_test, test_predictions, pos_label='ok') # pos_label='ok' spesifiserer hvilken klasse som er positiv.
        # Presisjon: andelen av de som er klassifisert som "ok guest" som faktisk er "ok guest"

        results.append({
            'Algorithm': f'Decision Tree (criterion={c})',
            'Train Accuracy': train_accuracy,
            'Test Accuracy': test_accuracy,
            'Precision': precision,
            'Confusion Matrix': cm
        })
    
    return results # returnerer resultatene fra Decision Tree algoritmen slik at vi kan bruke de i show_results metoden
        

def show_results():
    # Hent resultater fra de tre algoritmene 
    knn_results = knn()
    logistic_results = logistic_regression()
    decision_tree_results = decision_tree()

    # Slå sammen resultatene fra de tre algoritmene til en liste 
    all_results = knn_results + logistic_results + decision_tree_results

    # Lag en DataFrame av resultatene (en tabell med resultatene)
    results_df = pd.DataFrame(all_results)

    # Print ut resultatene
    print("\nSummary of Model Results:")
    print(results_df)
    print("\nThe best model with the highest precision is:")  
    print("High precision indicates that the model is good at predicting 'ok' guests with few false positives.")
    print(results_df.loc[results_df['Precision'].idxmax()])  # Get the row with the highest precision 

if __name__ == '__main__':
    show_results()