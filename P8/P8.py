import pandas as pd  #Manipulación de datos
from sklearn.model_selection import train_test_split  #Dividir el conjunto de datos en entrenamiento y prueba
from sklearn.naive_bayes import GaussianNB  #Naive Bayes
from sklearn.linear_model import LogisticRegression  #Regresión Logística
from sklearn.neighbors import KNeighborsClassifier  #K-Vecinos más cercanos
from sklearn.metrics import accuracy_score, confusion_matrix  #Evaluar la precisión del modelo y crear la matriz de confusión

#Definimos función para entrenar y evaluar un clasificador
def entrenar_y_evaluar(clf, X_entrenamiento, y_entrenamiento, X_prueba, y_prueba, caracteristica):
    clf.fit(X_entrenamiento, y_entrenamiento)
    y_prediccion = clf.predict(X_prueba) #Predecimos etiquetas para los datos de prueba
    precision = accuracy_score(y_prueba, y_prediccion)
    patrones_correctos = sum(y_prediccion == y_prueba)
    matriz_confusion = confusion_matrix(y_prueba, y_prediccion)
    print(f"\n● Accuracy ({caracteristica}): {precision}")
    print(f"● Número de patrones clasificados correctamente ({caracteristica}): {patrones_correctos}")
    print(f"● Matriz de confusión ({caracteristica}):\n{matriz_confusion}\n\n")

#Cargamos el conjunto de datos de entrenamiento
df_entrenamiento = pd.read_csv('train.csv')

#Dividimos los datos de entrenamiento en características (X) y etiquetas (y)
X_entrenamiento = df_entrenamiento.iloc[:, :-1]  #Todas las columnas
y_entrenamiento = df_entrenamiento.iloc[:, -1]  #La última columna (clase)

#Cargamos el conjunto de datos de prueba
df_prueba = pd.read_csv('test.csv')

#Dividimos los datos de prueba en características (X) y etiquetas (y)
X_prueba = df_prueba.iloc[:, :-1]
y_prueba = df_prueba.iloc[:, -1]

print("\n----------------Práctica 8 | Entrenamiento De Clasificador----------------\n")
print("Seleccione el algoritmo de entrenamiento:\n")
print("1. Naive Bayes\n2. Regresión Logística\n3. K-Vecinos más cercanos (KNN)\n4. Salir\n\n")
eleccion = input("✎ Ingrese el número correspondiente: ")

if eleccion == "1":
    print("\n》Naive Bayes\n")
    #Configuramos clasificadores
    clf_petallength = GaussianNB()
    clf_petalwidth = GaussianNB()

    #Carcaterísticas de entrenamiento
    X_entrenamiento_petallength = X_entrenamiento.iloc[:, 0].values.reshape(-1, 1)
    X_entrenamiento_petalwidth = X_entrenamiento.iloc[:, 1].values.reshape(-1, 1)
    
    #Características de prueba
    X_prueba_petallength = X_prueba.iloc[:, 0].values.reshape(-1, 1)
    X_prueba_petalwidth = X_prueba.iloc[:, 1].values.reshape(-1, 1)
    
    #Entrenamos y evaluamos los clasificadores
    entrenar_y_evaluar(clf_petallength, X_entrenamiento_petallength, y_entrenamiento, X_prueba_petallength, y_prueba, "petallength")
    entrenar_y_evaluar(clf_petalwidth, X_entrenamiento_petalwidth, y_entrenamiento, X_prueba_petalwidth, y_prueba, "petalwidth")

elif eleccion == "2":
    print("\n》Regresión Logística\n")
    clf_petallength = LogisticRegression()
    clf_petalwidth = LogisticRegression()
    X_entrenamiento_petallength = X_entrenamiento.iloc[:, 0].values.reshape(-1, 1)
    X_entrenamiento_petalwidth = X_entrenamiento.iloc[:, 1].values.reshape(-1, 1)
    X_prueba_petallength = X_prueba.iloc[:, 0].values.reshape(-1, 1)
    X_prueba_petalwidth = X_prueba.iloc[:, 1].values.reshape(-1, 1)
    entrenar_y_evaluar(clf_petallength, X_entrenamiento_petallength, y_entrenamiento, X_prueba_petallength, y_prueba, "petallength")
    entrenar_y_evaluar(clf_petalwidth, X_entrenamiento_petalwidth, y_entrenamiento, X_prueba_petalwidth, y_prueba, "petalwidth")

elif eleccion == "3":
    print("\n》K-Vecinos más cercanos (KNN)\n")
    clf_petallength = KNeighborsClassifier(n_neighbors=1)
    clf_petalwidth = KNeighborsClassifier(n_neighbors=1)
    X_entrenamiento_petallength = X_entrenamiento.iloc[:, 0].values.reshape(-1, 1)
    X_entrenamiento_petalwidth = X_entrenamiento.iloc[:, 1].values.reshape(-1, 1)
    X_prueba_petallength = X_prueba.iloc[:, 0].values.reshape(-1, 1)
    X_prueba_petalwidth = X_prueba.iloc[:, 1].values.reshape(-1, 1)
    entrenar_y_evaluar(clf_petallength, X_entrenamiento_petallength, y_entrenamiento, X_prueba_petallength, y_prueba, "petallength")
    entrenar_y_evaluar(clf_petalwidth, X_entrenamiento_petalwidth, y_entrenamiento, X_prueba_petalwidth, y_prueba, "petalwidth")

elif eleccion == '4':
    print("\n¡Hasta luego! ♥\n")
    exit()

else:
    print("\n☢ ERROR: Ingrese una opción válida \n\n")
    print("\n¡Hasta luego! ♥\n")
    exit()