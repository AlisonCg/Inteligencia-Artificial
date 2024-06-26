import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from scipy.spatial import distance
from sklearn.datasets import load_iris

# Implementar el Clasificador Euclidiano
class ClasificadorEuclidiano:
    def ajustar(self, X_entrenamiento, y_entrenamiento):
        self.X_entrenamiento = X_entrenamiento
        self.y_entrenamiento = y_entrenamiento

    def predecir(self, X_prueba):
        predicciones = []
        for punto_prueba in X_prueba:
            distancias = distance.cdist([punto_prueba], self.X_entrenamiento, 'euclidean').flatten()
            indice_vecino_mas_cercano = np.argmin(distancias)
            predicciones.append(self.y_entrenamiento[indice_vecino_mas_cercano])
        return np.array(predicciones)

def evaluar_modelo(X, y):
    # Hold-Out 70/30
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.3, random_state=42)

    clasificador_euclidiano = ClasificadorEuclidiano()
    clasificador_euclidiano.ajustar(X_entrenamiento, y_entrenamiento)
    y_pred = clasificador_euclidiano.predecir(X_prueba)

    exactitud_holdout = accuracy_score(y_prueba, y_pred)
    print(f'Precisión Hold-Out 70/30: {exactitud_holdout}')

    # Validación Cruzada 10-Fold
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    exactitudes_cross_val = []

    for indice_entrenamiento, indice_prueba in kf.split(X):
        X_entrenamiento, X_prueba = X[indice_entrenamiento], X[indice_prueba]
        y_entrenamiento, y_prueba = y[indice_entrenamiento], y[indice_prueba]
        
        clasificador_euclidiano.ajustar(X_entrenamiento, y_entrenamiento)
        y_pred = clasificador_euclidiano.predecir(X_prueba)
        
        exactitud_fold = accuracy_score(y_prueba, y_pred)
        exactitudes_cross_val.append(exactitud_fold)

    media_exactitud = np.mean(exactitudes_cross_val)
    std_exactitud = np.std(exactitudes_cross_val)
    print(f'Precisión Validación Cruzada 10-Fold: {media_exactitud} ± {std_exactitud}')

# Cargar el conjunto de datos Iris
iris = load_iris()
X_iris = iris.data
y_iris = iris.target

print("Evaluando Conjunto de Datos Iris")
evaluar_modelo(X_iris, y_iris)

# Cargar el conjunto de datos Wine desde un archivo CSV
wine_data = pd.read_csv('wine.csv')
X_wine = wine_data.iloc[:, 1:].values  
y_wine = wine_data.iloc[:, 0].values  
print("\nEvaluando Conjunto de Datos Wine")
evaluar_modelo(X_wine, y_wine)
