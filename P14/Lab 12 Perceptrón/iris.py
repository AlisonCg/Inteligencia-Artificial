import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.mixture import BayesianGaussianMixture
from sklearn.neural_network import MLPClassifier

# Cargar el conjunto de datos
df = pd.read_csv('iris.csv')

# Dividir los datos en características (X) y etiquetas (y)
X = df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]
y = df['variety']

# Método Hold-Out 70/30
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.3, random_state=42)

# Clasificador Naïve Bayes con Hold-Out
clf_nb = GaussianNB()
clf_nb.fit(X_entrenamiento, y_entrenamiento)
y_prediccion_nb = clf_nb.predict(X_prueba)
precision_hold_out_nb = accuracy_score(y_prueba, y_prediccion_nb)
matriz_confusion_hold_out_nb = confusion_matrix(y_prueba, y_prediccion_nb)

# Clasificador Naïve Bayes con Cross-Validation
puntajes_cv_nb = cross_val_score(clf_nb, X, y, cv=10)
precision_cross_val_nb = puntajes_cv_nb.mean()
y_prediccion_cv_nb = cross_val_predict(clf_nb, X, y, cv=10)
matriz_confusion_cross_val_nb = confusion_matrix(y, y_prediccion_cv_nb)

# Clasificador de Redes Bayesianas con Hold-Out usando BayesianGaussianMixture
clf_bgm = BayesianGaussianMixture(n_components=3, max_iter=1000, random_state=42)
clf_bgm.fit(X_entrenamiento)
y_prediccion_bgm = clf_bgm.predict(X_prueba)

# Mapeo de las etiquetas predichas a las etiquetas reales
mapeo_variedad = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
y_prediccion_bgm_mapeado = pd.Series(y_prediccion_bgm).map(mapeo_variedad)

precision_hold_out_bgm = accuracy_score(y_prueba, y_prediccion_bgm_mapeado)
matriz_confusion_hold_out_bgm = confusion_matrix(y_prueba, y_prediccion_bgm_mapeado)

# Clasificador de Redes Bayesianas con Cross-Validation usando BayesianGaussianMixture
puntajes_cv_bgm = cross_val_score(clf_bgm, X, y, cv=10)
precision_cross_val_bgm = puntajes_cv_bgm.mean()
y_prediccion_cv_bgm = cross_val_predict(clf_bgm, X, y, cv=10)

# Mapeo de las etiquetas predichas a las etiquetas reales
y_prediccion_cv_bgm_mapeado = pd.Series(y_prediccion_cv_bgm).map(mapeo_variedad)

matriz_confusion_cross_val_bgm = confusion_matrix(y, y_prediccion_cv_bgm_mapeado)

# Clasificador Perceptrón Multicapa
mlp_clf = MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=42)
mlp_clf.fit(X_entrenamiento, y_entrenamiento)
y_prediccion_mlp = mlp_clf.predict(X_prueba)
precision_hold_out_mlp = accuracy_score(y_prueba, y_prediccion_mlp)
matriz_confusion_hold_out_mlp = confusion_matrix(y_prueba, y_prediccion_mlp)

# Cross-Validation con Perceptrón Multicapa
puntajes_cv_mlp = cross_val_score(mlp_clf, X, y, cv=10)
precision_cross_val_mlp = puntajes_cv_mlp.mean()
y_prediccion_cv_mlp = cross_val_predict(mlp_clf, X, y, cv=10)
matriz_confusion_cross_val_mlp = confusion_matrix(y, y_prediccion_cv_mlp)

# Resultados
print(f'\nAccuracy (Hold-Out) - Naïve Bayes: {precision_hold_out_nb}')
print(f'Matriz de Confusión (Hold-Out) - Naïve Bayes:\n{matriz_confusion_hold_out_nb}')
print(f'Accuracy (Cross Validation) - Naïve Bayes: {precision_cross_val_nb}')
print(f'Matriz de Confusión (Cross Validation) - Naïve Bayes:\n{matriz_confusion_cross_val_nb}\n')

print(f'Accuracy (Hold-Out) - Redes Bayesianas (Aproximación con BGM): {precision_hold_out_bgm}')
print(f'Matriz de Confusión (Hold-Out) - Redes Bayesianas (Aproximación con BGM):\n{matriz_confusion_hold_out_bgm}')
print(f'Accuracy (Cross Validation) - Redes Bayesianas (Aproximación con BGM): {precision_cross_val_bgm}')
print(f'Matriz de Confusión (Cross Validation) - Redes Bayesianas (Aproximación con BGM):\n{matriz_confusion_cross_val_bgm}\n')

print(f'Accuracy (Hold-Out) - Perceptrón Multicapa: {precision_hold_out_mlp}')
print(f'Matriz de Confusión (Hold-Out) - Perceptrón Multicapa:\n{matriz_confusion_hold_out_mlp}')
print(f'Accuracy (Cross Validation) - Perceptrón Multicapa: {precision_cross_val_mlp}')
print(f'Matriz de Confusión (Cross Validation) - Perceptrón Multicapa:\n{matriz_confusion_cross_val_mlp}\n')
