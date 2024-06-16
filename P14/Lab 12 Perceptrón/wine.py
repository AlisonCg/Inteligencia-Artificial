import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.mixture import BayesianGaussianMixture
from sklearn.neural_network import MLPClassifier

# Cargar el conjunto de datos
df_wine = pd.read_csv('wine.csv')

# Dividir los datos en características (X) y etiquetas (y)
X_wine = df_wine.iloc[:, 1:]
y_wine = df_wine.iloc[:, 0]

# Método Hold-Out 70/30
X_entrenamiento_wine, X_prueba_wine, y_entrenamiento_wine, y_prueba_wine = train_test_split(X_wine, y_wine, test_size=0.3, random_state=42)

# Clasificador Naïve Bayes con Hold-Out
clf_nb_wine = GaussianNB()
clf_nb_wine.fit(X_entrenamiento_wine, y_entrenamiento_wine)
y_prediccion_nb_wine = clf_nb_wine.predict(X_prueba_wine)
precision_hold_out_nb_wine = accuracy_score(y_prueba_wine, y_prediccion_nb_wine)
matriz_confusion_hold_out_nb_wine = confusion_matrix(y_prueba_wine, y_prediccion_nb_wine)

# Clasificador Naïve Bayes con Cross-Validation
puntajes_cv_nb_wine = cross_val_score(clf_nb_wine, X_wine, y_wine, cv=10)
precision_cross_val_nb_wine = puntajes_cv_nb_wine.mean()
y_prediccion_cv_nb_wine = cross_val_predict(clf_nb_wine, X_wine, y_wine, cv=10)
matriz_confusion_cross_val_nb_wine = confusion_matrix(y_wine, y_prediccion_cv_nb_wine)

# Clasificador de Redes Bayesianas con Hold-Out usando BayesianGaussianMixture
clf_bgm_wine = BayesianGaussianMixture(n_components=3, max_iter=1000, random_state=42)
clf_bgm_wine.fit(X_entrenamiento_wine)
y_prediccion_bgm_wine = clf_bgm_wine.predict(X_prueba_wine)

precision_hold_out_bgm_wine = accuracy_score(y_prueba_wine, y_prediccion_bgm_wine)
matriz_confusion_hold_out_bgm_wine = confusion_matrix(y_prueba_wine, y_prediccion_bgm_wine)

# Clasificador de Redes Bayesianas con Cross-Validation usando BayesianGaussianMixture
puntajes_cv_bgm_wine = cross_val_score(clf_bgm_wine, X_wine, y_wine, cv=10)
precision_cross_val_bgm_wine = puntajes_cv_bgm_wine.mean()
y_prediccion_cv_bgm_wine = cross_val_predict(clf_bgm_wine, X_wine, y_wine, cv=10)
matriz_confusion_cross_val_bgm_wine = confusion_matrix(y_wine, y_prediccion_cv_bgm_wine)

# Clasificador Perceptrón Multicapa
mlp_clf_wine = MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=42)
mlp_clf_wine.fit(X_entrenamiento_wine, y_entrenamiento_wine)
y_prediccion_mlp_wine = mlp_clf_wine.predict(X_prueba_wine)
precision_hold_out_mlp_wine = accuracy_score(y_prueba_wine, y_prediccion_mlp_wine)
matriz_confusion_hold_out_mlp_wine = confusion_matrix(y_prueba_wine, y_prediccion_mlp_wine)

# Cross-Validation con Perceptrón Multicapa
puntajes_cv_mlp_wine = cross_val_score(mlp_clf_wine, X_wine, y_wine, cv=10)
precision_cross_val_mlp_wine = puntajes_cv_mlp_wine.mean()
y_prediccion_cv_mlp_wine = cross_val_predict(mlp_clf_wine, X_wine, y_wine, cv=10)
matriz_confusion_cross_val_mlp_wine = confusion_matrix(y_wine, y_prediccion_cv_mlp_wine)

# Resultados
print(f'\nAccuracy (Hold-Out) - Naïve Bayes (wine): {precision_hold_out_nb_wine}')
print(f'Matriz de Confusión (Hold-Out) - Naïve Bayes (wine):\n{matriz_confusion_hold_out_nb_wine}')
print(f'Accuracy (Cross Validation) - Naïve Bayes (wine): {precision_cross_val_nb_wine}')
print(f'Matriz de Confusión (Cross Validation) - Naïve Bayes (wine):\n{matriz_confusion_cross_val_nb_wine}\n')

print(f'Accuracy (Hold-Out) - Redes Bayesianas (Aproximación con BGM) (wine): {precision_hold_out_bgm_wine}')
print(f'Matriz de Confusión (Hold-Out) - Redes Bayesianas (Aproximación con BGM) (wine):\n{matriz_confusion_hold_out_bgm_wine}')
print(f'Accuracy (Cross Validation) - Redes Bayesianas (Aproximación con BGM) (wine): {precision_cross_val_bgm_wine}')
print(f'Matriz de Confusión (Cross Validation) - Redes Bayesianas (Aproximación con BGM) (wine):\n{matriz_confusion_cross_val_bgm_wine}\n')

print(f'Accuracy (Hold-Out) - Perceptrón Multicapa (wine): {precision_hold_out_mlp_wine}')
print(f'Matriz de Confusión (Hold-Out) - Perceptrón Multicapa (wine):\n{matriz_confusion_hold_out_mlp_wine}')
print(f'Accuracy (Cross Validation) - Perceptrón Multicapa (wine): {precision_cross_val_mlp_wine}')
print(f'Matriz de Confusión (Cross Validation) - Perceptrón Multicapa (wine):\n{matriz_confusion_cross_val_mlp_wine}\n')
