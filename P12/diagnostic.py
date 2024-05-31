import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.mixture import BayesianGaussianMixture

# Cargar el conjunto de datos
df_cancer = pd.read_csv('cancer.csv')

# Dividir los datos en características (X) y etiquetas (y)
X_cancer = df_cancer.iloc[:, 1:]
y_cancer = df_cancer['diagnosis']

# Método Hold-Out 70/30
X_entrenamiento_cancer, X_prueba_cancer, y_entrenamiento_cancer, y_prueba_cancer = train_test_split(X_cancer, y_cancer, test_size=0.3, random_state=42)

# Clasificador Naïve Bayes con Hold-Out
clf_nb_cancer = GaussianNB()
clf_nb_cancer.fit(X_entrenamiento_cancer, y_entrenamiento_cancer)
y_prediccion_nb_cancer = clf_nb_cancer.predict(X_prueba_cancer)
precision_hold_out_nb_cancer = accuracy_score(y_prueba_cancer, y_prediccion_nb_cancer)
matriz_confusion_hold_out_nb_cancer = confusion_matrix(y_prueba_cancer, y_prediccion_nb_cancer)

# Clasificador Naïve Bayes con Cross-Validation
puntuaciones_cv_nb_cancer = cross_val_score(clf_nb_cancer, X_cancer, y_cancer, cv=10)
precision_cross_val_nb_cancer = puntuaciones_cv_nb_cancer.mean()
y_prediccion_cv_nb_cancer = cross_val_predict(clf_nb_cancer, X_cancer, y_cancer, cv=10)
matriz_confusion_cross_val_nb_cancer = confusion_matrix(y_cancer, y_prediccion_cv_nb_cancer)

# Clasificador de Redes Bayesianas con Hold-Out usando BayesianGaussianMixture
clf_bgm_cancer = BayesianGaussianMixture(n_components=2, random_state=42)
clf_bgm_cancer.fit(X_entrenamiento_cancer)
y_prediccion_bgm_cancer = clf_bgm_cancer.predict(X_prueba_cancer)

# Mapeo de las etiquetas predichas (0, 1) a las etiquetas reales (B, M)
y_prediccion_bgm_cancer_mapeadas = pd.Series(y_prediccion_bgm_cancer).map({0: 'B', 1: 'M'})

precision_hold_out_bgm_cancer = accuracy_score(y_prueba_cancer, y_prediccion_bgm_cancer_mapeadas)
matriz_confusion_hold_out_bgm_cancer = confusion_matrix(y_prueba_cancer, y_prediccion_bgm_cancer_mapeadas)

# Clasificador de Redes Bayesianas con Cross-Validation usando BayesianGaussianMixture
puntuaciones_cv_bgm_cancer = cross_val_score(clf_bgm_cancer, X_cancer)
precision_cross_val_bgm_cancer = puntuaciones_cv_bgm_cancer.mean()
y_prediccion_cv_bgm_cancer = cross_val_predict(clf_bgm_cancer, X_cancer)

# Mapeo de las etiquetas predichas (0, 1) a las etiquetas reales (B, M)
y_prediccion_cv_bgm_cancer_mapeadas = pd.Series(y_prediccion_cv_bgm_cancer).map({0: 'B', 1: 'M'})

matriz_confusion_cross_val_bgm_cancer = confusion_matrix(y_cancer, y_prediccion_cv_bgm_cancer_mapeadas)

# Resultados
print(f'\nAccuracy (Hold-Out) - Naïve Bayes (Cancer): {precision_hold_out_nb_cancer}')
print(f'Matriz de Confusión (Hold-Out) - Naïve Bayes (Cancer):\n{matriz_confusion_hold_out_nb_cancer}')
print(f'Accuracy (Cross Validation) - Naïve Bayes (Cancer): {precision_cross_val_nb_cancer}')
print(f'Matriz de Confusión (Cross Validation) - Naïve Bayes (Cancer):\n{matriz_confusion_cross_val_nb_cancer}\n')

print(f'Accuracy (Hold-Out) - Redes Bayesianas (Aproximación con BGM) (Cancer): {precision_hold_out_bgm_cancer}')
print(f'Matriz de Confusión (Hold-Out) - Redes Bayesianas (Aproximación con BGM) (Cancer):\n{matriz_confusion_hold_out_bgm_cancer}')
print(f'Accuracy (Cross Validation) - Redes Bayesianas (Aproximación con BGM) (Cancer): {precision_cross_val_bgm_cancer}')
print(f'Matriz de Confusión (Cross Validation) - Redes Bayesianas (Aproximación con BGM) (Cancer):\n{matriz_confusion_cross_val_bgm_cancer}\n')