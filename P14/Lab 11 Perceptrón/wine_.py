import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier  # Importar MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Cargar el conjunto de datos
df_wine = pd.read_csv('wine.csv')

# Dividir los datos en características (X) y etiquetas (y)
X_wine = df_wine.iloc[:, 1:]
y_wine = df_wine.iloc[:, 0]

# Método Hold-Out 70/30
X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(X_wine, y_wine, test_size=0.3, random_state=42)

# Clasificador 1-NN
clf_1nn_wine = KNeighborsClassifier(n_neighbors=1)
clf_1nn_wine.fit(X_train_wine, y_train_wine)
y_pred_1nn_wine = clf_1nn_wine.predict(X_test_wine)
accuracy_hold_out_1nn_wine = accuracy_score(y_test_wine, y_pred_1nn_wine)
conf_matrix_hold_out_1nn_wine = confusion_matrix(y_test_wine, y_pred_1nn_wine)

# Clasificador k-NN con búsqueda de hiperparámetros
param_grid_wine = {'n_neighbors': range(1, 11)}
grid_search_wine = GridSearchCV(KNeighborsClassifier(), param_grid_wine, cv=5)
grid_search_wine.fit(X_wine, y_wine)
best_k_wine = grid_search_wine.best_params_['n_neighbors']
clf_knn_best_wine = KNeighborsClassifier(n_neighbors=best_k_wine)
cv_scores_knn_wine = cross_val_score(clf_knn_best_wine, X_wine, y_wine, cv=10)
accuracy_cross_val_knn_wine = cv_scores_knn_wine.mean()
y_pred_cv_knn_wine = cross_val_predict(clf_knn_best_wine, X_wine, y_wine, cv=10)
conf_matrix_cross_val_knn_wine = confusion_matrix(y_wine, y_pred_cv_knn_wine)

# Clasificador MLP
mlp = MLPClassifier(random_state=42, max_iter=1000)  # Crear el MLPClassifier
mlp.fit(X_train_wine, y_train_wine)  # Entrenar el modelo MLP
y_pred_mlp_wine = mlp.predict(X_test_wine)  # Predecir con el modelo MLP
accuracy_hold_out_mlp_wine = accuracy_score(y_test_wine, y_pred_mlp_wine)  # Precisión del Hold-Out
conf_matrix_hold_out_mlp_wine = confusion_matrix(y_test_wine, y_pred_mlp_wine)  # Matriz de confusión del Hold-Out

# Cross Validation con MLP
cv_scores_mlp_wine = cross_val_score(mlp, X_wine, y_wine, cv=10)
accuracy_cross_val_mlp_wine = cv_scores_mlp_wine.mean()
y_pred_cv_mlp_wine = cross_val_predict(mlp, X_wine, y_wine, cv=10)
conf_matrix_cross_val_mlp_wine = confusion_matrix(y_wine, y_pred_cv_mlp_wine)

# Resultados 1-NN
print(f'Accuracy (Hold-Out) - 1-NN (Wine): {accuracy_hold_out_1nn_wine}')
print(f'Matriz de Confusión (Hold-Out) - 1-NN (Wine):\n{conf_matrix_hold_out_1nn_wine}')

# Resultados k-NN
print(f'Accuracy (Cross Validation) - k-NN (k={best_k_wine}) (Wine): {accuracy_cross_val_knn_wine}')
print(f'Matriz de Confusión (Cross Validation) - k-NN (k={best_k_wine}) (Wine):\n{conf_matrix_cross_val_knn_wine}')

# Resultados MLP
print(f'Accuracy (Hold-Out) - MLP (Wine): {accuracy_hold_out_mlp_wine}')
print(f'Matriz de Confusión (Hold-Out) - MLP (Wine):\n{conf_matrix_hold_out_mlp_wine}')
print(f'Accuracy (Cross Validation) - MLP (Wine): {accuracy_cross_val_mlp_wine}')
print(f'Matriz de Confusión (Cross Validation) - MLP (Wine):\n{conf_matrix_cross_val_mlp_wine}')
