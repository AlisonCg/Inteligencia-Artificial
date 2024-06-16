import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Cargar el conjunto de datos
df_cancer = pd.read_csv('cancer.csv')

# Dividir los datos en características (X) y etiquetas (y)
X_cancer = df_cancer.iloc[:, 1:]
y_cancer = df_cancer['diagnosis']

# Método Hold-Out 70/30
X_train_cancer, X_test_cancer, y_train_cancer, y_test_cancer = train_test_split(X_cancer, y_cancer, test_size=0.3, random_state=42)

# Clasificador 1-NN
clf_1nn_cancer = KNeighborsClassifier(n_neighbors=1)
clf_1nn_cancer.fit(X_train_cancer, y_train_cancer)
y_pred_1nn_cancer = clf_1nn_cancer.predict(X_test_cancer)
accuracy_hold_out_1nn_cancer = accuracy_score(y_test_cancer, y_pred_1nn_cancer)
conf_matrix_hold_out_1nn_cancer = confusion_matrix(y_test_cancer, y_pred_1nn_cancer)

# Clasificador k-NN con búsqueda de hiperparámetros
param_grid_cancer = {'n_neighbors': range(1, 11)}
grid_search_cancer = GridSearchCV(KNeighborsClassifier(), param_grid_cancer, cv=5)
grid_search_cancer.fit(X_cancer, y_cancer)
best_k_cancer = grid_search_cancer.best_params_['n_neighbors']
clf_knn_best_cancer = KNeighborsClassifier(n_neighbors=best_k_cancer)
cv_scores_knn_cancer = cross_val_score(clf_knn_best_cancer, X_cancer, y_cancer, cv=10)
accuracy_cross_val_knn_cancer = cv_scores_knn_cancer.mean()
y_pred_cv_knn_cancer = cross_val_predict(clf_knn_best_cancer, X_cancer, y_cancer, cv=10)
conf_matrix_cross_val_knn_cancer = confusion_matrix(y_cancer, y_pred_cv_knn_cancer)

# Clasificador MLP
mlp = MLPClassifier(random_state=42, max_iter=1000)
mlp.fit(X_train_cancer, y_train_cancer)
y_pred_mlp_cancer = mlp.predict(X_test_cancer)
accuracy_hold_out_mlp_cancer = accuracy_score(y_test_cancer, y_pred_mlp_cancer)
conf_matrix_hold_out_mlp_cancer = confusion_matrix(y_test_cancer, y_pred_mlp_cancer)

# Cross Validation con MLP
cv_scores_mlp_cancer = cross_val_score(mlp, X_cancer, y_cancer, cv=10)
accuracy_cross_val_mlp_cancer = cv_scores_mlp_cancer.mean()
y_pred_cv_mlp_cancer = cross_val_predict(mlp, X_cancer, y_cancer, cv=10)
conf_matrix_cross_val_mlp_cancer = confusion_matrix(y_cancer, y_pred_cv_mlp_cancer)

# Resultados 1-NN
print(f'Accuracy (Hold-Out) - 1-NN (Cancer): {accuracy_hold_out_1nn_cancer}')
print(f'Matriz de Confusión (Hold-Out) - 1-NN (Cancer):\n{conf_matrix_hold_out_1nn_cancer}')

# Resultados k-NN
print(f'Accuracy (Cross Validation) - k-NN (k={best_k_cancer}) (Cancer): {accuracy_cross_val_knn_cancer}')
print(f'Matriz de Confusión (Cross Validation) - k-NN (k={best_k_cancer}) (Cancer):\n{conf_matrix_cross_val_knn_cancer}')

# Resultados MLP
print(f'Accuracy (Hold-Out) - MLP (Cancer): {accuracy_hold_out_mlp_cancer}')
print(f'Matriz de Confusión (Hold-Out) - MLP (Cancer):\n{conf_matrix_hold_out_mlp_cancer}')
print(f'Accuracy (Cross Validation) - MLP (Cancer): {accuracy_cross_val_mlp_cancer}')
print(f'Matriz de Confusión (Cross Validation) - MLP (Cancer):\n{conf_matrix_cross_val_mlp_cancer}')
