import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np

# Cargar el conjunto de datos
df = pd.read_csv('iris.csv')
X = df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]
y = df['variety']

# Hold Out 70/30 estratificado
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Verificar que los conjuntos son disjuntos
train_indices = set(X_train.index)
test_indices = set(X_test.index)
assert train_indices & test_indices == set(), "Los conjuntos de entrenamiento y prueba no son disjuntos."

# Verificar proporciones de clases
print("Proporción de clases en el conjunto completo:", y.value_counts(normalize=True))
print("Proporción de clases en entrenamiento:", y_train.value_counts(normalize=True))
print("Proporción de clases en prueba:", y_test.value_counts(normalize=True))

# 10-Fold Cross-Validation estratificado
skf = StratifiedKFold(n_splits=10)
for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
    y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]
    print(f"Fold {fold + 1} - proporciones de clases en entrenamiento:", y_train_fold.value_counts(normalize=True))
    print(f"Fold {fold + 1} - proporciones de clases en prueba:", y_test_fold.value_counts(normalize=True))
