import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

# Cargar el conjunto de datos
df_wine = pd.read_csv('wine.csv')

# Dividir los datos en características (X) y etiquetas (y)
X_wine = df_wine.iloc[:, 1:]  # Todas las columnas excepto la primera
y_wine = df_wine.iloc[:, 0]   # La primera columna

# Método Hold-Out 70/30 estratificado
X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(
    X_wine, y_wine, test_size=0.3, random_state=42, stratify=y_wine)

# Mostrar información de los conjuntos de entrenamiento y prueba
print("Hold Out 70/30 Estratificado:")
print("Tamaño del conjunto de entrenamiento:", len(X_train_wine))
print("Tamaño del conjunto de prueba:", len(X_test_wine))
print("Proporción de clases en entrenamiento:", y_train_wine.value_counts(normalize=True))
print("Proporción de clases en prueba:", y_test_wine.value_counts(normalize=True))

# 10-Fold Cross-Validation estratificado
skf = StratifiedKFold(n_splits=10)
print("10-Fold Cross-Validation Estratificado:")

# Ciclo para imprimir proporciones de clase en cada pliegue
for fold, (train_idx, test_idx) in enumerate(skf.split(X_wine, y_wine)):
    X_train_fold, X_test_fold = X_wine.iloc[train_idx], X_wine.iloc[test_idx]
    y_train_fold, y_test_fold = y_wine.iloc[train_idx], y_wine.iloc[test_idx]
    print(f"Fold {fold + 1}:")
    print("  Tamaño del conjunto de entrenamiento:", len(X_train_fold))
    print("  Tamaño del conjunto de prueba:", len(X_test_fold))
    print("  Proporción de clases en entrenamiento:", y_train_fold.value_counts(normalize=True))
    print("  Proporción de clases en prueba:", y_test_fold.value_counts(normalize=True))
