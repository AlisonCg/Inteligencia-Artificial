import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

# Cargar el conjunto de datos
df_cancer = pd.read_csv('cancer.csv')

# Dividir los datos en características (X) y etiquetas (y)
X_cancer = df_cancer.iloc[:, 1:]  # Todas las columnas excepto la primera
y_cancer = df_cancer['diagnosis']  # La columna 'diagnosis'

# Método Hold-Out 70/30 estratificado
X_train_cancer, X_test_cancer, y_train_cancer, y_test_cancer = train_test_split(
    X_cancer, y_cancer, test_size=0.3, random_state=42, stratify=y_cancer)

# Información de los conjuntos de entrenamiento y prueba
print("Hold-Out 70/30 Estratificado:")
print("Tamaño del conjunto de entrenamiento:", len(X_train_cancer))
print("Tamaño del conjunto de prueba:", len(X_test_cancer))

# Preparar 10-Fold Cross-Validation estratificado
skf = StratifiedKFold(n_splits=10)

# Verificar que los pliegues son disjuntos y estratificados
print("10-Fold Cross-Validation Estratificado:")
for fold, (train_idx, test_idx) in enumerate(skf.split(X_cancer, y_cancer)):
    X_train, X_test = X_cancer.iloc[train_idx], X_cancer.iloc[test_idx]
    y_train, y_test = y_cancer.iloc[train_idx], y_cancer.iloc[test_idx]
    print(f"Fold {fold+1}:")
    print("  Tamaño del conjunto de entrenamiento:", len(X_train))
    print("  Tamaño del conjunto de prueba:", len(X_test))
    print("  Proporción de clases en entrenamiento:", y_train.value_counts(normalize=True))
    print("  Proporción de clases en prueba:", y_test.value_counts(normalize=True))