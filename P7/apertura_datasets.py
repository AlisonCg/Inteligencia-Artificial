import pandas as pd

def calcular_estadisticas(df):
    # Calculamos la media, varianza (con ddof=1 para varianza muestral) y desviación estándar
    estadisticas = pd.DataFrame({
        'Media': df.mean(),
        'Varianza': df.var(ddof=1),
        'Desviación Estándar': df.std(ddof=1)  # Asegúrate de usar ddof=1 si deseas la desviación estándar de la muestra
    })
    return estadisticas

def calcular_estadisticas_por_categoria(df, columna_categoria):
    # Obtener categorías únicas
    categorias = df[columna_categoria].unique()
    resultados = {}
    for categoria in categorias:
        # Filtrar datos por categoría y seleccionar solo las columnas numéricas
        datos_categoria = df[df[columna_categoria] == categoria].iloc[:, :4]
        resultados[categoria] = calcular_estadisticas(datos_categoria)
    return resultados

def main():
    # Carga de datos
    datos = pd.read_csv('bezdekIris.data', header=None)
    datos.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']  # Nombres de columnas para claridad
    # Estadísticas generales
    print("Estadísticas generales de las columnas numéricas:")
    estadisticas_generales = calcular_estadisticas(datos.iloc[:, :4])
    print(estadisticas_generales.to_string(index=True))  # Usamos to_string para imprimir el DataFrame completo
    # Estadísticas por categoría
    print("\nEstadísticas por categoría:")
    estadisticas_categoria = calcular_estadisticas_por_categoria(datos, 'species')
    for categoria, estadisticas in estadisticas_categoria.items():
        print(f"\nCategoría: {categoria}")
        print(estadisticas.to_string(index=True))  # Usamos to_string para imprimir el DataFrame completo

if __name__ == "__main__":
    main()
