import random
import matplotlib.pyplot as plt
import numpy as nump
#variables
WIDTH, HEIGHT = 500, 500
#matriz
variables = [
     "B","D","E","F"
]
ecuaciones = [
    "16B - 6D + 4E + F = 36",
    "B - 8D + E + F = -64",
    "16B + 2D - 4E + F = -4",
    "9B + 8D - 3E + F = -64"
]

#matriz de coeficientes
ecs = nump.array([[16.0,-6.0,4.0,1.0],
                  [1.0,-8.0,1.0,1.0],
                  [16.0,2.0,-4.0,1.0],
                  [9.0,8.0,-3.0,1.0]]) 
#vector de resultados
sol_array= nump.array ([[36.0],[-64.0],[-4.0],[-64.0]])

#Array tipo numpy que rellena con valores random entre -100 y 100 y lo convierte a una matriz vertical de 1 columna
valoresAlt = nump.array([random.uniform(-100.0, 100.0) for _ in range(4)]).reshape(4, 1)

x = nump.dot(ecs,valoresAlt)#realiza la multiplicacion de matrices entre la matriz de coeficientes y la matriz de valores aleatorios
# Calculo de la distancia euclidiana
cercania = nump.allclose(x, sol_array, atol=1e-2)#Dice si la solucion aleatoria y la real son cercanas, con un margen de toleracia  de 0.01 
rest = x - sol_array#calcula la distancia entre la solucion real y la aleatoria
dist = nump.sqrt(nump.sum(nump.square(rest)))#calcula la distancia euclidiana


# Graficar la matriz de coeficientes y el vector de resultados
fig, ax = plt.subplots(figsize=(12, 7))
plt.title('Matriz de Coeficientes y Vector de Resultados')

spacing = 0.1 # Espacio entre las líneas de texto
for i,ec in enumerate(ecuaciones):
     ax.text(-0.1, 0.7 - (i * spacing), ec, fontsize=12, ha='left')
# Mostrar los valores aleatorios y los resultados
for i, (val, mult_result) in enumerate(zip(valoresAlt, x)):
    ax.text(0.5, 0.7 - (i * spacing), f'{variables[i]} = {val[0]:.2f}', fontsize=12, ha='right')
    ax.text(0.9, 0.7 - (i * spacing), f'Aleat. x coef. = {mult_result[0]:.2f}', fontsize=12, ha='right')
# Mostrar el vector de resultados real
for i, res in enumerate(sol_array):
    ax.text(1.15, 0.7 - (i * spacing), f'Real = {res[0]}', fontsize=12, ha='right')

plt.axis('off')


# Ajusta el espacio de la ventana para evitar que el texto se encime
plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.1)
plt.show()
print(f"Valores aleatorios: {valoresAlt.T}")
print(f"Resultados de multiplicar los coeficientes por los valores aleatorios: {x.T}")
print(f"Cercanía a la solución deseada: {cercania}")
print(f"Distancia euclidiana a la solución deseada: {dist}")
