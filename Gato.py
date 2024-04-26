import math
import time
from jugador import Humano, Computadora  # Importamos las clases de los jugadores

# Importamos las librerías necesarias para la animación
import time
from turtle import *
import turtle

class TicTacToe():
    def __init__(self):
        self.tablero = self.crear_tablero()  # Creamos el tablero vacío
        self.ganador_actual = None  # Inicializamos el ganador como None
        self.num_movimientos = 0  # Inicializamos el contador de movimientos en 0

    @staticmethod
    def crear_tablero():
        """
        Método estático que crea un tablero vacío (una lista de 9 espacios vacíos).
        """
        return [' ' for _ in range(9)]

    def movimiento(self, cuadrado, letra):
        """
        Método que realiza un movimiento en el tablero y actualiza el estado del juego.
        Args:
            cuadrado (int): Índice del cuadrado en el que se quiere realizar el movimiento.
            letra (str): Letra ('X' o 'O') que representa al jugador que realiza el movimiento.
        Returns:
            bool: True si el movimiento se realizó con éxito, False si el cuadrado ya está ocupado.
        """
        if self.tablero[cuadrado] == ' ':  # Verifica si el cuadrado está vacío
            self.tablero[cuadrado] = letra  # Coloca la letra del jugador en el cuadrado
            self.num_movimientos += 1  # Incrementa el contador de movimientos
            if self.ganador(cuadrado, letra):  # Verifica si el movimiento resulta en un ganador
                self.ganador_actual = letra
            return True
        return False

    def num_movimientos_realizados(self):
        """
        Método que devuelve el número de movimientos realizados en el juego.
        Returns:
            int: Número de movimientos realizados.
        """
        return self.num_movimientos

    def imprimir_tablero(self):
        """
        Método que imprime el tablero de juego visualmente utilizando la librería Turtle.
        """
        time.sleep(1)  # Espera para dar una sensación de animación
        t = turtle.Turtle()
        t.width(3)  # Grosor de las líneas
        
        turtle.title("Práctica 1 | Tic Tac Toe")  # Título de la ventana
        turtle.Screen().bgcolor("black")  # Fondo negro
        turtle.Screen().setup(600, 600)  # Tamaño de la ventana
        Screen()._root.resizable(False, False)  # Evita que se pueda redimensionar la ventana

        # Dibujar el tablero
        coordenadasX = -225  # Coordenada X inicial para la primera línea
        t.up()
        t.goto(-225,250)
        t.down()
        for i in range(0, 3):
            t.color('black')  # Color de las líneas
            t.speed(-100)  # Velocidad de la tortuga
            t.color('#074DD9')  # Color del trazo
            for _ in range(4):
                t.forward(150)
                t.right(90)
            coordenadasX += 150
            t.up()
            t.goto(coordenadasX,250)
            t.down()
        t.hideturtle()  # Ocultar la tortuga al finalizar el dibujo

        # Dibujar las letras en el tablero
        for i in range(9):
            fila = i // 3
            col = i % 3
            x = -225 + col * 160 + 60
            y = 250 - fila * 160 - 60
            if self.tablero[i] == 'X':
                self.DX(x, y)
            elif self.tablero[i] == 'O':
                self.DO(x, y)
    
    def DX(self, x, y):
        """
        Método para dibujar una 'X' en una posición específica 
        en el tablero.
        Args:
            x (int): Coordenada x del centro del cuadrado.
            y (int): Coordenada y del centro del cuadrado.
        """
        t = turtle.Turtle()
        t.up()
        t.goto(-225, 250)
        t.down()
        t.color('black')
        t.speed(0)
        t.color('#F2E205')
        t.penup()
        t.goto(x - 30, y - 30)
        t.pendown()
        t.width(5)
        t.goto(x + 30, y + 30)
        t.penup()
        t.goto(x + 30, y - 30)
        t.pendown()
        t.goto(x - 30, y + 30)
        t.penup()
        t.hideturtle()

    def DO(self, x, y):
        """
        Método para dibujar una 'O' en una posición específica en el tablero.
        Args:
            x (int): Coordenada x del centro del cuadrado.
            y (int): Coordenada y del centro del cuadrado.
        """
        t = turtle.Turtle()
        t.color('black')
        t.speed(0)
        t.color('#05F2DB')
        t.penup()
        t.goto(x, y - 30)
        t.pendown()
        t.width(5)
        t.circle(30)
        t.penup()
        t.hideturtle()

    @staticmethod
    def posicion():
        """
        Método estático para imprimir las posiciones del tablero.
        """
        print('\n✭ Posiciones Del Tablero:\n')
        posicion = [[str(i) for i in range(j*3,(j+1)*3)] for j in range(3)]
        for fila in posicion:
            print('| ' + ' | '.join(fila) + ' |')

    def movimiento(self, cuadrado, letra):
        """
        Método que realiza un movimiento en el tablero y actualiza el estado del juego.
        Args:
            cuadrado (int): Índice del cuadrado en el que se quiere realizar el movimiento.
            letra (str): Letra ('X' o 'O') que representa al jugador que realiza el movimiento.
        Returns:
            bool: True si el movimiento se realizó con éxito, False si el cuadrado ya está ocupado.
        """
        if self.tablero[cuadrado] == ' ':  # Verifica si el cuadrado está vacío
            self.tablero[cuadrado] = letra  # Coloca la letra del jugador en el cuadrado
            if self.ganador(cuadrado, letra):  # Verifica si el movimiento resulta en un ganador
                self.ganador_actual = letra
            return True
        return False

    def ganador(self, cuadrado, letra):
        """
        Método que verifica si el jugador actual ha ganado el juego.
        Args:
            cuadrado (int): Índice del último cuadrado en el que se realizó un movimiento.
            letra (str): Letra ('X' o 'O') del jugador actual.
        Returns:
            bool: True si el jugador ha ganado, False en caso contrario.
        """
        # Verifica la fila
        fila_ind = math.floor(cuadrado / 3)  # Índice de fila
        fila = self.tablero[fila_ind*3:(fila_ind+1)*3]  # Extrae la fila
        if all([s == letra for s in fila]):  # Comprueba si todas las posiciones de la fila tienen la misma letra
            return True
        
        # Verifica la columna
        col_ind = cuadrado % 3  # Índice de columna
        columna = [self.tablero[col_ind+i*3] for i in range(3)]  # Extrae la columna
        if all([s == letra for s in columna]):  # Comprueba si todas las posiciones de la columna tienen la misma letra
            return True
        
        # Verifica las diagonales
        if cuadrado % 2 == 0:  # Solo para cuadrados en posiciones diagonales
            diagonal1 = [self.tablero[i] for i in [0, 4, 8]]  # Primera diagonal
            if all([s == letra for s in diagonal1]):  # Comprueba si todas las posiciones tienen la misma letra
                return True
            diagonal2 = [self.tablero[i] for i in [2, 4, 6]]  # Segunda diagonal
            if all([s == letra for s in diagonal2]):  # Comprueba si todas las posiciones tienen la misma letra
                return True
        return False  # No hay ganador

    def vacios(self):
        """
        Método que verifica si hay casillas vacías en el tablero.
        Returns:
            bool: True si hay casillas vacías, False en caso contrario.
        """
        return ' ' in self.tablero

    def num_vacios(self):
        """
        Método que devuelve el número de casillas vacías en el tablero.
        Returns:
            int: Número de casillas vacías.
        """
        return self.tablero.count(' ')

    def mov_disponible(self):
        """
        Método que devuelve una lista de índices de las casillas vacías en el tablero.
        Returns:
            list: Lista de índices de las casillas vacías.
        """
        return [i for i, x in enumerate(self.tablero) if x == " "]


def juego(juego, JugadorX, JugadorO, Resultado=True):
    """
    Función principal que ejecuta el juego.
    Args:
        juego (TicTacToe): Instancia del juego TicTacToe.
        JugadorX (Jugador): Instancia del jugador X.
        JugadorO (Jugador): Instancia del jugador O.
        Resultado (bool, optional): Indica si se mostrará el resultado del
        juego al finalizar. Por defecto es True.
    Returns:
        str: Letra ('X' o 'O') del jugador ganador, o None en caso de empate.
    """
    if Resultado:
        juego.posicion()  # Muestra las posiciones iniciales del tablero
    
    letra = 'X'  # Inicia el jugador X
    while juego.vacios():  # Mientras haya casillas vacías en el tablero
        if letra == 'O':
            cuadrado = JugadorO.obt_movimiento(juego)  # Jugador O realiza un movimiento
        else:
            cuadrado = JugadorX.obt_movimiento(juego)  # Jugador X realiza un movimiento
        if juego.movimiento(cuadrado, letra):  # Intenta realizar el movimiento en el tablero
            if Resultado:
                print('\nꕥ   El Jugador "' + letra + '" hace un movimiento en el cuadrado {}'.format(cuadrado))
                juego.imprimir_tablero()  # Imprime el tablero actualizado
                print('')

            if juego.ganador_actual:  # Verifica si hay un ganador después del movimiento
                if Resultado:
                    print('\n⋆ ★ ¡El jugador "' + letra + '" es el ganador! ⋆ ★\n')
                    turtle.exitonclick()  # Sale del juego si se hace clic en la ventana
                return letra  # Finaliza el juego y devuelve la letra del jugador ganador
            letra = 'O' if letra == 'X' else 'X'  # Cambia al otro jugador

        time.sleep(.8)  # Espera antes del próximo movimiento

    if Resultado:
        print("\n★ ¡EMPATE!\n")
        turtle.exitonclick()  # Sale del juego si se hace clic en la ventana
    return None

if __name__ == '__main__':
    print("\n----------------Práctica 1 | Juego Del Gato (tic tac toe)----------------\n")
    decision = input("1. Humano vs Computadora\n2. Computadora vs Computadora\n3. Salir\n\n")

    if decision == '1':
        JugadorX = Computadora('X')
        JugadorO = Humano('O')
        t = TicTacToe()
        juego(t, JugadorX, JugadorO, Resultado=True)

    elif decision == '2':
        JugadorX = Computadora('X')
        JugadorO = Computadora('O')
        t = TicTacToe()
        juego(t, JugadorX, JugadorO, Resultado=True)
    
    elif decision == '3':
        print("\n¡Hasta luego! ♥\n")
        exit()

    else:
        print("\n☢ ERROR: Ingrese una opción válida \n\n")
        print("\n¡Hasta luego! ♥\n")
        exit()
