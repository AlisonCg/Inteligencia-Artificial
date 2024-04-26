import math
import random
class Jugador():

    def __init__(self, letra):
        self.letra = letra

    def obt_movimiento(self, juego):
        pass

class Humano(Jugador):
    # Inicializamos jugador humano
    def __init__(self, letra):
        super().__init__(letra)

    # Movimiento de jugador humano
    def obt_movimiento(self, juego):
        cuadrado_valido = False
        val = None
        while not cuadrado_valido:
            cuadrado = input('\n✎  Es turno del jugador "' + self.letra + '" - Ingrese una posición del tablero (0-8): ')
            try:
                val = int(cuadrado)
                if val not in juego.mov_disponible():
                    raise ValueError
                cuadrado_valido = True
            except ValueError:
                print('\n☢ ERROR: Cuadrado inválido. Ingrese una opción válida.')
        return val
class JugadorCompu(Jugador):
    def __init__(self, letra):
        super().__init__(letra)

    # Movimiento de computadora
    def obt_movimiento(self, juego):
        cuadrado = random.choice(juego.mov_disponible())
        return cuadrado
class Computadora(Jugador):

    def __init__(self, letra):
        super().__init__(letra)  # Inicializamos jugador

    def obt_movimiento(self, juego):
        if len(juego.mov_disponible()) == 9:
            cuadrado = random.choice(juego.mov_disponible())
        else:
            cuadrado = self.minimax(juego, self.letra, -math.inf, math.inf, 0)['posicion']
        return cuadrado

    # Algoritmo minimax con poda alfa-beta y conteo de movimientos
    def minimax(self, estado, Jugador, alpha, beta, movimientos):
        maJugadorX = self.letra
        jugador_contrario = 'O' if Jugador == 'X' else 'X'

        # Verificamos si el movimiento anterior llevó al jugador contrario a ganar
        if estado.ganador_actual == jugador_contrario:
            # Si el jugador contrario ganó => se asigna una puntuación basada en la cantidad de casillas vacías más uno
            # Si el jugador contrario es igual a 'maJugadorX'=> devolvemos una puntuación positiva sino negativa
            return {'posicion': None, 'puntaje': 1 * (estado.num_vacios() + 1) if jugador_contrario == maJugadorX else -1 * (estado.num_vacios() + 1)}
        elif not estado.vacios():
            # Si no hay ganador y no quedan casillas vacías 0> se considera un empate y se devuelve una puntuación de 0
            return {'posicion': None, 'puntaje': 0}

        if Jugador == maJugadorX:
            mejor = {'posicion': None, 'puntaje': -math.inf}  # Cada puntuación debe maximizarse
            for posible_mov in estado.mov_disponible():
                estado.movimiento(posible_mov, Jugador)
                sim_puntaje = self.minimax(estado, jugador_contrario, alpha, beta, movimientos + 1)  
                # Simulamos un juego después de hacer ese movimiento
                estado.tablero[posible_mov] = ' '
                estado.ganador_actual = None
                sim_puntaje['posicion'] = posible_mov  # Siguiente movimiento óptimo
                mejor = max(mejor, sim_puntaje, key=lambda x: x['puntaje'])
                alpha = max(alpha, mejor['puntaje'])
                if beta <= alpha:
                    break
            return mejor
        else:
            mejor = {'posicion': None, 'puntaje': math.inf}  # Cada puntuación debe minimizarse
            for posible_mov in estado.mov_disponible():
                estado.movimiento(posible_mov, Jugador)
                sim_puntaje = self.minimax(estado, jugador_contrario, alpha, beta, movimientos + 1)
                # Simulamos un juego después de hacer ese movimiento
                estado.tablero[posible_mov] = ' '
                estado.ganador_actual = None
                sim_puntaje['posicion'] = posible_mov  # Siguiente movimiento óptimo
                mejor = min(mejor, sim_puntaje, key=lambda x: x['puntaje'])
                beta = min(beta, mejor['puntaje'])
                if beta <= alpha:
                    break
            return mejor
