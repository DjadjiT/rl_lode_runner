'''
ESGI
Jeu de morpion 5AL
'''
from sklearn.neural_network import MLPRegressor
import numpy as np

DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_DISCOUNT_FACTOR = 0.1
DEFAULT_BOARD_SIZE = 3

REWARD_DEFAULT = -1
REWARD_KEY = 30
REWARD_STUCK = -6
REWARD_FORBIDDEN = -60

NOISE_INIT = 0.5
NOISE_DECAY = 0.9

UP, DOWN, LEFT, RIGHT = 'U', 'D', 'L', 'R'
ACTIONS = [UP, DOWN, LEFT, RIGHT]

MAZE = """
  --------#    
 __       #___ 
c      c  #    
_  __#__  #    
    c#  ___   _
  _#__        
 p #         __
_______________ 
"""

class Environment:
    def __init__(self, text):
        self.lines = text.strip().split('\n')
        self.key_found = 0
        self.height = len(self.lines)
        self.width = len(self.lines[0])

        self.actions = [[(i, j) for i in range(self.height)] for j in
                        range(self.width)]  # Les actions sont toutes les coordonnées possibles
        self.reset()

    def reset(self):
        self.board = [[self.lines[(i, j)] for i in range(self.height)] for j in
                      range(self.width)]  # On initialise un tableau size * size
        self.all_keys_found = False
        self.key_found = 0


    def find_keys(self):
        '''
            Identification du vainqueur
            Retourne le premier vainqueur trouvé (pas de gestion de ex-aequo - ce cas n'est pas supposé arriver)
            On pourrait coder plus propre...
        '''

        # Recherche par ligne
        for line in self.board:
            if line[0] is not EMPTY and line[0] == "c": #and line.count(line[0]) == self.width:
                self.key_found += 1

        # Recherche par colonne
        for j in range(len(self.board)):
            mark = self.board[0][j]
            if mark != EMPTY:
                for i in range(1, len(self.board)):
                    if self.board[i][j] == "c":
                        self.key_found += 1
                    else:
                        break

        # Et les deux diagonales
        mark = self.board[0][0]
        if mark != EMPTY:
            for i in range(1, len(self.board)):
                if self.board[i][i] == mark:
                     self.key_found += 1
                else:
                    break
            if count == self.height:
                return mark

        mark = self.board[0][self.height - 1]
        if mark != EMPTY:
            count = 1
            for i in range(1, len(self.board)):
                if self.board[i][self.height - i - 1] == mark:
                    count += 1
                else:
                    break
            if count == self.height:
                return mark

        return None

    def apply(self, action):
        '''
        Paramètres :
            action : coordonnées de la case jouée au format (i,j)
            player : joueur qui joue l'action, permet d'inscrire sa marque ("X" ou "O" par exemple)
        '''
        i, j = int(action / DEFAULT_BOARD_SIZE), action % DEFAULT_BOARD_SIZE  # On convertit l'action en coordonnées (ligne, colonne)
        reward = REWARD_DEFAULT
        if self.board[i][j] is EMPTY:
            self.board[i][j] = player.mark
            winner_mark = self.find_winner()
            self.game_over = winner_mark not in [EMPTY, None]
            if winner_mark == player.mark:
                reward = REWARD_KEY
            elif self.game_over:
                reward = REWARD_LOST
        else:
            reward = REWARD_FORBIDDEN

        return reward


class Agent:
    def __init__(self, environment):
        self.environment = environment
        self.policy = Policy(environment)
        self.reset()

    def reset(self):
        self.state = self.board_to_state(environment.board)
        self.previous_state = self.state
        self.score = 0

    # Transforme le tableau de jeu en vecteur d'entrée pour mon réseau de neurones
    # Mon état d'agent est ce vecteur d'entrée (9 case si le plateau fait 3x3)
    def board_to_state(self, board):
        vector = []
        for line in board:
            for v in line:
                if v == ' ':
                    vector.append(2)
                elif v == '#':
                    vector.append(1)
                elif v == 'c':
                    vector.append(0)
                elif v == '-':
                    vector.append(-1)
        return vector

    def best_action(self):
        return self.policy.best_action(self.state)

    def do(self, action):
        self.previous_state = self.board_to_state(environment.board)
        self.reward = self.environment.apply(action, self)
        self.state = self.board_to_state(environment.board)
        self.score += self.reward
        self.last_action = action

    def update_policy(self):
        self.policy.update(self.previous_state, self.state, self.last_action, self.reward)


class Policy:
    def __init__(self, environment,
                 learning_rate=DEFAULT_LEARNING_RATE,
                 discount_factor=DEFAULT_DISCOUNT_FACTOR):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.mlp = MLPRegressor(hidden_layer_sizes=(20,),
                                max_iter=1,
                                activation='tanh',
                                solver='sgd',
                                learning_rate_init=self.learning_rate,
                                warm_start=True)
        self.actions = list(range(environment.heigth * environment.width))  # Crée une liste de size*size actions
        self.noise = NOISE_INIT

        # On initialise le ANN avec 9 entrées, 9 sorties
        self.mlp.fit([[0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0]])

    def __repr__(self):
        res = ''
        for state in self.table:
            res += f'{state}\t{self.table[state]}\n'
        return res

    def best_action(self, state):
        # self.proba_state = self.mlp.predict_proba([state])[0] #Le RN fournit un vecteur de probabilité
        self.proba_state = self.mlp.predict([state])[0]  # Le RN fournit un vecteur de probabilité
        self.noise *= NOISE_DECAY
        self.proba_state += np.random.rand(len(self.proba_state)) * self.noise
        action = self.actions[np.argmax(self.proba_state)]  # On choisit l'action la plus probable
        return action

    def update(self, previous_state, state, last_action, reward):
        # Q(st, at) = Q(st, at) + learning_rate * (reward + discount_factor * max(Q(state)) - Q(st, at))
        # Mettre le réseau de neurone à jour, au lieu de la table
        maxQ = np.amax(self.proba_state)
        self.proba_state[last_action] = reward + self.discount_factor * maxQ
        inputs = [state]
        outputs = [self.proba_state]
        # print(inputs, outputs)
        self.mlp.fit(inputs, outputs)


def game_turn(environment, agent, verbose=False):
    environment.reset()
    agent.reset()
    while not environment.game_over:
        action = agent.best_action()
        agent.do(action)
        agent.update_policy()
        if verbose:
            environment.find_winner()


if __name__ == "__main__":
    environment = Environment()
    ##    environment.board[0][2] = 'o'
    ##    environment.board[1][1] = 'o'
    ##    environment.board[2][0] = 'o'
    ##    print(environment)
    ##    print(environment.find_winner())
    ##    STOP

    agent = Agent(environment)

    # Boucle du RL
    for i in range(100):
        game_turn(environment, agent)
        print(agent.score, agent.policy.noise)

    game_turn(environment, agent, True)