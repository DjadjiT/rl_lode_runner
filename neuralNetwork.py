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
        self.height = len(self.lines)
        self.width = len(self.lines[0])
        self.reset()

    def reset(self):
        self.starting_point = (None, None)
        self.keys = []
        self.keys_taken = 0

        self.board = [[self.lines[i][j] for i in range(self.height)] for j in range(self.width)]  # On initialise un tableau height * width

        for row in range(self.height):
            for col in range(len(self.board[row])):
                if self.board[row][col] == 'p':
                    self.starting_point = (row, col)
                elif self.board[row][col] == 'c':
                    self.keys.append((row, col))

    def apply(self, action):
        i, j = int(action/self.width), action%self.height

        if (i, j) in self.states:
            # calculer la récompense
            if self.board[(i, j)] in ['_', '#']:
                reward = REWARD_STUCK
            elif self.board[(i, j)] in ['c']:
                self.board[(i, j)] = " "
                self.keys_taken += 1
                reward = REWARD_KEY
            else:
                reward = REWARD_DEFAULT
        else:
            # Etat impossible: grosse pénalité
            reward = REWARD_FORBIDDEN

        return reward

    def map_is_done(self):
        return self.keys_taken == len(self.keys)

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
        self.actions = list(range(environment.width * environment.height))  # Crée une liste de size*size actions
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
    while not environment.map_is_done():
        action = agent.best_action()
        agent.do(action)
        agent.update_policy()
        if verbose:
            environment.find_winner()


if __name__ == "__main__":
    environment = Environment(MAZE)
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