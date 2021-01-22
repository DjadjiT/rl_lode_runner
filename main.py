import arcade

SPRITE_IMAGE_SIZE = 128

PLAYER_MOVEMENT_SPEED = 5
ENEMY_MOVEMENT_SPEED = 3

# Scale sprites up or down
SPRITE_SCALING_PLAYER = 0.5
SPRITE_SCALING_TILES = 0.5

# Scaled sprite size for tiles
SPRITE_SIZE = int(SPRITE_IMAGE_SIZE * SPRITE_SCALING_PLAYER)

# Size of grid to show on screen, in number of tiles
SCREEN_GRID_TILE_WIDTH = 10
SCREEN_GRID_TILE_HEIGHT = 8

SCREEN_WIDTH = SPRITE_SIZE * SCREEN_GRID_TILE_WIDTH
SCREEN_HEIGHT = SPRITE_SIZE * SCREEN_GRID_TILE_HEIGHT
SCREEN_TITLE = "Lode Runner RL"

REWARD_KEY = 100
REWARD_GOAL = 250
REWARD_DEFAULT = -1
REWARD_STUCK = -6
REWARD_IMPOSSIBLE = -60

DEFAULT_LEARNING_RATE = 1
DEFAULT_DISCOUNT_FACTOR = 0.5

UP, DOWN, LEFT, RIGHT = 'U', 'D', 'L', 'R'
ACTIONS = [UP, DOWN, LEFT, RIGHT]

MAZE = """
__________
_c   c * _
_____#____
_    #   _
_  p # c _
_  __#__ _
_  c # c _
__________
"""


class Environment:
    def __init__(self, text):
        self.states = {}
        self.init_env(text)

    def init_env(self, text):
        lines = text.strip().split('\n')
        self.height = len(lines)
        self.width = len(lines[0])
        self.starting_point = (None, None)
        self.keys = []
        self.keys_taken = 0

        for row in range(self.height):
            for col in range(len(lines[row])):
                self.states[(row, col)] = lines[row][col]
                if lines[row][col] == 'p':
                    self.starting_point = (row, col)
                elif lines[row][col] == 'c':
                    self.keys.append((row, col))
                elif lines[row][col] == '*':
                    self.exit = (row, col)

    def apply(self, state, action):
        new_state = state
        if action == UP and self.states[state] == "#":
            new_state = (state[0] - 1, state[1])
        elif action == DOWN and self.states[(state[0]+1, state[1])] == "#":
            new_state = (state[0] + 1, state[1])
        elif action == LEFT:
            if self.states[(state[0] + 1, state[1])] != " ":
                new_state = (state[0], state[1] - 1)
            else:
                new_state = (state[0] + 1, state[1])
        elif action == RIGHT:
            if self.states[(state[0] + 1, state[1])] != " ":
                new_state = (state[0], state[1] + 1)
            else:
                new_state = (state[0] + 1, state[1])

        if new_state in self.states:
            # calculer la récompense
            if self.states[new_state] in ['_', 'p']:
                reward = REWARD_STUCK
            elif self.states[new_state] in ['c']:
                self.states[new_state] = " "
                self.keys_taken += 1
                reward = REWARD_KEY
            elif self.states[new_state] in ['*'] and self.map_is_done():
                reward = REWARD_GOAL
            else:
                reward = REWARD_DEFAULT
        else:
            # Etat impossible: grosse pénalité
            new_state = state
            reward = REWARD_IMPOSSIBLE

        return new_state, reward

    def map_is_done(self):
        return self.keys_taken == len(self.keys)


class Agent:
    def __init__(self, env: Environment):
        self.environment = env
        self.policy = Policy(env.states.keys(), ACTIONS)
        self.state = None
        self.score = 0
        self.previous_state = None
        self.reward = None
        self.last_action = None
        self.reset()

    def reset(self):
        self.state = self.environment.starting_point
        self.previous_state = self.state
        self.score = 0
        self.step = 0
        self.environment.init_env(MAZE)

    def best_action(self):
        return self.policy.best_action(self.state)

    def do(self, action):
        self.previous_state = self.state
        self.state, self.reward = self.environment.apply(self.state, action)
        self.score += self.reward
        self.last_action = action

    def update_policy(self):
        self.step += 1
        self.policy.update(self.previous_state, self.state, self.last_action, self.reward)
        print('#',self.step, 'ACTION:', self.last_action, 'STATE:', self.previous_state, '->', self.state, 'SCORE:', self.score,
              'KEYS TAKEN', self.environment.keys_taken)


class Policy:  # Q-table
    def __init__(self, states, actions,
                 learning_rate=DEFAULT_LEARNING_RATE,
                 discount_factor=DEFAULT_DISCOUNT_FACTOR):
        self.table = {}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        for s in states:
            self.table[s] = {}
            for a in actions:
                self.table[s][a] = 0

    def __repr__(self):
        res = ''
        for state in self.table:
            res += f'{state}\t{self.table[state]}\n'
        return res

    def best_action(self, state):
        action = None
        for a in self.table[state]:
            if action is None or self.table[state][a] > self.table[state][action]:
                action = a
        return action

    def update(self, previous_state, state, last_action, reward):
        # Q(st, at) = Q(st, at) + learning_rate * (reward + discount_factor * max(Q(state)) - Q(st, at))
        max_q = max(self.table[state].values())
        self.table[previous_state][last_action] += self.learning_rate * \
                                                   (reward + self.discount_factor * max_q - self.table[previous_state][
                                                       last_action])


class MyGame(arcade.Window):

    def __init__(self, width, height, title, agent):
        super().__init__(width, height, title)
        arcade.set_background_color(arcade.color.AMAZON)

        self.agent = agent
        self.collect_key_sound = arcade.load_sound(":resources:sounds/coin2.wav")

    def setup(self):
        # Read in the tiled map
        # map_name = "level_0.tmx"
        # my_map = arcade.tilemap.read_tmx(map_name)
        arcade.set_background_color(arcade.color.AMAZON)
        self.grounds = arcade.SpriteList()
        self.walls = arcade.SpriteList()
        self.ladders = arcade.SpriteList()
        self.exit = arcade.SpriteList()
        self.keys = arcade.SpriteList()
        self.physics_engine = None

        for state in self.agent.environment.states:
            if self.agent.environment.states[state] == '_':
                sprite = arcade.Sprite(":resources:images/tiles/dirtCenter.png", 0.5)
                sprite.center_x = state[1] * sprite.width + sprite.width * 0.5
                sprite.center_y = self.height - (state[0] * sprite.width + sprite.width * 0.5)
                self.grounds.append(sprite)

            elif self.agent.environment.states[state] == 'c':
                sprite = arcade.Sprite(":resources:images/items/keyYellow.png", 0.5)
                sprite.center_x = state[1] * sprite.width + sprite.width * 0.5
                sprite.center_y = self.height - (state[0] * sprite.width + sprite.width * 0.5)
                self.keys.append(sprite)

            elif self.agent.environment.states[state] == '*':
                sprite = arcade.Sprite(":resources:images/tiles/signExit.png", 0.5)
                sprite.center_x = state[1] * sprite.width + sprite.width * 0.5
                sprite.center_y = self.height - (state[0] * sprite.width + sprite.width * 0.5)
                self.exit.append(sprite)

            elif self.agent.environment.states[state] == '#':
                sprite = arcade.Sprite(":resources:images/tiles/ladderMid.png", 0.5)
                sprite.center_x = state[1] * sprite.width + sprite.width * 0.5
                sprite.center_y = self.height - (state[0] * sprite.width + sprite.width * 0.5)
                self.ladders.append(sprite)

        self.player = arcade.Sprite(":resources:images/animated_characters/female_person/femalePerson_idle.png", 0.5)

        self.physics_engine = arcade.PhysicsEnginePlatformer(self.player,
                                                             self.grounds,
                                                             gravity_constant=1.5,
                                                             ladders=self.ladders)
        self.update_player()

    def update_player(self):
        self.player.center_x = (self.agent.state[1] * self.player.height) + self.player.height * 0.5
        self.player.center_y = (self.height - (self.agent.state[0] * self.player.height)
                                - self.player.height * 0.5)

    def on_update(self, delta_time):
        self.physics_engine.update()
        if not self.agent.environment.map_is_done() \
                or self.agent.state != self.agent.environment.exit:
            action = self.agent.best_action()
            self.agent.do(action)
            self.agent.update_policy()
            self.update_player()
        else:
            self.agent.reset()
            self.setup()

        key_hit_list = arcade.check_for_collision_with_list(self.player,
                                                            self.keys)

        for key in key_hit_list:
            key.remove_from_sprite_lists()
            arcade.play_sound(self.collect_key_sound, volume=0.01)

    def on_draw(self):
        arcade.start_render()
        self.grounds.draw()
        self.ladders.draw()
        self.keys.draw()
        self.player.draw()
        self.exit.draw()
        arcade.draw_text(f"Score: {self.agent.score}", 10, 10, arcade.csscolor.WHITE, 20)

    def on_key_press(self, key, modifiers):
        if key == arcade.key.R:
            self.agent.reset()
            self.setup()

    def is_on_the_right_edges(self):
        return self.player.center_x > (SPRITE_SIZE * SCREEN_GRID_TILE_WIDTH - SPRITE_SIZE / 2)

    def is_on_the_left_edges(self):
        return self.player.center_x < (SPRITE_SIZE / 2)


def main():
    """ Main method """
    env = Environment(MAZE)
    agent = Agent(env)
    game = MyGame(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE, agent)
    game.setup()
    arcade.run()


if __name__ == "__main__":
    main()
