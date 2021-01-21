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
SCREEN_GRID_TILE_WIDTH = 15
SCREEN_GRID_TILE_HEIGHT = 8

SCREEN_WIDTH = SPRITE_SIZE * SCREEN_GRID_TILE_WIDTH
SCREEN_HEIGHT = SPRITE_SIZE * SCREEN_GRID_TILE_HEIGHT
SCREEN_TITLE = "Lode Runner RL"

REWARD_GOAL = 60
#REWARD_KEY = 30
REWARD_DEFAULT = -1
REWARD_STUCK = -6
REWARD_FALL = 0
REWARD_IMPOSSIBLE = -60

DEFAULT_LEARNING_RATE = 1
DEFAULT_DISCOUNT_FACTOR = 0.5

UP, DOWN, LEFT, RIGHT = 'U', 'D', 'L', 'R'
ACTIONS = [UP, DOWN, LEFT, RIGHT]

MAZE = """
  --------#    
 __       #___ 
c      c  #    
_  __#__  #    
    c#  ___   _
  _#__        *
 p #         __
_______________ 
"""


class Environment:
    def __init__(self, text):
        self.states = {}
        self.lines = text.strip().split('\n')

        self.height = len(self.lines)
        self.width = len(self.lines[0])
        self.init_map()

    def init_map(self):
        self.starting_point = (None, None)
        self.keys = []
        self.exit = []
        self.keys_taken = 0

        for row in range(self.height):
            for col in range(len(self.lines[row])):
                self.states[(row, col)] = self.lines[row][col]
                if self.lines[row][col] == 'p':
                    self.starting_point = (row, col)
                #elif self.lines[row][col] == '*':
                    #self.exit = (row, col)
                elif self.lines[row][col] == 'c':
                    self.keys.append((row, col))

    def apply(self, state, action):
        new_state = (0, 0)
        if action == UP:
            new_state = (state[0] - 1, state[1])
        elif action == DOWN:
            new_state = (state[0] + 1, state[1])
        elif action == LEFT:
            new_state = (state[0], state[1] - 1)
        elif action == RIGHT:
            new_state = (state[0], state[1] + 1)

        if new_state in self.states:
            # calculer la récompense
            if self.states[new_state] in ['_']:
                reward = REWARD_STUCK
            #elif self.states[new_state] in ['*'] and self.map_is_done():  # Sortie du labyrinthe : grosse récompense

            elif self.states[new_state] in ['c'] and not self.map_is_done():
                self.states[new_state] = " "
                self.keys_taken += 1
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
        self.environment.init_map()
        self.state = self.environment.starting_point
        self.previous_state = self.state
        self.score = 0
        self.environment.keys_taken = 0

    def best_action(self):
        return self.policy.best_action(self.state)

    def do(self, action):
        self.previous_state = self.state
        self.state, self.reward = self.environment.apply(self.state, action)
        self.score += self.reward
        self.last_action = action

    def update_policy(self):
        self.policy.update(self.previous_state, self.state, self.last_action, self.reward)


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

    def __init__(self, width, height, title):
        super().__init__(width, height, title)
        arcade.set_background_color(arcade.color.AMAZON)

        self.grounds_list = None
        self.ladders_list = None
        self.keys_list = None
        self.exit_list = None
        self.player_list = None
        self.enemy_list = None
        self.collect_key_sound = arcade.load_sound(":resources:sounds/coin2.wav")

        self.player_sprite = arcade.Sprite(":resources:images/animated_characters/female_person/femalePerson_idle.png",
                                           0.5)
        self.physics_engine = None

    def setup(self):
        # Read in the tiled map
        map_name = "level_1.tmx"
        my_map = arcade.tilemap.read_tmx(map_name)
        arcade.set_background_color(arcade.color.AMAZON)
        self.grounds_list = arcade.SpriteList()
        self.ladders_list = arcade.SpriteList()
        self.keys_list = arcade.SpriteList()
        self.exit_list = arcade.SpriteList()
        self.enemy_list = arcade.SpriteList()
        self.player_list = arcade.SpriteList()

        self.grounds_list = arcade.tilemap.process_layer(map_object=my_map, layer_name="grounds", scaling=0.5,
                                                         use_spatial_hash=True)
        self.ladders_list = arcade.tilemap.process_layer(map_object=my_map, layer_name="ladders", scaling=0.5,
                                                         use_spatial_hash=True)
        self.keys_list = arcade.tilemap.process_layer(map_object=my_map, layer_name="keys", scaling=0.5,
                                                      use_spatial_hash=True)
        self.exit_list = arcade.tilemap.process_layer(map_object=my_map, layer_name="exit", scaling=0.5,
                                                      use_spatial_hash=True)

        self.physics_engine = arcade.PhysicsEnginePlatformer(self.player_sprite,
                                                             self.grounds_list,
                                                             gravity_constant=1.5,
                                                             ladders=self.ladders_list)

        grid_x = 1
        grid_y = 1
        self.player_sprite.center_x = SPRITE_SIZE * grid_x + SPRITE_SIZE / 2
        self.player_sprite.center_y = SPRITE_SIZE * grid_y + SPRITE_SIZE / 2
        # Add to player sprite list
        self.player_list.append(self.player_sprite)

    def on_draw(self):
        arcade.start_render()
        self.grounds_list.draw()
        self.ladders_list.draw()
        self.keys_list.draw()
        self.exit_list.draw()
        self.player_list.draw()

    def on_update(self, delta_time):
        self.grounds_list.update()
        self.physics_engine.update()
        self.player_list.update()

        key_hit_list = arcade.check_for_collision_with_list(self.player_sprite,
                                                            self.keys_list)

        for key in key_hit_list:
            key.remove_from_sprite_lists()
            arcade.play_sound(self.collect_key_sound, volume=0.08)

        if len(arcade.check_for_collision_with_list(self.player_sprite, self.enemy_list)) > 0:
            self.setup()

        exit_collision = len(arcade.check_for_collision_with_list(self.player_sprite, self.exit_list)) > 0
        if exit_collision and len(self.keys_list) == 0:
            self.setup()
        elif exit_collision and len(self.keys_list) != 0:
            print("level not finished")

    def on_key_press(self, key, modifiers):
        if key == arcade.key.UP or key == arcade.key.W:
            if self.physics_engine.is_on_ladder():
                self.player_sprite.change_y = PLAYER_MOVEMENT_SPEED
        elif key == arcade.key.DOWN or key == arcade.key.S:
            if self.physics_engine.is_on_ladder():
                self.player_sprite.change_y = -PLAYER_MOVEMENT_SPEED
        elif (key == arcade.key.LEFT or key == arcade.key.A) and (not self.is_on_the_left_edges()):
            self.player_sprite.change_x = -PLAYER_MOVEMENT_SPEED
        elif (key == arcade.key.RIGHT or key == arcade.key.D) and (not self.is_on_the_right_edges()):
            self.player_sprite.change_x = PLAYER_MOVEMENT_SPEED
        elif (key == arcade.key.LEFT or key == arcade.key.A) and self.physics_engine.is_on_ladder():
            self.player_sprite.change_x = -PLAYER_MOVEMENT_SPEED
        elif (key == arcade.key.RIGHT or key == arcade.key.D) and self.physics_engine.is_on_ladder():
            self.player_sprite.change_x = PLAYER_MOVEMENT_SPEED

    def on_key_release(self, key, modifiers):
        if key == arcade.key.UP or key == arcade.key.W:
            if self.physics_engine.is_on_ladder():
                self.player_sprite.change_y = 0
        elif key == arcade.key.DOWN or key == arcade.key.S:
            if self.physics_engine.is_on_ladder():
                self.player_sprite.change_y = 0
        elif key == arcade.key.LEFT or key == arcade.key.A:
            self.player_sprite.change_x = 0
        elif key == arcade.key.RIGHT or key == arcade.key.D:
            self.player_sprite.change_x = 0

    def on_mouse_motion(self, x, y, delta_x, delta_y):
        """
        Called whenever the mouse moves.
        """
        pass

    def on_mouse_press(self, x, y, button, key_modifiers):
        """
        Called when the user presses a mouse button.
        """
        pass

    def on_mouse_release(self, x, y, button, key_modifiers):
        """
        Called when a user releases a mouse button.
        """
        pass

    def is_on_the_right_edges(self):
        return self.player_sprite.center_x > (SPRITE_SIZE * SCREEN_GRID_TILE_WIDTH - SPRITE_SIZE / 2)

    def is_on_the_left_edges(self):
        return self.player_sprite.center_x < (SPRITE_SIZE / 2)

    def move_towards_player(self, enemy: arcade.Sprite):
        dx, dy = self.player_sprite.center_x - enemy.center_x, self.player_sprite.center_y - enemy.center_y
        if dx > 0:
            enemy.change_x = ENEMY_MOVEMENT_SPEED
        elif dx < 0:
            enemy.change_x = -ENEMY_MOVEMENT_SPEED


def main():
    """ Main method """
    env = Environment(MAZE)
    agent = Agent(env)

    # Boucle principale
    for i in range(50):
        agent.reset()


        # Tant que l'agent n'est pas sorti du labyrinthe
        step = 1
        while (agent.state != env.exit) and not agent.environment.map_is_done():
            # Choisir la meilleure action de l'agent
            action = agent.best_action()

            # Obtenir le nouvel état de l'agent et sa récompense
            agent.do(action)
            print('#', step, 'ACTION:', action, 'STATE:', agent.previous_state, '->', agent.state, 'SCORE:',
                  agent.score)
            step += 1

            # A partir de St, St+1, at, rt+1, on met à jour la politique (policy, q-table, etc.)
            agent.update_policy()
            # print(agent.policy)
        print('----')
    # game = MyGame(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
    # game.setup()
    # arcade.run()


if __name__ == "__main__":
    main()
