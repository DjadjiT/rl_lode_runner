import math
from typing import Optional
import arcade

# How big are our image tiles?
SPRITE_IMAGE_SIZE = 128

PLAYER_MOVEMENT_SPEED = 5
ENEMY_MOVEMENT_SPEED = 3

# Scale sprites up or down
SPRITE_SCALING_PLAYER = 0.5
SPRITE_SCALING_TILES = 0.5

# Scaled sprite size for tiles
SPRITE_SIZE = int(SPRITE_IMAGE_SIZE * SPRITE_SCALING_PLAYER)

# Size of grid to show on screen, in number of tiles
SCREEN_GRID_WIDTH = 15
SCREEN_GRID_HEIGHT = 8

# Size of screen to show, in pixels
SCREEN_WIDTH = SPRITE_SIZE * SCREEN_GRID_WIDTH
SCREEN_HEIGHT = SPRITE_SIZE * SCREEN_GRID_HEIGHT
SCREEN_TITLE = "Lode Runner RL"


class Environment:
    def __init__(self, text):
        self.states = {}
        lines = text.strip().split('\n')

        self.height = len(lines)
        self.width = len(lines[0])
        self.wall = []
        self.ladder = []
        self.ground = []


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
        print(SPRITE_SIZE * SCREEN_GRID_WIDTH - SPRITE_SIZE / 2)
        print(SPRITE_SIZE / 2)

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
        print(self.player_sprite.center_x)

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
        return self.player_sprite.center_x > (SPRITE_SIZE * SCREEN_GRID_WIDTH - SPRITE_SIZE / 2)

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
    game = MyGame(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
    game.setup()
    arcade.run()


if __name__ == "__main__":
    main()
