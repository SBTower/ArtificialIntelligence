import copy
import numpy as np
import scipy
from Environment import Environment
from vizdoom import *

class DoomEnvironment(Environment):

  def __init__(self, name = 'Doom', maxEpisodeLength = None):
    self.name = name
    self.env = DoomGame()
    self.env.set_doom_scenario_path("basic.wad")
    self.env.set_doom_map("map01")
    self.env.set_screen_resolution(ScreenResolution.RES_160X120)
    self.env.set_screen_format(ScreenFormat.GRAY8)
    self.env.set_render_hud(False)
    self.env.set_render_crosshair(False)
    self.env.set_render_weapon(True)
    self.env.set_render_decals(False)
    self.env.set_render_particles(False)
    self.env.add_available_button(Button.MOVE_LEFT)
    self.env.add_available_button(Button.MOVE_RIGHT)
    self.env.add_available_button(Button.ATTACK)
    self.env.add_available_game_variable(GameVariable.AMMO2)
    self.env.add_available_game_variable(GameVariable.POSITION_X)
    self.env.add_available_game_variable(GameVariable.POSITION_Y)
    self.env.set_episode_timeout(300)
    self.env.set_episode_start_time(10)
    self.env.set_window_visible(False)
    self.env.set_sound_enabled(False)
    self.env.set_living_reward(-1)
    self.env.set_mode(Mode.PLAYER)
    self.env.init()
    state = self.env.get_state().screen_buffer
    self.state = self.enumerateState(state)
    self.done = False
    self.count = 0
    self.maxEpisodeLength = maxEpisodeLength

  def getPossibleActions(self):
    return [[True,False,False],[False,True,False],[False,False,True]]

  def update(self, action):
    self.reward = self.env.make_action(self.getPossibleActions()[action]) / 100.0
    self.done = self.env.is_episode_finished()
    if self.done is False:
      new_state = self.env.get_state().screen_buffer
      self.state = self.enumerateState(new_state)
    self.count = self.count + 1
    if self.count > self.maxEpisodeLength:
      self.done = True

  def reset(self):
    self.env.new_episode()
    state = self.env.get_state().screen_buffer
    self.state = self.enumerateState(state)
    self.count = 0
    self.done = False

  def enumerateState(self, state):
    s = state[10:-10,30:-30]
    s = scipy.misc.imresize(s,[84,84])
    s = np.reshape(s,[np.prod(s.shape)]) / 255.0
    return s

  def getActionSize(self):
    return 3

  def getStateSize(self):
    return 7056

  def render(self):
    pass

  def isActionContinuous(self):
    return False
