"""Author: Stuart Tower
"""

import copy
import time
import numpy as np
from .Environment import Environment

from math import pi, sin, cos

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from direct.actor.Actor import Actor

class PandaEnvironment(ShowBase):
    """An environment testing the integration with Panda3D.
    Still in progress.
    """
    def __init__(self):
        ShowBase.__init__(self)
        self.Actors = []
        self.Models = []
        self.done = False
        self.state = 0

    def loadActor(self, model_file, texture_file=None, pos=[0,0,0], scale=[1,1,1]):
        new_actor = Actor(model_file)
        if texture_file:
            texture = self.loader.loadTexture(texture_file)
            new_actor.setTexture(texture, 1)
        new_actor.setScale(scale[0], scale[1], scale[2])
        new_actor.setPos(pos[0], pos[1], pos[2])
        new_actor.reparentTo(self.render)
        return new_actor

    def loadModel(self, model_file, texture_file=None, pos=[0,0,0], scale=[1,1,1]):
        new_model = self.loader.loadModel(model_file)
        if texture_file:
            texture = self.loader.loadTexture(texture_file)
            new_model.setTexture(texture, 1)
        new_model.setScale(scale[0], scale[1], scale[2])
        new_model.setPos(pos[0], pos[1], pos[2])
        new_model.reparentTo(self.render)
        return new_model

    def update(self, action):
        pass

    def get_state(self):
        return self.state

    def get_possible_actions(self):
        pass

    def get_reward(self):
        pass

    def check_terminal(self):
        return self.done

    def reset(self):
        self.__init__()

    def enumerate_state(self, state):
        return state

    def get_action_size(self):
        return len(self.get_possible_actions())

    def get_state_size(self):
        return len(self.get_state())

    def animate(self):
        pass

    def is_action_continuous(self):
        return False

    def save_figure(self):
        pass


class PandaMaze(PandaEnvironment):
    def __init__(self, name='PandaMaze', show_visuals=False):
        PandaEnvironment.__init__(self)
        self.name = name
        self.show_visuals = show_visuals
        self.Models.append(self.loadModel("Environments/Models/oldWall", scale=[0.25,0.25,0.25],pos=[0,1,0]))
        for model in self.Models:
            model.reparentTo(self.render)
        self.goal = [3,3]
        self.actor = self.loadActor("models/panda-model", pos=[0,0,0], scale=[0.0005,0.0005,0.0005])
        base.disableMouse()
        self.camera.setPos(0,0,15)
        self.camera.setHpr(0,-90,0)
        self.actor.reparentTo(self.render)
        self.num_episodes = 0

    def get_state(self):
        return self.enumerate_state(self.actor.getPos())

    def get_reward(self):
        if self.check_terminal():
            reward = 0
        else:
            reward = -1
        return reward

    def inWall(self):
        return False

    def update(self, action):
        previous_position = copy.copy(self.actor.getPos())
        if action == 0:
            self.actor.setPos(self.actor.getPos().getX() + 1, self.actor.getPos().getY(), self.actor.getPos().getZ())
        elif action == 1:
            self.actor.setPos(self.actor.getPos().getX(), self.actor.getPos().getY() + 1, self.actor.getPos().getZ())
        elif action == 2:
            self.actor.setPos(self.actor.getPos().getX() - 1, self.actor.getPos().getY(), self.actor.getPos().getZ())
        elif action == 3:
            self.actor.setPos(self.actor.getPos().getX(), self.actor.getPos().getY() - 1, self.actor.getPos().getZ())
        if self.inWall == 1:
            self.actor.setPos(previous_position)
        print(self.actor.getPos())
        time.sleep(1)
        taskMgr.step()
        time.sleep(1)

    def enumerate_state(self, state):
        state = (self.actor.getPos().getX() * self.goal[0]) + self.actor.getPos().getY()
        return [state]

    def check_terminal(self):
        if self.actor.getPos().getX() == self.goal[0] and self.actor.getPos().getY() == self.goal[1]:
            return True
        return False

    def reset(self):
        self.destroy()
        self.num_episodes += 1
        self.__init__(self)

    def get_possible_actions(self):
        return [0, 1, 2, 3]
