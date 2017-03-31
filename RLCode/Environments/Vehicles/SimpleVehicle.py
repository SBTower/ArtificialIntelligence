
import math
import numpy as np

class SimpleVehicle:

  def __init__(self, pos = None, width = 15, length = 30, sensorRange = 100):
    self.sensorRange = 100
    self.width = width
    self.length = length
    if pos is None:
      self.pos = [2*length, 200, math.pi/2]
    else:
      self.pos = pos

    self.speed = 0
    self.acceleration = 0
    self.maxSpeed = 5
    
    self.angularVelocity = 0
    self.angularAcceleration = 0
    self.maxAngularVelocity = 0.1

    self.updateOutline()

  def updateOutline(self):
    x_mid_1 = self.pos[0] - (self.length / 2) * math.sin(self.pos[2])
    y_mid_1 = self.pos[1] - (self.length / 2) * math.cos(self.pos[2])

    x_mid_2 = self.pos[0] + (self.length / 2) * math.sin(self.pos[2])
    y_mid_2 = self.pos[1] + (self.length / 2) * math.cos(self.pos[2])

    x1 = x_mid_1 - (self.width / 2) * math.cos(self.pos[2])
    y1 = y_mid_1 + (self.width / 2) * math.sin(self.pos[2])

    x2 = x_mid_2 - (self.width / 2) * math.cos(self.pos[2])
    y2 = y_mid_2 + (self.width / 2) * math.sin(self.pos[2])

    x3 = x_mid_2 + (self.width / 2) * math.cos(self.pos[2])
    y3 = y_mid_2 - (self.width / 2) * math.sin(self.pos[2])

    x4 = x_mid_1 + (self.width / 2) * math.cos(self.pos[2])
    y4 = y_mid_1 - (self.width / 2) * math.sin(self.pos[2])

    self.outline = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

  def updatePosition(self, time):
    self.angularVelocity = self.angularVelocity + self.angularAcceleration * time
    if abs(self.angularVelocity) > self.maxAngularVelocity:
      self.angularVelocity = math.copysign(self.maxAngularVelocity, self.angularVelocity)

    self.pos[2] = self.pos[2] + self.angularVelocity * time

    if self.pos[2] > math.pi:
      self.pos[2] = self.pos[2] - 2*math.pi
    elif self.pos[2] < -1*math.pi:
      self.pos[2] = self.pos[2] + 2*math.pi

    self.speed = self.speed + self.acceleration * time
    if abs(self.speed) > self.maxSpeed:
      self.speed = math.copysign(self.maxSpeed, self.speed)

    self.pos[0] = self.pos[0] + self.speed * time * math.sin(self.pos[2])
    self.pos[1] = self.pos[1] + self.speed * time * math.cos(self.pos[2])

    self.updateOutline()

  def changeAcceleration(self, acceleration):
    self.acceleration = acceleration

  def changeAngularAcceleration(self, angularAcceleration):
    self.angularAcceleration = angularAcceleration
