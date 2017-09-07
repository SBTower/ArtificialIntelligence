import math


class SimpleVehicle:
    def __init__(self, pos=None, width=15, length=30, sensor_range=100):
        self.sensor_range = sensor_range
        self.width = width
        self.length = length
        if pos is None:
            self.pos = [2 * length, 200, math.pi / 2]
        else:
            self.pos = pos

        self.speed = 0
        self.acceleration = 0
        self.maxSpeed = 5

        self.angularVelocity = 0
        self.angularAcceleration = 0
        self.maxAngularVelocity = 0.1

        self.outline = self.update_outline()

    def update_outline(self):
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

        outline = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        return outline

    def update_position(self, time):
        self.angularVelocity += self.angularAcceleration * time
        if abs(self.angularVelocity) > self.maxAngularVelocity:
            self.angularVelocity = math.copysign(self.maxAngularVelocity, self.angularVelocity)

        self.pos[2] += self.angularVelocity * time

        if self.pos[2] > math.pi:
            self.pos[2] -= 2 * math.pi
        elif self.pos[2] < -1 * math.pi:
            self.pos[2] += 2 * math.pi

        self.speed += self.acceleration * time
        if abs(self.speed) > self.maxSpeed:
            self.speed = math.copysign(self.maxSpeed, self.speed)

        self.pos[0] += self.speed * time * math.sin(self.pos[2])
        self.pos[1] += self.speed * time * math.cos(self.pos[2])

        self.outline = self.update_outline()

    def change_acceleration(self, acceleration):
        self.acceleration = acceleration

    def change_angular_acceleration(self, angular_acceleration):
        self.angularAcceleration = angular_acceleration

    def reset_position(self, pos):
        if pos is None:
            self.pos = [2 * self.length, 200, math.pi / 2]
        else:
            self.pos = pos
        self.outline = self.update_outline()