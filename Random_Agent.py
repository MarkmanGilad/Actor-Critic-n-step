import pygame
import random

class Random_Agent:
    def __init__ (self):
        self.action = 0

    def get_Action (self, events=None, state= None):
        actions = [0, 1, 2, 3]
        return random.choice(actions)
    
    