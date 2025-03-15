import pygame
from Random_Agent import Random_Agent
from Graphics import Graphics
from Environment import Environment
import matplotlib.pyplot as plt
import torch
import statistics as stat

class Tester:

    def __init__(self, player):
        self.player = player
        self.graphics = Graphics()
        self.env = Environment(surface=self.graphics.main_surf)

    def test (self, epochs = 100, show=False):
        player = self.player
        self.step = 0
        self.scores = []
        for epoch in range(epochs):
            self.env.restart()
            done = False
            self.moves = 0
            state = self.env.state()        
            if self.env.level == 1:                 # clearing score after logging only when new_game
                self.env.score = 0
            while not done:
                if show:
                    self.graphics.clear()
                    self.graphics.event_pump()
                self.graphics.events()
                action = player.get_Action(state)
                reward, done = self.env.move(action=action)
                self.moves += 1
                state = self.env.state()
                self.graphics.header_writing(env=self.env, epoch=epoch, chkpt=None)
                if show:
                    self.graphics.update()
            print(epoch, self.moves , self.env.score)
            self.scores.append(self.env.score)
            

        pygame.quit()
        return self.scores


    def load (self, path):
        return torch.load(path)
      
if __name__ == "__main__":
    player = Random_Agent()
    tester = Tester(player)
    scores = tester.test(epochs=100, show=True)
    torch.save(scores, "Data/tester3")
    # scores = tester.load("Data/tester1")
    print(scores)
    print("mean", stat.mean(scores))
    
    plt.plot(scores)
    plt.show()

