import pygame
from random import randint
from snake_game.Snake import Snake


class Game:
    def __init__(self, width=400, height=400, gui=False):
        self.clock = pygame.time.Clock()
        self.score = 0
        self.done = False
        self.board = {'width': width, 'height': height}
        self.snake = Snake()
        self.food = []
        self.food_color = (255, 0, 0)
        self.gui = gui

    def start(self):
        # self.generate_food()
        if self.gui:
            self.render_init()
        while True:
            game.step(randint(0, 3))

    def render_init(self):
        pygame.init()
        pygame.display.set_caption("Snake")
        self.dis = pygame.display.set_mode((self.board["width"] + 2, self.board["height"] + 2))
        self.render()

    def render(self):
        self.dis.fill((255, 255, 255))
        # pygame.draw.rect(self.dis, self.food_color, [self.food[0], self.food[1], 10, 10])
        self.snake.render(self.dis)
        pygame.display.update()

    def generate_food(self):
        food = []
        while not food:
            food = [randint(1, self.board["width"]), randint(1, self.board["height"])]
            if food in self.snake:
                food = []
        self.food = food

    def step(self, move):
        self.snake.move(move)
        if self.gui:
            self.render()
            self.clock.tick(3)


if __name__ == "__main__":
    game = Game(gui=True)
    game.start()
