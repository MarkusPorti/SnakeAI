import pygame
from random import randint
from snake_game.Snake import Snake


class Game:
    def __init__(self, width=400, height=400, gui=False):
        self.clock = pygame.time.Clock()
        self.score = 0
        self.running = True
        self.board = {'width': width, 'height': height}
        self.snake = Snake(width, height)
        self.food = []
        self.food_color = (255, 0, 0)
        self.gui = gui

    def start(self):
        # self.generate_food()
        if self.gui:
            self.render_init()
        while self.running:
            self.clock.tick(3)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.snake.move(0)
                    elif event.key == pygame.K_RIGHT:
                        self.snake.move(1)
                    elif event.key == pygame.K_DOWN:
                        self.snake.move(2)
                    elif event.key == pygame.K_LEFT:
                        self.snake.move(3)

            game.step()

        while not self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

    def render_init(self):
        pygame.init()
        pygame.display.set_caption("Snake")
        self.dis = pygame.display.set_mode((self.board["width"], self.board["height"]))
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

    def step(self):
        if not self.snake.step():
            self.running = False
        if self.gui:
            self.render()


if __name__ == "__main__":
    game = Game(gui=True)
    game.start()
