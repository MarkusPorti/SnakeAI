import pygame
from random import randint
from snake_game.Snake import Snake


class Game:
    def __init__(self, width=20, height=20, gui=False):
        self.clock = pygame.time.Clock()
        self.score = 0
        self.running = True
        self.pixel = 20
        self.board = {'width': width, 'height': height}
        self.snake = Snake(width, height, self.pixel)
        self.food = []
        self.food_color = (255, 0, 0)
        self.gui = gui

    def start(self):
        self.generate_food()
        if self.gui:
            self.render_init()
        while self.running:
            # Framerate
            self.clock.tick(10)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.snake.move(0)
                    elif event.key == pygame.K_RIGHT:
                        self.snake.move(1)
                    elif event.key == pygame.K_DOWN:
                        self.snake.move(2)
                    elif event.key == pygame.K_LEFT:
                        self.snake.move(3)
                    break

            game.step()

        while not self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

    def render_init(self):
        pygame.init()
        pygame.display.set_caption("Snake")
        self.dis = pygame.display.set_mode((self.board["width"] * self.pixel, self.board["height"] * self.pixel))
        self.render()

    def render(self):
        self.dis.fill((255, 255, 255))
        pygame.draw.rect(self.dis, self.food_color,
                         [self.food[0] * self.pixel, self.food[1] * self.pixel, self.pixel, self.pixel])
        self.snake.render(self.dis)
        pygame.display.update()

    def step(self):
        result = self.snake.step(self.food)
        if result == -1:
            self.running = False
        elif result == 1:
            self.generate_food()
        if self.gui:
            self.render()

    def generate_food(self):
        food = []
        while food == []:
            food = [randint(0, self.board["width"]-1), randint(0, self.board["height"]-1)]
            if food in self.snake:
                food = []
        self.food = food


if __name__ == "__main__":
    game = Game(gui=True)
    game.start()
