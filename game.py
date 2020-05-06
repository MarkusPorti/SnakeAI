import pygame
from random import randint


class SnakeGame:
    def __init__(self, width=600, height=400, gui=False):
        self.score = 0
        self.done = False
        self.board = {'width': width, 'height': height}
        self.snake_head_color = (0, 200, 0)
        self.snake_body_color = (50, 200, 50)
        self.food_color = (255, 0, 0)
        self.gui = gui

    def start(self):
        self.snake_init()
        self.generate_food()
        if self.gui:
            self.render_init()
        return self.generate_observations()

    def snake_init(self):
        x = randint(5, self.board['width'] - 5)
        y = randint(5, self.board['height'] - 5)
        self.snake = []
        vertical = randint(0, 1) == 0
        for i in range(3):
            point = [x + i, y] if vertical else [x, y + i]
            self.snake.insert(0, point)

    def generate_food(self):
        food = []
        while food == []:
            food = [randint(1, self.board["width"]), randint(1, self.board["height"])]
            if food in self.snake:
                food = []
        self.food = food

    def render_init(self):
        pygame.init()
        pygame.display.set_caption("Snake")
        self.dis = pygame.display.set_mode((self.board["width"] + 2, self.board["height"] + 2))
        self.render()

    def render(self):
        self.dis.fill((255, 255, 255))
        # self.win.addch(self.food[0], self.food[1], '🍎')
        pygame.draw.rect(self.dis, self.food_color, [self.food[0], self.food[1], 10, 10])
        for i, point in enumerate(self.snake):
            if i == 0:
                pygame.draw.rect(self.dis, self.snake_head_color, [point[0], point[1], 10, 10])
            else:
                pygame.draw.rect(self.dis, self.snake_body_color, [point[0], point[1], 10, 10])
        pygame.display.update()

    def step(self, key):
        # 0 - UP
        # 1 - RIGHT
        # 2 - DOWN
        # 3 - LEFT
        # if self.done == True:
        #     self.end_game()
        self.create_new_point(key)
        if self.food_eaten():
            self.score += 1
            self.generate_food()
        else:
            self.remove_last_point()
        # self.check_collisions()
        if self.gui:
            self.render()
            pygame.time.Clock().tick(30)
        return self.generate_observations()

    def create_new_point(self, key):
        new_point = [self.snake[0][0], self.snake[0][1]]
        if key == 0:
            new_point[0] -= 1
        elif key == 1:
            new_point[1] += 1
        elif key == 2:
            new_point[0] += 1
        elif key == 3:
            new_point[1] -= 1
        self.snake.insert(0, new_point)

    def remove_last_point(self):
        self.snake.pop()

    def food_eaten(self):
        return self.snake[0] == self.food

    def check_collisions(self):
        if (self.snake[0][0] == 0 or
                self.snake[0][0] == self.board["width"] + 1 or
                self.snake[0][1] == 0 or
                self.snake[0][1] == self.board["height"] + 1 or
                self.snake[0] in self.snake[1:-1]):
            self.done = True

    def generate_observations(self):
        return self.done, self.score, self.snake, self.food

    def render_destroy(self):
        pygame.quit()

    def end_game(self):
        if self.gui:
            self.render_destroy()
        raise Exception("Game over")


if __name__ == "__main__":
    game = SnakeGame(gui=True)
    game.start()
    while not game.done:
        game.step(randint(0, 3))
