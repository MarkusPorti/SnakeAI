import pygame


class Snake:
    def __init__(self, width, height, pixel=20, x=10, y=10):
        self.pixel = pixel
        self.width = width
        self.height = height
        self.color_head = (0, 150, 0)
        self.color_body = (25, 200, 25)

        self.score = 0

        self.direction = 0
        self.snake = [[x, y]]
        for i in range(1, 4):
            self.snake.append([x, y + i])

    def move(self, move):
        # 0 - up
        # 1 - right
        # 2 - down
        # 3 - left
        if (move == self.direction + 1 or
                move == self.direction - 1 or
                move == self.direction + 3 or
                move == self.direction - 3):
            self.direction = move

    def step(self, food):
        head = [self.snake[0][0], self.snake[0][1]]
        if self.direction == 0:
            head[1] -= 1
        elif self.direction == 1:
            head[0] += 1
        elif self.direction == 2:
            head[1] += 1
        elif self.direction == 3:
            head[0] -= 1
        self.snake.insert(0, head)

        if not self.eat(food):
            self.snake.pop()
        else:
            self.score += 1
            return 1

        if self.check_collision():
            return -1
        else:
            return 0

    def eat(self, food):
        return self.snake[0] == food

    def check_collision(self):
        return (self.snake[0][0] == -1 or
                self.snake[0][1] == -1 or
                self.snake[0][0] == self.width or
                self.snake[0][1] == self.height or
                self.snake[0] in self.snake[1:])

    def render(self, display):
        for i, point in enumerate(self.snake):
            if i == 0:
                pygame.draw.rect(display, self.color_head,
                                 [self.pixel * point[0], self.pixel * point[1], self.pixel, self.pixel])
            else:
                pygame.draw.rect(display, self.color_body,
                                 [self.pixel * point[0], self.pixel * point[1], self.pixel, self.pixel])

    def __iter__(self):
        return iter(self.snake)
