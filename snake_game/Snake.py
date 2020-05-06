import pygame


class Snake:
    def __init__(self, x=20, y=20):
        self.pixel = 10
        self.color_head = (0, 200, 0)
        self.color_body = (50, 200, 50)

        self.snake = [[x, y]]
        for i in range(3):
            self.snake.append([x, y + 1])

    def move(self, move):
        # 0 - up
        # 1 - right
        # 2 - down
        # 3 - left
        head = self.snake[0]
        if move == 0:
            head[1] += 1
        elif move == 1:
            head[0] += 1
        elif move == 2:
            head[1] -= 1
        elif move == 3:
            head[0] += 1
        self.snake.insert(0, head)
        self.snake.pop()

    def render(self, display):
        for i, point in enumerate(self.snake):
            if i == 0:
                pygame.draw.rect(display, self.color_head,
                                 [self.pixel * point[0], self.pixel * point[1], self.pixel, self.pixel])
            else:
                pygame.draw.rect(display, self.color_body,
                                 [self.pixel * point[0], self.pixel * point[1], self.pixel, self.pixel])
