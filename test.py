import numpy as np

if __name__ == "__main__":
    x = np.zeros(14*3).reshape((14, 3))
    x[:,0] = 5
    print(x)

    # pygame.init()
    # dis = pygame.display.set_mode((18 * 20, 18 * 20))
    # pygame.draw.rect(dis, (255, 0, 0),
    #                  [17 * 20, 5 * 20, 20, 20])
    # pygame.display.update()
    # while True:
    #     for i in range(3):
    #         i += 1