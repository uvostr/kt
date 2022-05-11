import pygame
import sys
import numpy as np

from python_verlet import sol_verlet

m = np.array([1989000.00000e24, 0.32868e24, 4.81068e24, 0.63345e24, 5.97600e24, 1876.64328e24, 561.80376e24, 86.05440e24, 101.59200e24], np.double)
distance_from_sun = np.array([0, 58e9, 108e9, 150e9, 228e9, 778e9, 1429e9, 2875e9, 4497e9], np.double) 
planet_v = np.array([0, 47.36, 35.02, 29.78, 24.13, 13.07, 9.69, 6.81, 5.43], np.double) * 1e3

N = m.shape[0]

T = 1e8
k = 1000

r0 = np.zeros((N, 2), np.double)
r0[:, 0] = distance_from_sun

v0 = np.zeros((N, 2), np.double)
v0[:, 1] = planet_v

distance_for_draw = np.zeros((N - 1, 2), np.int64)

r, v = sol_verlet(m, r0, v0, T, k)


SPACE_COLOR = pygame.Color("#000022")
SUN_COLOR = pygame.Color("yellow")
PLANET_COLOR = pygame.Color("blue")

SCREEN_SIZE = WIDTH, HEIGHT = (800, 800)
PLANET_RADIUS = 3
SUN_RADIUS = 5

X0 = WIDTH // 2
Y0 = HEIGHT // 2
cur_t = 0

pygame.init()
screen = pygame.display.set_mode(SCREEN_SIZE)
pygame.display.set_caption('Solar System')
fps = pygame.time.Clock()
paused = False

def update():
    global distance_for_draw, cur_t
    if cur_t < k:
        distance_for_draw[:, 0] = (np.sign(r[1:, 0, cur_t]) * abs(r[1:, 0, cur_t] / 1e9) ** 0.7).astype(np.int32)
        distance_for_draw[:, 1]  = (np.sign(r[1:, 1, cur_t]) * abs(r[1:, 1, cur_t] / 1e9) ** 0.7).astype(np.int32)
        cur_t += 1

def render():
    screen.fill(SPACE_COLOR)
    pygame.draw.circle(screen, SUN_COLOR, (X0, Y0), SUN_RADIUS, 0)
    for i in range(N - 1):
        pygame.draw.circle(screen, PLANET_COLOR, (X0 + distance_for_draw[i, 0], Y0 + distance_for_draw[i, 1]), PLANET_RADIUS, 0)
    pygame.display.update()
    fps.tick(30)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_SPACE:
                paused = not paused
    if not paused:
        update()
        render()