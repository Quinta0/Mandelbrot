import pygame
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Constants
WIDTH, HEIGHT = 800, 600
MAX_ITER = 256
ZOOM_FACTOR = 1.2
RENDER_WIDTH, RENDER_HEIGHT = 400, 300  # Lower resolution for rendering

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Mandelbrot Set Zoom")

# Functions to calculate Mandelbrot set
def mandelbrot(c, max_iter):
    z = c
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z * z + c
    return max_iter

def mandelbrot_row(xmin, xmax, ymin, ymax, width, height, max_iter, y):
    r1 = np.linspace(xmin, xmax, width)
    r2 = np.linspace(ymin, ymax, height)
    row = np.empty(width, dtype=np.int32)
    for i in range(width):
        row[i] = mandelbrot(r1[i] + 1j * r2[y], max_iter)
    return row

def mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter):
    with ThreadPoolExecutor() as executor:
        rows = list(executor.map(lambda y: mandelbrot_row(xmin, xmax, ymin, ymax, width, height, max_iter, y), range(height)))
    return np.array(rows)

def draw_mandelbrot(surface, mandelbrot_image, width, height):
    for x in range(width):
        for y in range(height):
            color_value = mandelbrot_image[y, x] / MAX_ITER * 255
            color = (int(color_value), int(color_value), int(color_value))
            surface.set_at((x, y), color)

# Initial boundaries
xmin, xmax = -2.0, 1.0
ymin, ymax = -1.5, 1.5

# Create a surface for rendering
render_surface = pygame.Surface((RENDER_WIDTH, RENDER_HEIGHT))

# Main loop
running = True
needs_update = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button for zoom in
                mouse_x, mouse_y = event.pos
                mouse_x = mouse_x * RENDER_WIDTH // WIDTH
                mouse_y = mouse_y * RENDER_HEIGHT // HEIGHT
                r1 = np.linspace(xmin, xmax, RENDER_WIDTH)
                r2 = np.linspace(ymin, ymax, RENDER_HEIGHT)
                x_center = r1[mouse_x]
                y_center = r2[mouse_y]
                width = (xmax - xmin) / ZOOM_FACTOR
                height = (ymax - ymin) / ZOOM_FACTOR
                xmin, xmax = x_center - width / 2, x_center + width / 2
                ymin, ymax = y_center - height / 2, y_center + height / 2
                needs_update = True
            elif event.button == 3:  # Right mouse button for zoom out
                mouse_x, mouse_y = event.pos
                mouse_x = mouse_x * RENDER_WIDTH // WIDTH
                mouse_y = mouse_y * RENDER_HEIGHT // HEIGHT
                r1 = np.linspace(xmin, xmax, RENDER_WIDTH)
                r2 = np.linspace(ymin, ymax, RENDER_HEIGHT)
                x_center = r1[mouse_x]
                y_center = r2[mouse_y]
                width = (xmax - xmin) * ZOOM_FACTOR
                height = (ymax - ymin) * ZOOM_FACTOR
                xmin, xmax = x_center - width / 2, x_center + width / 2
                ymin, ymax = y_center - height / 2, y_center + height / 2
                needs_update = True
        elif event.type == pygame.MOUSEWHEEL:
            if event.y > 0:  # Scroll up for zoom in
                x_center = (xmin + xmax) / 2
                y_center = (ymin + ymax) / 2
                width = (xmax - xmin) / ZOOM_FACTOR
                height = (ymax - ymin) / ZOOM_FACTOR
                xmin, xmax = x_center - width / 2, x_center + width / 2
                ymin, ymax = y_center - height / 2, y_center + height / 2
                needs_update = True
            elif event.y < 0:  # Scroll down for zoom out
                x_center = (xmin + xmax) / 2
                y_center = (ymin + ymax) / 2
                width = (xmax - xmin) * ZOOM_FACTOR
                height = (ymax - ymin) * ZOOM_FACTOR
                xmin, xmax = x_center - width / 2, x_center + width / 2
                ymin, ymax = y_center - height / 2, y_center + height / 2
                needs_update = True

    if needs_update:
        mandelbrot_image = mandelbrot_set(xmin, xmax, ymin, ymax, RENDER_WIDTH, RENDER_HEIGHT, MAX_ITER)
        draw_mandelbrot(render_surface, mandelbrot_image, RENDER_WIDTH, RENDER_HEIGHT)
        needs_update = False

    # Scale the render surface to the screen size and blit it
    scaled_surface = pygame.transform.scale(render_surface, (WIDTH, HEIGHT))
    screen.blit(scaled_surface, (0, 0))
    pygame.display.flip()

pygame.quit()
