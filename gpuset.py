import os
import pygame
import numpy as np
import pyopencl as cl
import threading
from queue import Queue as ThreadQueue
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
os.environ['PYOPENCL_NO_CACHE'] = '1'

# Constants
WIDTH, HEIGHT = 800, 600
INITIAL_MAX_ITER = 256
ZOOM_FACTOR = 1.2
MIN_RENDER_WIDTH, MIN_RENDER_HEIGHT = 100, 75
MAX_RENDER_WIDTH, MAX_RENDER_HEIGHT = WIDTH, HEIGHT

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Mandelbrot Set Zoom")

# OpenCL setup
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)
program_src = """
__kernel void mandelbrot(
    const int width, const int height,
    const float xmin, const float xmax,
    const float ymin, const float ymax,
    const int max_iter, __global int* output) {

    int gid = get_global_id(0);
    int x = gid % width;
    int y = gid / width;

    float real = xmin + (xmax - xmin) * x / (float) width;
    float imag = ymin + (ymax - ymin) * y / (float) height;
    float c_real = real;
    float c_imag = imag;

    int iter;
    for (iter = 0; iter < max_iter; iter++) {
        float real2 = real * real;
        float imag2 = imag * imag;
        if (real2 + imag2 > 4.0f) break;
        imag = 2.0f * real * imag + c_imag;
        real = real2 - imag2 + c_real;
    }
    output[gid] = iter;
}
"""
program = cl.Program(context, program_src).build()

# Functions to calculate Mandelbrot set using OpenCL
def mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter):
    output = np.empty((height, width), dtype=np.int32)
    output_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, output.nbytes)

    program.mandelbrot(queue, (width * height,), None,
                       np.int32(width), np.int32(height),
                       np.float32(xmin), np.float32(xmax),
                       np.float32(ymin), np.float32(ymax),
                       np.int32(max_iter), output_buffer)

    cl.enqueue_copy(queue, output, output_buffer).wait()
    return output

# Function to draw the Mandelbrot set with colors
def draw_mandelbrot(surface, mandelbrot_image, width, height, max_iter):
    for x in range(width):
        for y in range(height):
            iter_count = mandelbrot_image[y, x]
            if iter_count == max_iter:
                color = (0, 0, 0)
            else:
                color = (iter_count % 256, (iter_count * 5) % 256, (iter_count * 13) % 256)
            surface.set_at((x, y), color)

# Initial boundaries
xmin, xmax = -2.0, 1.0
ymin, ymax = -1.5, 1.5

# Initial resolution and iteration count
RENDER_WIDTH, RENDER_HEIGHT = 800, 600
MAX_ITER = INITIAL_MAX_ITER

# Create a surface for rendering
render_surface = pygame.Surface((RENDER_WIDTH, RENDER_HEIGHT))

# Variables for pre-rendering
next_mandelbrot_image = None
next_render_surface = None
needs_update = False
is_rendering = False

# Render queue setup
render_queue = ThreadQueue()
render_thread = None

def render_worker():
    global next_mandelbrot_image, next_render_surface, is_rendering
    while True:
        task = render_queue.get()
        if task is None:
            break
        print("Pre-rendering started")
        is_rendering = True
        next_mandelbrot_image = mandelbrot_set(xmin, xmax, ymin, ymax, RENDER_WIDTH, RENDER_HEIGHT, MAX_ITER)
        next_render_surface = pygame.Surface((RENDER_WIDTH, RENDER_HEIGHT))
        draw_mandelbrot(next_render_surface, next_mandelbrot_image, RENDER_WIDTH, RENDER_HEIGHT, MAX_ITER)
        is_rendering = False
        print("Pre-rendering finished")
        render_queue.task_done()

def start_new_render():
    global render_thread, needs_update
    if render_thread is None or not render_thread.is_alive():
        render_thread = threading.Thread(target=render_worker)
        render_thread.start()
    render_queue.put(True)
    needs_update = True

def auto_zoom(zoom_factor, mouse_x, mouse_y):
    global xmin, xmax, ymin, ymax, RENDER_WIDTH, RENDER_HEIGHT, MAX_ITER
    # Convert screen coordinates to fractal coordinates
    x_center = xmin + (xmax - xmin) * mouse_x / WIDTH
    y_center = ymin + (ymax - ymin) * mouse_y / HEIGHT
    width = (xmax - xmin) / zoom_factor
    height = (ymax - ymin) / zoom_factor
    xmin, xmax = x_center - width / 2, x_center + width / 2
    ymin, ymax = y_center - height / 2, y_center + height / 2
    RENDER_WIDTH = min(MAX_RENDER_WIDTH, int(RENDER_WIDTH * zoom_factor))
    RENDER_HEIGHT = min(MAX_RENDER_HEIGHT, int(RENDER_HEIGHT * zoom_factor))
    MAX_ITER = min(2048, int(MAX_ITER * zoom_factor))
    return pygame.Surface((RENDER_WIDTH, RENDER_HEIGHT))

# Start pre-rendering the first frame
start_new_render()

# Main loop
running = True
auto_zooming = False
mouse_x, mouse_y = WIDTH // 2, HEIGHT // 2  # Default to center

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                auto_zooming = not auto_zooming
                print("Auto-zooming:", "ON" if auto_zooming else "OFF")
        elif event.type == pygame.MOUSEMOTION:
            mouse_x, mouse_y = event.pos
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and not is_rendering:  # Left mouse button for zoom in
                render_surface = auto_zoom(ZOOM_FACTOR, *event.pos)
                start_new_render()
                print(f"Zoom in at {event.pos}: needs_update set to True")
            elif event.button == 3 and not is_rendering:  # Right mouse button for zoom out
                render_surface = auto_zoom(1/ZOOM_FACTOR, *event.pos)
                start_new_render()
                print(f"Zoom out at {event.pos}: needs_update set to True")
        elif event.type == pygame.MOUSEWHEEL and not is_rendering:
            if event.y > 0:  # Scroll up for zoom in
                render_surface = auto_zoom(ZOOM_FACTOR, mouse_x, mouse_y)
                start_new_render()
                print(f"Scroll up zoom at ({mouse_x}, {mouse_y}): needs_update set to True")
            elif event.y < 0:  # Scroll down for zoom out
                render_surface = auto_zoom(1/ZOOM_FACTOR, mouse_x, mouse_y)
                start_new_render()
                print(f"Scroll down zoom at ({mouse_x}, {mouse_y}): needs_update set to True")

    if auto_zooming and not is_rendering:
        render_surface = auto_zoom(ZOOM_FACTOR, mouse_x, mouse_y)
        start_new_render()
        print(f"Auto zoom at ({mouse_x}, {mouse_y}): needs_update set to True")

    if next_render_surface is not None:
        print("Updating render_surface with pre-rendered surface")
        render_surface = next_render_surface
        next_render_surface = None
        next_mandelbrot_image = None
        needs_update = False
        if not is_rendering and not auto_zooming:
            start_new_render()

    # Scale the render surface to the screen size and blit it
    scaled_surface = pygame.transform.scale(render_surface, (WIDTH, HEIGHT))
    screen.blit(scaled_surface, (0, 0))
    pygame.display.flip()

# Cleanup
render_queue.put(None)
if render_thread:
    render_thread.join()
pygame.quit()