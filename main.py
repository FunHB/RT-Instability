import sys
import time
import traceback
from threading import Event, Thread

from matplotlib.pylab import spacing
import numpy as np

from Types import Particles
from computeEngine import ComputeEngine
from renderEngine import RenderEngine

def calculate_lattice(n, width, height):
    target_aspect = width / height
    best = None

    for rows in range(2, int(np.sqrt(n)) + 1):
        if n % rows != 0:
            continue

        columns = n // rows

        if columns < 2:
            continue

        spacing_x = width / (columns - 1)
        spacing_y = height / (rows - 1)
        spacing = min(spacing_x, spacing_y)

        aspect_error = abs(
            np.log((columns / rows) / target_aspect)
        )

        if best is None or aspect_error < best[0]:
            best = (aspect_error, spacing, columns, rows)

    if best is None:
        raise ValueError(f"Cannot create a 2D lattice for {n} particles")

    _, spacing, columns, rows = best
    return spacing, columns, rows

if __name__ == "__main__":
    n = 6384
    dt = 0.0005

    m_up = 2
    sigma_up = 0.9412

    m_down = 1
    sigma_down = 1

    epsilon = 1
    gravity = (0, -12.44)

    a = 140
    b = 37.5
    xmin, xmax, ymin, ymax = -.5 * a, .5 * a, -.5 * b, .5 * b
    interface_y = 0.0

    region_width = xmax - xmin
    region_height = (ymax - ymin) / 2.0
    particles_per_fluid = n // 2

    spacing, columns, rows = calculate_lattice(particles_per_fluid, xmax - xmin, (ymax - ymin) / 2)

    particles = Particles((xmin, xmax, ymin, ymax), maxSigma=max(sigma_up, sigma_down))

    lower = Particles.GridRegion(xmin, xmax, ymin, -spacing / 2, spacing, count=3192)
    upper = Particles.GridRegion(xmin, xmax, spacing / 2, ymax, spacing, count=3192)

    particles.AddParticles(lower, mass=m_down, sigma=sigma_down, epsilon=epsilon, id=2)
    particles.AddParticles(upper, mass=m_up, sigma=sigma_up, epsilon=epsilon, id=1)

    positions = particles.Get()['position']

    particles.InitializeVelocities()

    shutdownEvent = Event()

    def Dispatch():
        nextStep = time.perf_counter()

        while not shutdownEvent.is_set():
            computeEngine.Dispatch()

            nextStep += dt

            sleepTime = nextStep - time.perf_counter()

            if sleepTime > 0:
                time.sleep(sleepTime)

    computeEngine = ComputeEngine(particles, dt, gravity)
    computeThread = Thread(target=Dispatch, daemon=True)
    computeThread.start()

    renderEngine = RenderEngine(particles, (xmin, xmax, ymin, ymax), computeEngine, print=False)
    renderEngine.RenderLoop()

    shutdownEvent.set()
    computeThread.join()