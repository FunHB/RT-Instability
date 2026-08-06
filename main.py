import sys
import time
import traceback
from threading import Event, Thread

from Types import Particles
from computeEngine import ComputeEngine
from renderEngine import RenderEngine

if __name__ == "__main__":

    dt = 0.0005
    sigma = 1
    epsilon = 1
    G = (0, -12)

    r = 15

    a = 125
    b = 100
    bounds = (-a, a, -b, b)

    particles = Particles(bounds, sigma)

    # particles.AddShape(position = (0, .5 * b), size = (2 * (r + 3), 2*(r + 3)), mass = 1, f = Particles.Circle(r, (0, .5 * b)), id = 1)
    particles.AddShape(position = (0, -70), size = (230, 60), mass = 1, id = 2)

    shutdownEvent = Event()

    def Dispatch():
        nextStep = time.perf_counter()

        while not shutdownEvent.is_set():
            computeEngine.Dispatch()

            nextStep += dt

            sleepTime = nextStep - time.perf_counter()

            if sleepTime > 0:
                time.sleep(sleepTime)

    computeEngine = ComputeEngine(particles, dt, epsilon, G)
    computeThread = Thread(target=Dispatch, daemon=True)
    computeThread.start()

    renderEngine = RenderEngine(particles, bounds, computeEngine, print=False)
    renderEngine.RenderLoop()

    shutdownEvent.set()
    computeThread.join()