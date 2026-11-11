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
    epsilon = 5
    gravity = (0, -12)

    a = 250
    b = 100
    bounds = (-.5 * a, .5 * a, -.5 * b, .5 * b)

    particles = Particles(bounds)

    r = 12.5
    ballPos = (0, .25 * b)

    basinPos = (0, -(1 - .6) * b)

    particles.AddShape(position = ballPos, size = (2 * int(r + 3), 2 * int(r + 3)), mass = 2, sigma = sigma, epsilon = epsilon, f = Particles.Circle(r, ballPos), id = 1)
    particles.AddShape(position = basinPos, size = (a, int(-2 * basinPos[1])), mass = 1, sigma = sigma, epsilon = epsilon, id = 2)

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

    renderEngine = RenderEngine(particles, bounds, computeEngine, print=False)
    renderEngine.RenderLoop()

    shutdownEvent.set()
    computeThread.join()