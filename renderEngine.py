import time
from typing import Tuple

import numpy as np

import fastplotlib as fpl
from fastplotlib.layouts._subplot import Subplot

from Types import Particles
from computeEngine import ComputeEngine

type Color = Tuple[float, float, float, float]

COLORS: list[Tuple[int, Color]] = [(1, (.5, .5, .5, .5)), (0, (1.0, 1.0, 1.0, 1.0))]
# COLORS: list[Tuple[int, Color]] = [(2, (1.0, 1.0, 1.0, 1.0)), (3, (0.2, 0.2, 1.0, 1.0))]

CMAP = 'matplotlib:coolwarm'

TIMES = [0.1, 18.5, 19.5, 20.5, 21.5, 23, 29.5]

class RenderEngine:
    def __init__(self, particles: Particles, bounds: Tuple[float, float, float, float], computeEngine: ComputeEngine, print: bool = False):
        self.lastRender = time.perf_counter()
        self.time = time.time()
        self.fps = 30

        self.computeEngine = computeEngine

        self.print = print

        self.particles = particles
        self.bounds = bounds

        self.figure = fpl.Figure(size=(700, 560))
        self.subplot = self.figure[0, 0]

        self.subplot.add_animations(self.Update)
        self.subplot.title = f'Simulation'

        self.scatter = self.figure[0, 0].add_scatter(data=np.zeros((self.particles.size, 3), dtype=np.float32), cmap=CMAP)
        self.scatter.sizes = 5.0

        self.InitializeBounds()

        self.renderPositions = np.empty((self.particles.size, 3), dtype=np.float32)
        self.colors = np.ones(shape=(self.particles.size, 4), dtype=np.float32)

        for i, color in COLORS:
            self.colors[self.particles.id == i] = color

        self.scatter.colors = self.colors

    def InitializeBounds(self):
        lines = list(np.array([
                    [[self.bounds[0], self.bounds[2]], [self.bounds[0], self.bounds[3]]],
                    [[self.bounds[0], self.bounds[2]], [self.bounds[1], self.bounds[2]]],
                    [[self.bounds[1], self.bounds[2]], [self.bounds[1], self.bounds[3]]],
                    [[self.bounds[0], self.bounds[3]], [self.bounds[1], self.bounds[3]]]]).astype(np.float32))
        
        whiteColor = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

        self.subplot.add_line_stack(lines, separation=.0, separation_axis='z', colors=whiteColor)

    def Update(self, subplot: Subplot):
        now = time.perf_counter()
        if now - self.lastRender < 1 / self.fps:
            return
        self.lastRender = now


        positions = self.computeEngine.ReadRenderResults()

        self.renderPositions[:, :2] = positions
        self.scatter.data = self.renderPositions

        # self.scatter.colors = self.colors
        subplot.title = f'Simulation | {time.time() - self.time:.2f} s{f' | {self.computeEngine.time:.2f} s' if self.computeEngine.time > 0 else ''}'

        if self.print:
            if self.computeEngine.time in TIMES:
                print(f'Frame: {self.computeEngine.frame} | Time: {self.computeEngine.time:.2f} s | Particles: {self.particles.size}')
                self.figure.export(f'img/{self.computeEngine.time:.1f}.png', extension='.png')

    def RenderLoop(self):
        self.figure.show()
        fpl.loop.run()