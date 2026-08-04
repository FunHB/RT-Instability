from typing import Callable, Tuple
from typing_extensions import Buffer
import numpy as np

from numpy.typing import NDArray

from .SharedBuffer import SharedBuffer

PARTICLE_PROPERTY_TYPE = np.float32
PARTICLE_DTYPE = np.dtype([
    ('mass',     PARTICLE_PROPERTY_TYPE),      # float  (4 bytes)
    ('position', PARTICLE_PROPERTY_TYPE, (2,)),  # float2 (8 bytes)
    ('velocity', PARTICLE_PROPERTY_TYPE, (2,)),  # float2 (8 bytes)
    ('force',    PARTICLE_PROPERTY_TYPE, (2,)),  # float2 (8 bytes)
    ('cell',     np.uint32,  (2,)),  # uint2   (8 bytes)
    ('id',       np.uint32),        # uint    (4 bytes)
    # --------------------------------------------------
    # Hidden Compiler Padding:       (4 bytes added automatically to align struct size to 8)
], align=True)

POSITION_DTYPE = np.dtype([
    ('position', np.float32, (2,)),  # float2 (8 bytes)
    # --------------------------------------------------
    # Hidden Compiler Padding:       (4 bytes added automatically to align struct size to 8)
], align=True)

class Particles:
    particles: NDArray
    size: int

    def __init__(self, bounds: Tuple[float, float, float, float], sigma: float):
        self.bounds = bounds
        self.sigma = sigma

        self.bufferSize = PARTICLE_DTYPE.itemsize
        self.Set(np.empty((0, 10), dtype=PARTICLE_DTYPE))

        # self.Set(np.empty((0, 10), dtype=PARTICLE_DTYPE))

    def Get(self) -> NDArray:
        return self.particles

    def Set(self, particles: NDArray):
        self.particles = particles
        self.size = int(particles.shape[0])
        self.ExtractProperties()

    # def Set(self, particles: NDArray):
    #     self.positions.Resize((particles.shape[0], 2))
    #     backBuffer = self.positions.GetBackBuffer()
    #     backBuffer[:] = particles

    #     self.size = int(backBuffer.shape[0])
    #     # self.ExtractProperties()

    #     self.positions.SwapBuffers()

    def FromBuffer(self, buffer: Buffer) -> NDArray:
        return np.frombuffer(buffer, dtype=PARTICLE_DTYPE)

    def ExtractProperties(self) -> None:
        # backBuffer = self.positions.GetBackBuffer()
        # self.mass = backBuffer['mass']
        # self.position = backBuffer['position']
        # self.velocity = backBuffer['velocity']
        # self.cell = backBuffer['cell']
        # self.id = backBuffer['id']
        self.mass = self.particles['mass']
        self.id = self.particles['id']

    def GetPositionOnGrid(self, position: Tuple[float, float], size: Tuple[float, float], index: float) -> Tuple[float, float]:
        x, y = position
        columns, rows = size
        gridResolution = 2**(1/6) * self.sigma
        return (
            x + gridResolution * ((index % columns) - .5 * (columns - 1)),
            y + gridResolution * (index // columns - .5 * (rows - 1))
        )

    def CreateRectangle(self, mass: float, position: Tuple[float, float], size: Tuple[float, float], velocity: Tuple[float, float], id: int) -> NDArray:
        columns, rows = size
        return np.array([(mass, self.GetPositionOnGrid(position, size, int(index)), velocity, (.0, .0), (0, 0), id) for index in range(int(columns * rows))], dtype=PARTICLE_DTYPE)

    def GetParticlesInBounds(self, particles: NDArray):
        maskX = (particles['position'][:, 0] > self.bounds[0] + 1) & (particles['position'][:, 0] < self.bounds[1] - 1)
        maskY = (particles['position'][:, 1] > self.bounds[2] + 1) & (particles['position'][:, 1] < self.bounds[3] - 1)
        return particles[maskX & maskY]

    def AddShape(self, position: Tuple[float, float], size: Tuple[float, float], mass: float = 1, velocity: Tuple[float, float] = (0, 0), id: int = 0, f: Callable[[NDArray], NDArray[np.bool]] = lambda p: p['mass'] > 0) -> None:
        square = self.CreateRectangle(mass, position, size, velocity, id)
        shape = square[f(square)]
        shape = self.GetParticlesInBounds(shape)

        if self.size > 0:
            shape = np.concatenate((self.Get(), shape))

        self.Set(shape)

    @staticmethod
    def Circle(radius: float, center: Tuple[float, float]) -> Callable[[NDArray], NDArray[np.bool_]]:
        return lambda p: (p['position'][:, 0] - center[0])**2 + (p['position'][:, 1] - center[1])**2 <= radius**2
