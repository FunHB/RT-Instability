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
    # ('cell',     np.uint32,  (2,)),  # uint2   (8 bytes)
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
    positions: SharedBuffer
    size: int

    def __init__(self, bounds: Tuple[float, float, float, float], sigma: float):
        self.bounds = bounds
        self.sigma = sigma

        self.positions = SharedBuffer(dtype=POSITION_DTYPE)

        self.bufferSize = PARTICLE_DTYPE.itemsize
        self.Set(np.zeros(0, dtype=PARTICLE_DTYPE))

    def Get(self) -> NDArray:
        return self.particles
    
    def Set(self, particles: NDArray):
        self.particles = particles.ravel()
        self.size = int(self.particles.shape[0])

        self.ExtractProperties()

        self.positions.Resize((self.size,))
        self.SetPositions(self.particles['position'])

    def GetPositions(self) -> NDArray:
        return self.positions.GetFrontBuffer()['position']

    def SetPositions(self, positions: NDArray):
        self.positions.WriteBackAndSwap(positions, field='position')

    def FromBuffer(self, buffer: Buffer) -> NDArray:
        return np.frombuffer(buffer, dtype=PARTICLE_DTYPE)

    def ExtractProperties(self) -> None:
        self.mass = self.particles['mass']
        self.id = self.particles['id']

    def GetPositionsOnGrid(self, numberOfParticles: int, position: Tuple[float, float], size: Tuple[float, float]) -> NDArray:
        x, y = position
        columns, rows = size
        gridResolution = 2**(1/6) * self.sigma
        indices = np.arange(numberOfParticles, dtype=np.int32)

        columnIndices = indices % columns
        rowIndices = indices // columns

        xOnGrid = x + gridResolution * (columnIndices - 0.5 * (columns - 1))
        yOnGrid = y + gridResolution * (rowIndices - 0.5 * (rows - 1))

        return np.stack((xOnGrid, yOnGrid), axis=-1).astype(np.float32)

    def CreateRectangle(self, mass: float, position: Tuple[float, float], size: Tuple[float, float], velocity: Tuple[float, float], id: int) -> NDArray:
        columns, rows = size
        numberOfParticles = int(columns * rows)

        particles = np.zeros(numberOfParticles, dtype=PARTICLE_DTYPE)
        particles['mass'] = mass
        particles['velocity'] = velocity
        particles['id'] = id
        particles['position'] = self.GetPositionsOnGrid(numberOfParticles, position, size)

        return particles

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
