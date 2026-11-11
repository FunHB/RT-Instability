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
    ('sigma',    PARTICLE_PROPERTY_TYPE),      # float  (4 bytes)
    ('epsilon',  PARTICLE_PROPERTY_TYPE),      # float  (4 bytes)
    # --------------------------------------------------
    # Hidden Compiler Padding:       (4 bytes added automatically to align struct size to 8)
], align=True)

POSITION_DTYPE = np.dtype([
    ('position', np.float32, (2,)),  # float2 (8 bytes)
    # --------------------------------------------------
    # Hidden Compiler Padding:       (4 bytes added automatically to align struct size to 8)
], align=True)

assert PARTICLE_DTYPE.itemsize == 40, f'Unexpected GPU particle stride: {PARTICLE_DTYPE.itemsize}'

class Particles:
    particles: NDArray
    positions: SharedBuffer
    size: int

    def __init__(self, bounds: Tuple[float, float, float, float], maxSigma: float = 1):
        self.bounds = bounds
        self.maxSigma = maxSigma

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

    def AddParticles(self, positions, mass: float = 1, sigma: float = 1, epsilon: float = 0.1, id: int = 0, velocities: NDArray | None = None):
        n = len(positions)
        particles = np.zeros(n, dtype=PARTICLE_DTYPE)

        particles['position'] = positions
        particles['mass'] = mass
        particles['sigma'] = sigma
        particles['epsilon'] = epsilon
        particles['id'] = id

        if velocities is None:
            particles['velocity'] = 0.0
        else:
            particles['velocity'] = velocities

        particles['force'] = 0.0

        if self.size:
            particles = np.concatenate((self.Get(), particles))

        self.Set(particles)

    def GetVelocities(self, numberOfParticles: int, mass: float):
        velocities = np.random.normal(0.0, np.sqrt(60.0 / mass), size=(numberOfParticles, 2))
        return velocities

    def InitializeVelocities(self):
        masses = self.particles['mass'].astype(np.float32)
        n = self.size

        self.particles['velocity'] = np.random.normal(0.0, np.sqrt(60.0 / masses)[:, None], size=(n, 2)).astype(np.float32)

        energy = 0.5 * np.sum(masses[:, None] * self.particles['velocity']**2)
        self.particles['velocity'] *= np.sqrt((60.0 * n) / energy)

    @staticmethod
    def GridRegion(xmin: float, xmax: float, ymin: float, ymax: float, spacing: float, count=None):
        x = np.arange(xmin, xmax + 1e-6, spacing)
        y = np.arange(ymin, ymax + 1e-6, spacing)

        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack((xx.ravel(), yy.ravel()))

        if count is not None:
            positions = positions[:count]

        return positions.astype(np.float32)

    @staticmethod
    def CircleRegion(center, radius, spacing):
        positions = Particles.GridRegion(center[0] - radius, center[0] + radius, center[1] - radius, center[1] + radius, spacing)

        dx = positions[:, 0] - center[0]
        dy = positions[:, 1] - center[1]

        return positions[dx**2 + dy**2 <= radius**2]