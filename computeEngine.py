import struct
import time
from typing import Tuple

import numpy as np

from Types import Particles

import compushady
import compushady.formats
from compushady import Buffer, HEAP_READBACK, HEAP_UPLOAD, Compute
from compushady.shaders import hlsl
from compushady.backends import d3d12

THREAD_NUM = 256

SHADER_DIRECTORY = 'Shaders/'

COMPUTE_FORCES_SHADER_PATH = SHADER_DIRECTORY + 'ComputeForces.hlsl'
COMPUTE_POSITIONS_SHADER_PATH = SHADER_DIRECTORY + 'ComputePositions.hlsl'
COUNT_PARTICLES_SHADER_PATH = SHADER_DIRECTORY + 'CountParticlesPerCell.hlsl'
PREFIX_SUM_SHADER_PATH = SHADER_DIRECTORY + 'PrefixSum.hlsl'
SORT_PARTICLE_INDICES_SHADER_PATH = SHADER_DIRECTORY + 'SortParticleIndices.hlsl'
TIME_SHADER_PATH = SHADER_DIRECTORY + 'Time.hlsl'
CLEAR_SHADER_PATH = SHADER_DIRECTORY + 'ClearShader.hlsl'

def Align256(size):
    return (size + 255) & ~255

class ComputeEngine:
    def __init__(self, particles: Particles, dt: float, epsilon: float, g: Tuple[float, float]):
        self.particles = particles

        self.rCut = 2.5 * self.particles.sigma
        self.epsilon = epsilon
        self.g = np.array(g, dtype=np.float64)

        padding = 1.1
        self.gridSize = np.ceil(
            np.array([
                (self.particles.bounds[1] - self.particles.bounds[0]), 
                (self.particles.bounds[3] - self.particles.bounds[2])
            ]) / (self.rCut * padding)
        ).astype(np.uint32)

        self.time = .0
        self.dt = dt
        self.frame = 0

        self.InitializeConstantBuffers()
        self.InitializeIOBuffers()
        
        self.UpdateParticles()
        
        self.InitializeTimeShader()
        self.InitializeClearShader()
        self.InitializeShader()
        self.InitializeBinningShader()
        
    def InitializeConstantBuffers(self):
        # Config
        self.configBuffer = Buffer(Align256(7 * 4))
        self.configUpload = Buffer(self.configBuffer.size, HEAP_UPLOAD)

        self.configUpload.upload(struct.pack('ifff2i2f', self.particles.size, self.particles.sigma, self.epsilon, self.rCut, *self.gridSize, *self.g))
        self.configUpload.copy_to(self.configBuffer)

        # Bounds
        self.boundsBuffer = Buffer(Align256(4 * 4))
        self.boundsUpload = Buffer(self.boundsBuffer.size, HEAP_UPLOAD)

        self.boundsUpload.upload(struct.pack('4f', *self.particles.bounds))
        self.boundsUpload.copy_to(self.boundsBuffer)

        # Time
        self.timeBuffer = Buffer(2 * 4, stride=4)
        self.timeUpload = Buffer(self.timeBuffer.size, HEAP_UPLOAD)
        self.timeReadBack = Buffer(self.timeBuffer.size, HEAP_READBACK)

        self.timeUpload.upload(struct.pack('ff', self.time, self.dt))
        self.timeUpload.copy_to(self.timeBuffer)

        # Cells Quantity
        self.CellsQuantityBuffer = Buffer(4, format=compushady.formats.R32_UINT)
        self.CellsQuantityUpload = Buffer(self.CellsQuantityBuffer.size, HEAP_UPLOAD)

        self.CellsQuantityUpload.upload(struct.pack('I', self.gridSize[0] * self.gridSize[1]))
        self.CellsQuantityUpload.copy_to(self.CellsQuantityBuffer)

        # # Single Zero Buffer
        # self.singleZeroBuffer = Buffer(4)
        # self.singleZeroUpload = Buffer(self.singleZeroBuffer.size, HEAP_UPLOAD)

        # self.singleZeroUpload.upload(struct.pack('I', 0))
        # self.singleZeroUpload.copy_to(self.singleZeroBuffer)

        # # Zero Buffer
        # self.zeroBuffer = Buffer(4 * (self.gridSize[0] * self.gridSize[1] + 1))
        # self.zeroUpload = Buffer(self.zeroBuffer.size, HEAP_UPLOAD)

        # self.zeroUpload.upload(struct.pack('I', 0) * (self.gridSize[0] * self.gridSize[1] + 1))
        # self.zeroUpload.copy_to(self.zeroBuffer)

    def InitializeIOBuffers(self):
        self.ParticlesBuffer = Buffer(self.particles.bufferSize * self.particles.size, stride=self.particles.bufferSize)
        self.ParticlesUpload = Buffer(self.ParticlesBuffer.size, HEAP_UPLOAD)
        self.ParticlesReadBack = Buffer(self.ParticlesBuffer.size, HEAP_READBACK)

        self.PositionBuffer = Buffer(4 * 2 * self.particles.size, format=compushady.formats.R32G32_FLOAT)
        self.PositionReadBack = Buffer(self.PositionBuffer.size, HEAP_READBACK)

        self.CellOffsetsBuffer = Buffer(4 * (self.gridSize[0] * self.gridSize[1] + 1), format=compushady.formats.R32_UINT)
        self.WorkingCellOffsetsBuffer = Buffer(self.CellOffsetsBuffer.size, format=compushady.formats.R32_UINT)
        self.SortedParticleIndicesBuffer = Buffer(4 * self.particles.size, format=compushady.formats.R32_UINT)

        self.GlobalBlockTrackerBuffer = Buffer(4, format=compushady.formats.R32_UINT)
        self.GlobalBlockTrackerUpload = Buffer(self.GlobalBlockTrackerBuffer.size, HEAP_UPLOAD)
    
    def InitializeShader(self):

        with open(COMPUTE_POSITIONS_SHADER_PATH) as file:
            computePositions = hlsl.compile(file.read())

        self.computePositions = Compute(
            shader=computePositions,
            cbv=[self.configBuffer, self.boundsBuffer],
            srv=[self.timeBuffer],
            uav=[self.ParticlesBuffer, self.PositionBuffer]
        )

        with open(COMPUTE_FORCES_SHADER_PATH) as file:
            computeForces = hlsl.compile(file.read())

        self.computeForces = Compute(
            shader=computeForces,
            cbv=[self.configBuffer, self.boundsBuffer],
            srv=[self.CellOffsetsBuffer, self.SortedParticleIndicesBuffer, self.timeBuffer],
            uav=[self.ParticlesBuffer]
        )

    def InitializeTimeShader(self):
        with open(TIME_SHADER_PATH) as file:
            shader = hlsl.compile(file.read())

        self.updateTime = Compute(
            shader=shader,
            uav=[self.timeBuffer]
        )

    def InitializeClearShader(self):
        with open(CLEAR_SHADER_PATH) as file:
            shader = hlsl.compile(file.read())

        self.clearCellOffsets = Compute(
            shader=shader,
            cbv=[self.configBuffer],
            uav=[self.CellOffsetsBuffer, self.GlobalBlockTrackerBuffer]
        )

    def InitializeBinningShader(self):
        with open(COUNT_PARTICLES_SHADER_PATH) as file:
            shader = hlsl.compile(file.read())

        self.countParticles = Compute(
            shader=shader,
            cbv=[self.configBuffer],
            srv=[self.ParticlesBuffer],
            uav=[self.CellOffsetsBuffer]
        )

        with open(PREFIX_SUM_SHADER_PATH) as file:
            shader = hlsl.compile(file.read())

        self.prefixSum = Compute(
            shader=shader,
            cbv=[self.CellsQuantityBuffer],
            uav=[self.CellOffsetsBuffer, self.GlobalBlockTrackerBuffer]
        )

        with open(SORT_PARTICLE_INDICES_SHADER_PATH) as file:
            shader = hlsl.compile(file.read())

        self.sortParticleIndices = Compute(
            shader=shader,
            cbv=[self.configBuffer],
            srv=[self.ParticlesBuffer],
            uav=[self.WorkingCellOffsetsBuffer, self.SortedParticleIndicesBuffer]
        )

    def UpdateParticles(self) -> None:
        buffer = self.particles.Get().tobytes()
        self.ParticlesUpload.upload(buffer)
        self.ParticlesUpload.copy_to(self.ParticlesBuffer)

    def Dispatch(self) -> None:
        self.updateTime.dispatch(1, 1, 1)
        self.frame += 1

        # t = time.perf_counter()

        x = int(np.ceil(self.particles.size / THREAD_NUM))

        self.computePositions.dispatch(x, 1, 1)

        if (self.frame % 50) == 0:
            self.clearCellOffsets.dispatch(int(np.ceil(self.gridSize[0] * self.gridSize[1] / THREAD_NUM)), 1, 1)

            self.countParticles.dispatch(x, 1, 1)
            self.prefixSum.dispatch(8, 1, 1)

            self.CellOffsetsBuffer.copy_to(self.WorkingCellOffsetsBuffer)
            self.sortParticleIndices.dispatch(x, 1, 1)

        self.computeForces.dispatch(x, 1, 1)

        # print("Dispatch time:", (time.perf_counter() - t) * 1000, "ms")

    def ReadResults(self):
        self.timeBuffer.copy_to(self.timeReadBack)
        self.time, self.dt = struct.unpack('ff', self.timeReadBack.readback(2 * 4))

        self.ParticlesBuffer.copy_to(self.ParticlesReadBack)
        self.particles.Set(self.particles.FromBuffer(self.ParticlesReadBack.readback(self.particles.size * self.particles.bufferSize)))

    def ReadRenderResults(self):
        self.timeBuffer.copy_to(self.timeReadBack)
        self.time, self.dt = struct.unpack('ff', self.timeReadBack.readback(2 * 4))

        self.PositionBuffer.copy_to(self.PositionReadBack)

        return np.frombuffer(self.PositionReadBack.readback(self.particles.size * 8), dtype=np.float32).reshape((-1,2))