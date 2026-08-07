import struct
from typing import Tuple

import numpy as np

from Types import Particles

import compushady
import compushady.formats
from compushady import Buffer, HEAP_READBACK, HEAP_UPLOAD, Compute
from compushady.shaders import hlsl

THREAD_NUM = 256

SHADER_DIRECTORY = 'Shaders/'

COMPUTE_FORCES_SHADER_PATH = SHADER_DIRECTORY + 'ComputeForces.hlsl'
COMPUTE_POSITIONS_SHADER_PATH = SHADER_DIRECTORY + 'ComputePositions.hlsl'
COUNT_PARTICLES_SHADER_PATH = SHADER_DIRECTORY + 'CountParticlesPerCell.hlsl'
PREFIX_SUM_BLOCKS_SHADER_PATH = SHADER_DIRECTORY + 'PrefixSum.hlsl'
PREFIX_SUM_BLOCK_SUMS_SHADER_PATH = SHADER_DIRECTORY + 'PrefixSumBlockSums.hlsl'
PREFIX_SUM_FINALIZE_SHADER_PATH = SHADER_DIRECTORY + 'PrefixSumFinalize.hlsl'
SORT_PARTICLE_INDICES_SHADER_PATH = SHADER_DIRECTORY + 'SortParticleIndices.hlsl'
CLEAR_SHADER_PATH = SHADER_DIRECTORY + 'ClearShader.hlsl'
PRESHADER_PATH = SHADER_DIRECTORY + 'PreShader.hlsl'

def Align256(size):
    return (size + 255) & ~255

class ComputeEngine:
    def __init__(self, particles: Particles, dt: float, epsilon: float, g: Tuple[float, float]):
        self.particles = particles

        self.rCut = 2.5 * self.particles.sigma
        self.epsilon = epsilon
        self.g = np.array(g, dtype=np.float32)

        padding = 1.1
        self.gridSize = np.ceil(
            np.array([
                (self.particles.bounds[1] - self.particles.bounds[0]), 
                (self.particles.bounds[3] - self.particles.bounds[2])
            ]) / (self.rCut * padding)
        ).astype(np.uint32)
        self.numberOfCells = int(self.gridSize[0] * self.gridSize[1])

        self.dt = dt
        self.frame = 0

        self.particleDispatchX = (self.particles.size + THREAD_NUM - 1) // THREAD_NUM
        self.cellDispatchX = (self.numberOfCells + THREAD_NUM - 1) // THREAD_NUM

        if self.cellDispatchX > THREAD_NUM:
            raise ValueError(f'Grid is too large for two-level scan: {self.numberOfCells} cells (max {THREAD_NUM * THREAD_NUM}).')

        self.InitializeConstantBuffers()
        self.InitializeIOBuffers()
        
        self.UpdateParticles()
        
        self.InitializeShader()
        
    def InitializeConstantBuffers(self):
        # Config
        self.configBuffer = Buffer(Align256(7 * 4))
        self.configUpload = Buffer(self.configBuffer.size, HEAP_UPLOAD)

        self.configUpload.upload(struct.pack('Ifff2I2f', self.particles.size, self.particles.sigma, self.epsilon, self.rCut, *self.gridSize, *self.g))
        self.configUpload.copy_to(self.configBuffer)

        # Bounds
        self.boundsBuffer = Buffer(Align256(4 * 4))
        self.boundsUpload = Buffer(self.boundsBuffer.size, HEAP_UPLOAD)

        self.boundsUpload.upload(struct.pack('4f', *self.particles.bounds))
        self.boundsUpload.copy_to(self.boundsBuffer)

        # Time
        self.timeBuffer = Buffer(2 * 4, stride=4)
        self.timeUpload = Buffer(self.timeBuffer.size, HEAP_UPLOAD)

        self.timeUpload.upload(struct.pack('If', self.frame, self.dt))
        self.timeUpload.copy_to(self.timeBuffer)

        # Clear shader config
        self.clearConfigBuffer = Buffer(Align256(4))
        self.clearConfigUpload = Buffer(self.clearConfigBuffer.size, HEAP_UPLOAD)

        self.clearConfigUpload.upload(struct.pack('I', self.numberOfCells))
        self.clearConfigUpload.copy_to(self.clearConfigBuffer)

    def InitializeIOBuffers(self):
        self.ParticlesBuffer = Buffer(self.particles.bufferSize * self.particles.size, stride=self.particles.bufferSize)
        self.ParticlesUpload = Buffer(self.ParticlesBuffer.size, HEAP_UPLOAD)

        self.PositionBuffer = Buffer(8 * self.particles.size, format=compushady.formats.R32G32_FLOAT)
        self.PositionReadBack = Buffer(self.PositionBuffer.size, HEAP_READBACK)

        self.EnergyBuffer = Buffer(4, format=compushady.formats.R32_FLOAT)

        self.CellCountsBuffer = Buffer(4 * self.numberOfCells, format=compushady.formats.R32_UINT)
        self.CellOffsetsBuffer = Buffer(4 * (self.gridSize[0] * self.gridSize[1] + 1), format=compushady.formats.R32_UINT)
        self.CellWriteHeadsBuffer = Buffer(4 * self.numberOfCells, format=compushady.formats.R32_UINT)
        self.BlockSumsBuffer = Buffer(4 * self.cellDispatchX, format=compushady.formats.R32_UINT)
        self.BlockOffsetsBuffer = Buffer(4 * self.cellDispatchX, format=compushady.formats.R32_UINT)
        self.ParticleCellIndicesBuffer = Buffer(4 * self.particles.size, format=compushady.formats.R32_UINT)

        self.SortedParticleIndicesBuffer = Buffer(4 * self.particles.size, format=compushady.formats.R32_UINT)
    
    def InitializeShader(self):

        with open(PRESHADER_PATH) as file:
            preShader = hlsl.compile(file.read())
        
        self.preShader = Compute(
            shader=preShader,
            uav=[self.timeBuffer, self.EnergyBuffer]
        )

        with open(CLEAR_SHADER_PATH) as file:
            clearShader = hlsl.compile(file.read())

        self.clearCellCounts = Compute(
            shader=clearShader,
            cbv=[self.clearConfigBuffer],
            uav=[self.CellCountsBuffer]
        )

        with open(COUNT_PARTICLES_SHADER_PATH) as file:
            countParticlesPerCell = hlsl.compile(file.read())

        self.countParticlesPerCell = Compute(
            shader=countParticlesPerCell,
            cbv=[self.configBuffer, self.boundsBuffer],
            srv=[self.ParticlesBuffer],
            uav=[self.CellCountsBuffer, self.ParticleCellIndicesBuffer]
        )

        with open(PREFIX_SUM_BLOCKS_SHADER_PATH) as file:
            prefixSumBlocks = hlsl.compile(file.read())

        self.prefixSumBlocks = Compute(
            shader=prefixSumBlocks,
            cbv=[self.configBuffer],
            uav=[self.CellCountsBuffer, self.CellOffsetsBuffer, self.BlockSumsBuffer]
        )

        with open(PREFIX_SUM_BLOCK_SUMS_SHADER_PATH) as file:
            prefixSumBlockSums = hlsl.compile(file.read())

        self.prefixSumBlockSums = Compute(
            shader=prefixSumBlockSums,
            cbv=[self.configBuffer],
            uav=[self.BlockSumsBuffer, self.BlockOffsetsBuffer]
        )

        with open(PREFIX_SUM_FINALIZE_SHADER_PATH) as file:
            prefixSumFinalize = hlsl.compile(file.read())

        self.prefixSumFinalize = Compute(
            shader=prefixSumFinalize,
            cbv=[self.configBuffer],
            srv=[self.BlockOffsetsBuffer],
            uav=[self.CellOffsetsBuffer, self.CellWriteHeadsBuffer]
        )

        with open(SORT_PARTICLE_INDICES_SHADER_PATH) as file:
            sortParticleIndices = hlsl.compile(file.read())

        self.sortParticleIndices = Compute(
            shader=sortParticleIndices,
            cbv=[self.configBuffer],
            srv=[self.ParticleCellIndicesBuffer],
            uav=[self.CellWriteHeadsBuffer, self.SortedParticleIndicesBuffer]
        )

        with open(COMPUTE_POSITIONS_SHADER_PATH) as file:
            computePositions = hlsl.compile(file.read())

        self.computePositions = Compute(
            shader=computePositions,
            cbv=[self.configBuffer, self.boundsBuffer],
            srv=[self.timeBuffer, self.EnergyBuffer],
            uav=[self.ParticlesBuffer, self.PositionBuffer]
        )

        with open(COMPUTE_FORCES_SHADER_PATH) as file:
            computeForces = hlsl.compile(file.read())

        self.computeForces = Compute(
            shader=computeForces,
            cbv=[self.configBuffer, self.boundsBuffer],
            srv=[self.CellOffsetsBuffer, self.SortedParticleIndicesBuffer, self.timeBuffer],
            uav=[self.ParticlesBuffer, self.EnergyBuffer]
        )

    def UpdateParticles(self) -> None:
        buffer = self.particles.Get().tobytes()
        self.ParticlesUpload.upload(buffer)
        self.ParticlesUpload.copy_to(self.ParticlesBuffer)

    def CalculateCells(self) -> None:
        self.clearCellCounts.dispatch(self.cellDispatchX, 1, 1)
        self.countParticlesPerCell.dispatch(self.particleDispatchX, 1, 1)
        self.prefixSumBlocks.dispatch(self.cellDispatchX, 1, 1)
        self.prefixSumBlockSums.dispatch(1, 1, 1)
        self.prefixSumFinalize.dispatch(self.cellDispatchX, 1, 1)
        self.sortParticleIndices.dispatch(self.particleDispatchX, 1, 1)

    def Dispatch(self) -> None:
        self.frame += 1

        self.preShader.dispatch(1, 1, 1)

        # compute cells and upload cell offsets
        self.CalculateCells()

        # dispatch compute shader
        self.computeForces.dispatch(self.particleDispatchX, 1, 1)
        self.computePositions.dispatch(self.particleDispatchX, 1, 1)

    def ReadRenderResults(self):
        self.PositionBuffer.copy_to(self.PositionReadBack)
        self.particles.SetPositions(np.frombuffer(self.PositionReadBack.readback(self.particles.size * 8), dtype=np.float32).reshape((self.particles.size,2)))

        return self.particles.GetPositions()
