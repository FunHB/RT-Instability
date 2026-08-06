import struct
from typing import Tuple

import numpy as np
import numba as nb
from numba.extending import overload

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
        self.timeReadBack = Buffer(self.timeBuffer.size, HEAP_READBACK)

        self.timeUpload.upload(struct.pack('ff', self.time, self.dt))
        self.timeUpload.copy_to(self.timeBuffer)

    def InitializeIOBuffers(self):
        self.ParticlesBuffer = Buffer(self.particles.bufferSize * self.particles.size, stride=self.particles.bufferSize)
        self.ParticlesUpload = Buffer(self.ParticlesBuffer.size, HEAP_UPLOAD)
        self.ParticlesReadBack = Buffer(self.ParticlesBuffer.size, HEAP_READBACK)

        self.PositionBuffer = Buffer(8 * self.particles.size, format=compushady.formats.R32G32_FLOAT)
        self.PositionReadBack = Buffer(self.PositionBuffer.size, HEAP_READBACK)

        self.CellOffsetsBuffer = Buffer(4 * (self.gridSize[0] * self.gridSize[1] + 1), format=compushady.formats.R32_UINT)
        self.CellOffsetsUpload = Buffer(self.CellOffsetsBuffer.size, HEAP_UPLOAD)

        self.SortedParticleIndicesBuffer = Buffer(4 * self.particles.size, format=compushady.formats.R32_UINT)
        self.SortedParticleIndicesUpload = Buffer(self.SortedParticleIndicesBuffer.size, HEAP_UPLOAD)
    
    def InitializeShader(self):

        with open(COMPUTE_POSITIONS_SHADER_PATH) as file:
            computePositions = hlsl.compile(file.read())

        self.computePositions = Compute(
            shader=computePositions,
            cbv=[self.configBuffer, self.boundsBuffer],
            uav=[self.ParticlesBuffer, self.PositionBuffer, self.timeBuffer]
        )

        with open(COMPUTE_FORCES_SHADER_PATH) as file:
            computeForces = hlsl.compile(file.read())

        self.computeForces = Compute(
            shader=computeForces,
            cbv=[self.configBuffer, self.boundsBuffer],
            srv=[self.CellOffsetsBuffer, self.SortedParticleIndicesBuffer, self.timeBuffer],
            uav=[self.ParticlesBuffer]
        )

    def UpdateParticles(self) -> None:
        buffer = self.particles.Get().tobytes()
        self.ParticlesUpload.upload(buffer)
        self.ParticlesUpload.copy_to(self.ParticlesBuffer)

    def CalculateCells(self) -> None:
        offsets, sortedIndices, cellIndices = ComputeCellsCountingSort(self.particles.GetPositions(), self.gridSize, self.particles.bounds)

        self.cells = cellIndices

        buffer = offsets.astype(np.uint32).tobytes()
        self.CellOffsetsUpload.upload(buffer)
        self.CellOffsetsUpload.copy_to(self.CellOffsetsBuffer)

        buffer = sortedIndices.astype(np.uint32).tobytes()
        self.SortedParticleIndicesUpload.upload(buffer)
        self.SortedParticleIndicesUpload.copy_to(self.SortedParticleIndicesBuffer)

    def Dispatch(self) -> None:
        self.frame += 1

        # t = time.perf_counter()

        # compute cells and upload cell offsets
        self.CalculateCells()

        x = int(np.ceil(self.particles.size / THREAD_NUM))

        # dispatch compute shader
        self.computeForces.dispatch(x, 1, 1)
        self.computePositions.dispatch(x, 1, 1)

        # read results back to CPU
        self.ReadResults()

        # print("Dispatch time:", (time.perf_counter() - t) * 1000, "ms")

    def ReadResults(self):
        self.PositionBuffer.copy_to(self.PositionReadBack)
        self.particles.SetPositions(np.frombuffer(self.PositionReadBack.readback(self.particles.size * 8), dtype=np.float32).reshape((self.particles.size,2)))

    def ReadRenderResults(self):
        self.timeBuffer.copy_to(self.timeReadBack)
        self.time, self.dt = struct.unpack('ff', self.timeReadBack.readback(2 * 4))

        return self.particles.GetPositions()

@nb.njit(fastmath=True)
def ComputeCellsCountingSort(positions, gridSize, bounds):
    numberOfParticles = positions.shape[0]
    gridWidth = int(gridSize[0])
    gridHeight = int(gridSize[1])
    numberOfCells = gridWidth * gridHeight

    slopeX = gridWidth / (bounds[1] - bounds[0])
    slopeY = gridHeight / (bounds[3] - bounds[2])
    minX = bounds[0]
    minY = bounds[2]
    
    # 1. Compute cell IDs and count particles per cell
    cellIndices = np.zeros(numberOfParticles, dtype=np.int32)
    cellCounts = np.zeros(numberOfCells, dtype=np.int32)
    
    for i in range(numberOfParticles):

        cellX = int((positions[i, 0] - minX) * slopeX)
        cellY = int((positions[i, 1] - minY) * slopeY)
        
        # Clamp to grid bounds (mimicking shader clamp safety)
        if cellX < 0: cellX = 0
        elif cellX >= gridWidth: cellX = gridWidth - 1
        
        if cellY < 0: cellY = 0
        elif cellY >= gridHeight: cellY = gridHeight - 1
            
        cellIndex = cellX + gridWidth * cellY
        cellIndices[i] = cellIndex
        cellCounts[cellIndex] += 1
        
    # 2. Compute offsets via prefix sum
    cellOffsets = np.zeros(numberOfCells + 1, dtype=np.int32)
    currentOffset = 0
    for i in range(numberOfCells):
        cellOffsets[i] = currentOffset
        currentOffset += cellCounts[i]

    cellOffsets[numberOfCells] = numberOfParticles
        
    # 3. Sort indices into their bins
    sortedIndices = np.zeros(numberOfParticles, dtype=np.int32)
    
    # We use a copy of offsets as write-cursors for each bin
    insertOffsets = np.copy(cellOffsets)
    
    for i in range(numberOfParticles):
        cellIndex = cellIndices[i]
        position = insertOffsets[cellIndex]
        sortedIndices[position] = i
        insertOffsets[cellIndex] += 1

    return cellOffsets, sortedIndices, cellIndices