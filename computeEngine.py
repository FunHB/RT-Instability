import struct
import threading
from typing import Tuple
from queue import SimpleQueue, Empty

import numpy as np

from Types import Particles

import compushady
import compushady.formats
from compushady import Buffer, HEAP_READBACK, HEAP_UPLOAD, Compute
from compushady.shaders import hlsl

THREAD_NUM = 256

SHADER_DIRECTORY = 'Shaders/Compute/'

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
    def __init__(self, particles: Particles, dt: float, gravity: Tuple[float, float]):
        self.particles = particles

        maxSigma = np.max(self.particles.particles['sigma'])
        rCut = 2.5 * maxSigma
        self.gravity = np.array(gravity, dtype=np.float32)

        padding = 1.1
        self.gridSize = np.ceil(
            np.array([
                (self.particles.bounds[1] - self.particles.bounds[0]), 
                (self.particles.bounds[3] - self.particles.bounds[2])
            ]) / (rCut * padding)
        ).astype(np.uint32)

        self.numberOfCells = int(self.gridSize[0] * self.gridSize[1])

        self.dt = dt
        self.frame = 0

        self._energyReadbackConfigured = False
        self._energyReadbackEnabled = False
        self._energyStopEvent = threading.Event()
        self._energyStateLock = threading.Lock()
        self._energySamples = SimpleQueue()
        self._energyReadbackThread = None

        self.particleDispatchX = (self.particles.size + THREAD_NUM - 1) // THREAD_NUM
        self.cellDispatchX = (self.numberOfCells + THREAD_NUM - 1) // THREAD_NUM

        if self.cellDispatchX > THREAD_NUM:
            raise ValueError(f'Grid is too large for two-level scan: {self.numberOfCells} cells (max {THREAD_NUM * THREAD_NUM}).')

        self.InitializeConstantBuffers()
        self.InitializeIOBuffers()
        
        self.UpdateParticles()
        
        self.InitializeShader()

        # Auto-enable async sampling when a project-specific EnergyBuffer exists.
        if 'EnergyBuffer' in self.__dict__:
            self.ConfigureAsyncReadback(sourceBuffer=self.TotalEnergyBuffer, sampleSize=4, intervalFrames=120, slots=4, lagSamples=2)
        
    def InitializeConstantBuffers(self):
        # Config
        self.configBuffer = Buffer(Align256(6 * 4))
        self.configUpload = Buffer(self.configBuffer.size, HEAP_UPLOAD)

        self.configUpload.upload(struct.pack('I2II2f', self.particles.size, *self.gridSize, 0, *self.gravity))
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

        self.EnergyBuffer = Buffer(4 * self.particleDispatchX, format=compushady.formats.R32_FLOAT)
        self.TotalEnergyBuffer = Buffer(4, format=compushady.formats.R32_FLOAT)

        self.CellCountsBuffer = Buffer(4 * self.numberOfCells, format=compushady.formats.R32_UINT)
        self.CellOffsetsBuffer = Buffer(int(4 * (self.gridSize[0] * self.gridSize[1] + 1)), format=compushady.formats.R32_UINT)
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
            uav=[self.timeBuffer]
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
            srv=[self.timeBuffer, self.TotalEnergyBuffer],
            uav=[self.ParticlesBuffer]
        )

        with open(COMPUTE_FORCES_SHADER_PATH) as file:
            computeForces = hlsl.compile(file.read())

        self.computeForces = Compute(
            shader=computeForces,
            cbv=[self.configBuffer, self.boundsBuffer],
            srv=[self.CellOffsetsBuffer, self.SortedParticleIndicesBuffer, self.timeBuffer],
            uav=[self.ParticlesBuffer, self.EnergyBuffer]
        )

        with open(SHADER_DIRECTORY + 'ReduceEnergy.hlsl') as file:
            reduceEnergyShader = hlsl.compile(file.read())

        self.energyConfigBuffer = Buffer(Align256(4))
        self.energyConfigUpload = Buffer(self.energyConfigBuffer.size, HEAP_UPLOAD)
        self.energyConfigUpload.upload(struct.pack('I', self.particleDispatchX))
        self.energyConfigUpload.copy_to(self.energyConfigBuffer)
        self.reduceEnergy = Compute(shader=reduceEnergyShader, cbv=[self.energyConfigBuffer], srv=[self.EnergyBuffer], uav=[self.TotalEnergyBuffer])

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
        self.reduceEnergy.dispatch(1, 1, 1)
        self.computePositions.dispatch(self.particleDispatchX, 1, 1)

    # def ReadResults(self):
    #     self.PositionBuffer.copy_to(self.PositionReadBack)
    #     self.particles.SetPositions(np.frombuffer(self.PositionReadBack.readback(self.particles.size * 8), dtype=np.float32).reshape((self.particles.size,2)))

    # def ReadRenderResults(self):
    #     self.PositionBuffer.copy_to(self.PositionReadBack)
    #     self.particles.SetPositions(np.frombuffer(self.PositionReadBack.readback(self.particles.size * 8), dtype=np.float32).reshape((self.particles.size,2)))

    #     return self.particles.GetPositions()

    def ConfigureAsyncReadback(self, sourceBuffer: Buffer, sampleSize: int, intervalFrames: int = 120, slots: int = 4, lagSamples: int = 2):
        if sampleSize <= 0:
            raise ValueError('sampleSize must be greater than 0.')
        if intervalFrames <= 0:
            raise ValueError('intervalFrames must be greater than 0.')
        if slots < 2:
            raise ValueError('slots must be at least 2.')
        if lagSamples < 1:
            raise ValueError('lagSamples must be at least 1.')

        self._energySourceBuffer = sourceBuffer
        self._energySampleSize = sampleSize
        self._energyIntervalFrames = intervalFrames
        self._energyLagSamples = lagSamples
        self._energyReadbackBuffers = [Buffer(sampleSize, HEAP_READBACK) for _ in range(slots)]

        with self._energyStateLock:
            self._energySlotFrames = [-1] * slots
            self._energyWriteSlot = 0

        self._energyReadbackConfigured = True
        self._energyReadbackEnabled = True

        if self._energyReadbackThread is None or not self._energyReadbackThread.is_alive():
            self._energyStopEvent.clear()
            self._energyReadbackThread = threading.Thread(target=self._AsyncReadbackLoop, daemon=True)
            self._energyReadbackThread.start()

    def DisableAsyncReadback(self):
        self._energyReadbackEnabled = False

    def Shutdown(self):
        self._energyStopEvent.set()
        if self._energyReadbackThread is not None and self._energyReadbackThread.is_alive():
            self._energyReadbackThread.join(timeout=1.0)

    def _ScheduleAsyncReadback(self):
        if not self._energyReadbackConfigured:
            return
        if self.frame % self._energyIntervalFrames != 0:
            return

        with self._energyStateLock:
            slot = self._energyWriteSlot

            # Skip this sample when consumer is slower than producer.
            if self._energySlotFrames[slot] != -1:
                return

            self._energySourceBuffer.copy_to(self._energyReadbackBuffers[slot], size=self._energySampleSize)
            self._energySlotFrames[slot] = self.frame
            self._energyWriteSlot = (slot + 1) % len(self._energyReadbackBuffers)

    def _AsyncReadbackLoop(self):
        while not self._energyStopEvent.is_set():
            slot = self._GetReadyReadbackSlot()

            if slot == -1:
                self._energyStopEvent.wait(0.01)
                continue

            with self._energyStateLock:
                frame = self._energySlotFrames[slot]
            payload = self._energyReadbackBuffers[slot].readback(self._energySampleSize)
            with self._energyStateLock:
                self._energySlotFrames[slot] = -1
            self._energySamples.put((frame, payload))

    def _GetReadyReadbackSlot(self) -> int:
        if not self._energyReadbackConfigured:
            return -1

        threshold = self.frame - self._energyIntervalFrames * self._energyLagSamples

        candidate = -1
        oldestFrame = None

        with self._energyStateLock:
            for i, sampleFrame in enumerate(self._energySlotFrames):
                if sampleFrame == -1 or sampleFrame > threshold:
                    continue

                if oldestFrame is None or sampleFrame < oldestFrame:
                    oldestFrame = sampleFrame
                    candidate = i

        return candidate

    def GetAsyncReadbackSamples(self, maxSamples: int = 0):
        samples = []

        while True:
            if maxSamples > 0 and len(samples) >= maxSamples:
                break

            try:
                samples.append(self._energySamples.get_nowait())
            except Empty:
                break

        return samples

    def GetLatestAsyncReadbackFloat(self):
        latest = None
        for sample in self.GetAsyncReadbackSamples():
            latest = sample

        if latest is None:
            return None

        frame, payload = latest
        value = struct.unpack('f', payload[:4])[0]
        return frame, value
