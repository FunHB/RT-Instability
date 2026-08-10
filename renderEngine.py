import struct
import time
from typing import Tuple

import compushady.formats
import glfw
from compushady import Buffer, Compute, Texture2D, HEAP_UPLOAD
from compushady.shaders import hlsl
from compushady.utils import create_swapchain_from_glfw

from Types import Particles
from computeEngine import ComputeEngine

SHADER_DIRECTORY = 'Shaders/'

CLEAR_RENDER_TARGET_SHADER_PATH = SHADER_DIRECTORY + 'ClearRenderTarget.hlsl'
DRAW_PARTICLES_SHADER_PATH = SHADER_DIRECTORY + 'DrawParticles.hlsl'


def Align256(size):
    return (size + 255) & ~255


class RenderEngine:
    def __init__(self, particles: Particles, bounds: Tuple[float, float, float, float], computeEngine: ComputeEngine, print: bool = False):
        self.computeEngine = computeEngine
        self.particles = particles
        self.bounds = bounds
        self.print = print

        self.fps = 60
        self.lastRender = time.perf_counter()

        if not glfw.init():
            raise RuntimeError('Failed to initialize GLFW.')

        glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
        self.window = glfw.create_window(1200, 900, 'Simulation', None, None)

        if not self.window:
            glfw.terminate()
            raise RuntimeError('Failed to create GLFW window.')

        self.swapchain = create_swapchain_from_glfw(self.window, format=compushady.formats.B8G8R8A8_UNORM, num_buffers=3)
        self.renderTarget = Texture2D(self.swapchain.width, self.swapchain.height, compushady.formats.B8G8R8A8_UNORM)

        self.InitializeShader()

    def InitializeShader(self):
        self.clearConfigBuffer = Buffer(Align256(3 * 4))
        self.clearConfigUpload = Buffer(self.clearConfigBuffer.size, HEAP_UPLOAD)

        self.renderConfigBuffer = Buffer(Align256((4 + 4) * 4))
        self.renderConfigUpload = Buffer(self.renderConfigBuffer.size, HEAP_UPLOAD)

        clearColor = 0xFF000000
        self.clearConfigUpload.upload(struct.pack('3I', self.swapchain.width, self.swapchain.height, clearColor))
        self.clearConfigUpload.copy_to(self.clearConfigBuffer)

        pointRadiusPx = 2
        self.renderConfigUpload.upload(
            struct.pack('4I4f', self.particles.size, self.swapchain.width, self.swapchain.height, pointRadiusPx, *self.bounds)
        )
        self.renderConfigUpload.copy_to(self.renderConfigBuffer)

        with open(CLEAR_RENDER_TARGET_SHADER_PATH) as file:
            clearShader = hlsl.compile(file.read())

        self.clearRenderTarget = Compute(
            shader=clearShader,
            cbv=[self.clearConfigBuffer],
            uav=[self.renderTarget]
        )

        with open(DRAW_PARTICLES_SHADER_PATH) as file:
            drawParticlesShader = hlsl.compile(file.read())

        self.drawParticles = Compute(
            shader=drawParticlesShader,
            cbv=[self.renderConfigBuffer],
            srv=[self.computeEngine.PositionBuffer, self.computeEngine.ParticlesBuffer],
            uav=[self.renderTarget]
        )

        self.clearDispatchX = (self.swapchain.width + 15) // 16
        self.clearDispatchY = (self.swapchain.height + 15) // 16
        self.drawDispatchX = (self.particles.size + 255) // 256

    def Update(self):
        self.clearRenderTarget.dispatch(self.clearDispatchX, self.clearDispatchY, 1)
        self.drawParticles.dispatch(self.drawDispatchX, 1, 1)
        self.swapchain.present(self.renderTarget)

    def RenderLoop(self):
        try:
            while not glfw.window_should_close(self.window):
                now = time.perf_counter()
                if now - self.lastRender >= 1 / self.fps:
                    self.lastRender = now
                    self.Update()

                glfw.poll_events()
        finally:
            glfw.destroy_window(self.window)
            glfw.terminate()