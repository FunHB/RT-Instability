import struct
import time
from typing import Tuple

import numpy as np

import compushady.formats
import glfw
from compushady import Buffer, Compute, Texture2D, HEAP_UPLOAD
from compushady.shaders import hlsl
from compushady.utils import create_swapchain_from_glfw

from Types import Particles
from computeEngine import ComputeEngine, Align256

SHADER_DIRECTORY = 'Shaders/Render/'

CLEAR_RENDER_TARGET_SHADER_PATH = SHADER_DIRECTORY + 'ClearRenderTarget.hlsl'
DRAW_PARTICLES_SHADER_PATH = SHADER_DIRECTORY + 'DrawParticles.hlsl'

class RenderEngine:
    def __init__(self, particles: Particles, bounds: Tuple[float, float, float, float], computeEngine: ComputeEngine, print: bool = False):
        self.computeEngine = computeEngine
        self.particles = particles
        self.bounds = bounds
        self.print = print

        self.particleSize = 3   # px
        self.colors = np.array([0xFF000000, 0xFF7F7F7F, 0xFFFFFFFF], dtype=np.uint32)

        boundsWidth = (self.bounds[1] - self.bounds[0])
        boundsHeight = (self.bounds[3] - self.bounds[2])

        baseHeight = 900
        baseWidth = 1600

        self.aspectRatio = abs(boundsWidth / boundsHeight)
        height = baseHeight if boundsHeight >= boundsWidth else int(baseWidth // self.aspectRatio)
        width = baseWidth if boundsWidth > boundsHeight else int(baseHeight * self.aspectRatio)

        self.fps = 30
        self.lastRender = time.perf_counter()

        if not glfw.init():
            raise RuntimeError('Failed to initialize GLFW.')

        glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
        self.window = glfw.create_window(width, height, 'Simulation', None, None)

        if not self.window:
            glfw.terminate()
            raise RuntimeError('Failed to create GLFW window.')

        self.width, self.height = glfw.get_framebuffer_size(self.window)

        self.swapchain = create_swapchain_from_glfw(self.window, format=compushady.formats.B8G8R8A8_UNORM)
        self.renderTarget = Texture2D(self.width, self.height, compushady.formats.B8G8R8A8_UNORM)

        self.InitializeBuffers()
        self.InitializeShader()

    def InitializeBuffers(self):
        self.clearConfigBuffer = Buffer(Align256(3 * 4))
        self.clearConfigUpload = Buffer(self.clearConfigBuffer.size, HEAP_UPLOAD)

        self.clearConfigUpload.upload(struct.pack('3I', self.width, self.height, self.colors[0]))
        self.clearConfigUpload.copy_to(self.clearConfigBuffer)

        self.renderConfigBuffer = Buffer(Align256((4 + 4) * 4))
        self.renderConfigUpload = Buffer(self.renderConfigBuffer.size, HEAP_UPLOAD)

        self.renderConfigUpload.upload(struct.pack('4I4f', self.particles.size, self.width, self.height, self.particleSize, *self.bounds))
        self.renderConfigUpload.copy_to(self.renderConfigBuffer)

        self.colorsBuffer = Buffer(4 * self.colors.size, format=compushady.formats.B8G8R8A8_UNORM)
        self.colorsUpload = Buffer(self.colorsBuffer.size, HEAP_UPLOAD)

        self.colorsUpload.upload(self.colors.tobytes())
        self.colorsUpload.copy_to(self.colorsBuffer)

    def InitializeShader(self):

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
            srv=[self.computeEngine.ParticlesBuffer, self.colorsBuffer],
            uav=[self.renderTarget]
        )

        self.clearDispatchX = (self.width + 15) // 16
        self.clearDispatchY = (self.height + 15) // 16
        self.drawDispatchX = (self.particles.size + 255) // 256

    def Update(self):
        self.clearRenderTarget.dispatch(self.clearDispatchX, self.clearDispatchY, 1)
        self.drawParticles.dispatch(self.drawDispatchX, 1, 1)
        self.swapchain.present(self.renderTarget)

        glfw.set_window_title(self.window, f"Simulation | {self.computeEngine.frame * self.computeEngine.dt:.2f} s")

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