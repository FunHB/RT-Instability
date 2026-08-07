import threading

import numpy as np

class SharedBuffer:
    def __init__(self, dtype: np.dtype):
        self.dtype = dtype
        self.lock = threading.Lock()
        
    def AllocateMemory(self, shape: tuple):
        self.shape = shape

        self.memory = np.zeros((2,) + shape, dtype=self.dtype)
        self.writeIndex = 0
        self.readIndex = 1
    
    def GetBackBuffer(self):
        with self.lock:
            return self.memory[self.writeIndex]
        
    def GetFrontBuffer(self):
        with self.lock:
            return self.memory[self.readIndex]

    def WriteBackAndSwap(self, data, field=None):
        with self.lock:
            if field is None:
                self.memory[self.writeIndex][...] = data
            else:
                self.memory[self.writeIndex][field] = data

            self.writeIndex, self.readIndex = self.readIndex, self.writeIndex

    def SwapBuffers(self):
        with self.lock:
            self.writeIndex, self.readIndex = self.readIndex, self.writeIndex

    def Resize(self, newShape):
        with self.lock:
            if hasattr(self, "memory") and self.memory.shape[1:] == newShape:
                return

            self.AllocateMemory(newShape)