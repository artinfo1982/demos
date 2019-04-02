import ctypes

ll = ctypes.cdll.LoadLibrary
lib = ll('./test.so')

print(lib.add(3, 4))
