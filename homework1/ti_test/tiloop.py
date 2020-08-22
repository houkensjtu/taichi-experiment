import taichi as ti
import time

ti.init()
nx, ny = 10000,10000
p = ti.field(dtype=ti.f32, shape=(nx,ny))
@ti.kernel
def main():
    for i,j in p:
        p[i,j] = p[i,j] + 0.1

start = time.time()
main()
print(time.time()-start)
