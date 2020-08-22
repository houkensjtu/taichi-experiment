import numpy as np
import time

nx, ny = 10000,10000
pnp = np.zeros((nx,ny))
def main():
#    for i in range(nx):
#        for j in range(ny):
#            pnp[i,j] = pnp[i,j] + 0.1
    global pnp
    pnp += 0.1
start = time.time()
main()
print(time.time()-start)
