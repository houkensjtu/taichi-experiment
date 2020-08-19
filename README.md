# taichi-experiment
My experimental code using Taichi programming language.

### Tricks learned from other Taichi projects

+ Update velocity field using a switch

[source](https://github.com/Wimacs/taichi_code/blob/master/hw1/stable.py)

```Python
class TexPair:
    def __init__(self, cur, nxt):
        self.cur = cur
        self.nxt = nxt

    def swap(self):
        self.cur, self.nxt = self.nxt, self.cur
velocities_pair = TexPair(_velocities, _new_velocities)
```

+ Apply boundary condition using a helper function

[source](https://github.com/hietwll/LBM_Taichi/blob/b2d758a5d14ebcf3a59ad0b11b12bcbfd1c11187/lbm_solver.py#L135)

```Python
# ibc, jbc stand for boundary
# inb, jnb stand for the corresponding internal cell, probably derived from "neighbour"
# dr stands for direction, has 4 values 0-3, refer to top, bottom, left and right
# bc_type is a 4 value list, represent the bc type for top bottom left and right bc; 
# for example [0,0,0,0] where 0 -> Dirichlet ; 1 -> Neumann
 @ti.func
 def apply_bc_core(self, outer, dr, ibc, jbc, inb, jnb):
     if (outer == 1):  # handle outer boundary
         if (self.bc_type[dr] == 0):
             self.vel[ibc, jbc][0] = self.bc_value[dr, 0]
             self.vel[ibc, jbc][1] = self.bc_value[dr, 1]
         elif (self.bc_type[dr] == 1):
             self.vel[ibc, jbc][0] = self.vel[inb, jnb][0]
             self.vel[ibc, jbc][1] = self.vel[inb, jnb][1]
        self.rho[ibc, jbc] = self.rho[inb, jnb]
        for k in ti.static(range(9)):
            self.f_old[ibc,jbc][k] = self.f_eq(ibc,jbc,k) - self.f_eq(inb,jnb,k) + \
                                        self.f_old[inb,jnb][k]
```
