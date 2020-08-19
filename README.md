# taichi-experiment
My experimental code using Taichi programming language.

### Tricks learned from other Taichi projects

+ Update velocity field using a switch

src:https://github.com/Wimacs/taichi_code/blob/master/hw1/stable.py

```Python
class TexPair:
    def __init__(self, cur, nxt):
        self.cur = cur
        self.nxt = nxt

    def swap(self):
        self.cur, self.nxt = self.nxt, self.cur
velocities_pair = TexPair(_velocities, _new_velocities)
```
