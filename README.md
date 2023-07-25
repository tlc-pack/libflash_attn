The flash attention v2 kernel has been extracted from [the original repo](https://github.com/Dao-AILab/flash-attention) into this repo to make it easier to integrate into a third-party project. In particular, the dependency on libtorch was removed.

As a consquence, dropout is not supported (since the original code uses randomness provided by libtorch). Also, only forward is supported for now.


Build with
```
mkdir build && cd build
cmake ..
make
```
