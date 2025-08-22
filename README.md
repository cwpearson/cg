# CG

```bash

cmake -S . -B build \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ARCH_AMPERE86=ON \
  -DCMAKE_CXX_FLAGS="-Wno-deprecated-declarations"


cmake --build build --parallel 48
```

