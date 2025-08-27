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
## lychee

```bash
podman build -f Dockerfile -t cg-cuda1281:latest

podman run --rm -it -v ${PWD}:/cg --device=nvidia.com/gpu=all \
   cg-cuda1281:latest

cmake -S /cg -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ARCH_HOPPER90=ON
  -DCMAKE_CXX_FLAGS="-Wno-deprecated-declarations"

cmake --build build --parallel 48

build/main 512 112 164 f32
```
