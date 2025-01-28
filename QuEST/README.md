# Compiling a Program in QuEST: A Step-by-Step Guide

1. Clone QuEST Repository:

   ```bash
   git clone --depth 1 https://github.com/QuEST-Kit/QuEST
   cd QuEST
   ```

2. Create Build Folder:

   ```bash
   mkdir build
   cd build
   ```

3. Generate Makefile using CMake:
   ```bash
   # Set the following options:
   # - DUSER_SOURCE: the path to the program you want to compile
   # - DGPUACCELERATED: 1 to use GPU, 0 for CPU
   # - DGPU_COMPUTE_CAPABILITY: specify GPU's compute capability (e.g., 75 for 2080Ti, 89 for 4090)
   cmake .. -DUSER_SOURCE=../../QAOA_QuEST.c -DGPUACCELERATED=1 -DGPU_COMPUTE_CAPABILITY=75
   ```
4. Compile the Program:

   ```bash
   make
   ```

5. Execute the Program:

   ```bash
   # The default name of the compiled binary is 'demo'
   ./demo
   ```
