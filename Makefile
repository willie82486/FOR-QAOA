CXX = g++ -std=c++17
MPICXX = mpicxx -std=c++17
CFLAGS = -Wall -g -O3 -D_GNU_SOURCE -fopenmp
MPICFLAGS = -Wall -g -O3 -D_GNU_SOURCE -fopenmp -DUSE_MPI
SRCS = $(wildcard *.cpp)

ifdef USE_AVX
CFLAGS += -mavx512f -march=native -mtune=native -I/opt/AMD/aocl/aocl-linux-gcc-4.2.0/gcc/include/ -L/opt/AMD/aocl/aocl-linux-gcc-4.2.0/gcc/lib/ -lalmfast -lalm -lm -laoclutils -lstdc++ -DUSE_AVX
MPICFLAGS += -I/opt/AMD/aocl/aocl-linux-gcc-4.2.0/gcc/include -L/opt/AMD/aocl/aocl-linux-gcc-4.2.0/gcc/lib -mavx512f -mtune=native -march=native -lalm -lm -laoclutils -DUSE_AVX
SRCS += $(wildcard ./avx/*.cpp)
endif 

OBJS = $(SRCS:.cpp=.o)

TARGET = qaoa

.PHONY: all clean run run_single run_rdma

all: $(TARGET)

$(TARGET): $(OBJS)
ifeq ($(USE_MPI), 1)
	$(MPICXX) $^ $(MPICFLAGS) -o $@
else 
	$(CXX) $^ $(CFLAGS) -o $@
endif

%.o: %.cpp
ifeq ($(USE_MPI), 1)
	$(MPICXX) $(MPICFLAGS) -c $< -o $@
else
	$(CXX) $(CFLAGS) -c $< -o $@
endif 


clean:
	rm -rf *.o avx/*.o qaoa
