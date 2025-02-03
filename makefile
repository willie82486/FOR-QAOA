NVCXX = nvcc
MPICXX = mpicxx

INCLUDE = include/

SRCS = src/graph.cpp src/state.cpp src/Weighted.cpp
GPU_SRCS = src/rx.cu src/rzz.cu src/swap.cu
OBJS = $(patsubst src/%.cpp, obj/%.o, $(SRCS))
GPU_OBJS = $(patsubst src/%.cu, obj/%.o, $(GPU_SRCS))
D_LINK = obj/dlink.o

CXXFLAGS = -DCHUNK_QUBIT=$(CHUNK_QUBIT) -DUSE_MPI -O3
NVCXXFLAGS = -arch=sm_90 -maxrregcount=64 -Xcompiler -fopenmp
LDFLAGS = -lnccl -lcudart -lmpi
CXXFLAGS += -I$(MPI_HOME)include
LDFLAGS += -L$(MPI_HOME)lib -L$(MPI_HOME)lib64

.PHONY: clean all

all: weighted

weighted: $(OBJS) $(GPU_OBJS) $(D_LINK)
	$(MPICXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Device linking step
$(D_LINK): $(GPU_OBJS)
	@mkdir -p obj
	$(NVCXX) $(CXXFLAGS) $(NVCXXFLAGS) -dlink $^ -o $@

obj/%.o: src/%.cu
	@mkdir -p obj
	$(NVCXX) $(CXXFLAGS) $(NVCXXFLAGS) -dc --expt-relaxed-constexpr -I $(INCLUDE) -o $@ $<

obj/%.o: src/%.cpp
	@mkdir -p obj
	$(MPICXX) $(CXXFLAGS) -c -I $(INCLUDE) -o $@ $<

clean:
	-rm -f obj/*.o
	-rm -f weighted
