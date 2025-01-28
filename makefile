#use: make clean && make -j CHUNK_QUBIT={C}
CC = /usr/local/cuda/bin/nvcc
INCLUDE = include/
OBJS = $(addprefix obj/, graph.o state.o rzz.o rx.o swap.o)
CXXFLAGS = -lnccl -DCHUNK_QUBIT=$(CHUNK_QUBIT) -arch=sm_80 -O3 -maxrregcount=64
vpath %.cu src/

.PHONY: clean

all: weighted

# unweighted: obj/Unweighted.o $(OBJS)
# 	$(CC) -o $@ $^

weighted: obj/Weighted.o $(OBJS)
	$(CC) $(CXXFLAGS) -o $@ $^

obj/%.o: %.cu
	@mkdir -p obj
	$(CC) $(CXXFLAGS) -dc --expt-relaxed-constexpr -I $(INCLUDE) -o $@ $<

clean:
	-rm obj/*.o
	-rm weighted
	# -rm unweighted
