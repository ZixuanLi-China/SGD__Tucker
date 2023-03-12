CXX=g++

LIB_FLAGS = -larmadillo -llapack -lblas -DARMA_DONT_USE_WRAPPER

OPT = -O2 -mcmodel=medium  -fopenmp -w -std=c++11

CXXFLAGS = $(DEBUG) $(FINAL) $(OPT) $(EXTRA_OPT)

all: SGD__Tucker demo


SGD__Tucker: SGD__Tucker.cpp 
	$(CXX) $(CXXFLAGS)  -o $@  $< $(LIB_FLAGS)

demo: SGD__Tucker.cpp
	g++ -std=c++11 -o SGD__Tucker SGD__Tucker.cpp -O2 -fopenmp -w -mcmodel=medium -larmadillo -llapack -lblas -DARMA_DONT_USE_WRAPPER
	./SGD__Tucker ./Data/movielens_tensor.train ./Data/movielens_tensor.test 4 3 4 4 4
	
.PHONY: clean

clean:
	rm -f SGD__Tucker

