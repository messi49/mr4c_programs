CC=g++
NVCC=nvcc
SRC_DIR =./src
LIB=./lib
OBJ_DIR=./objs


MR4C_HOME ?= /usr/local/mr4c
MR4CINC = $(MR4C_HOME)/native/include
MR4CGEOINC = $(MR4C_HOME)/geospatial/include
MR4CLIB = $(MR4C_HOME)/native/lib

GDALINC = /usr/include/gdal
GDALLIB = /usr/local/lib

CUDALINK = /usr/local/cuda/include
CUDALIB = /usr/local/cuda/lib64

#default target
all: clean init compile libMap libReduce

clean:
	rm -rf $(LIB)
	rm -rf $(OBJ_DIR)

init:
	mkdir -p $(LIB)
	mkdir -p $(OBJ_DIR)

compile: $(SRC_DIR)/map.cu $(SRC_DIR)/reduce.cu
	$(NVCC) -c -I$(MR4CINC)  -I$(MR4CGEOINC) -I$(GDALINC) -I$(CUDALINK) -I$(SRC_DIR) -Xcompiler -fPIC --compiler-options -Wall -std=c++11 -o $(OBJ_DIR)/map.o $(SRC_DIR)/map.cu
	$(NVCC) -c -I$(MR4CINC) -I$(MR4CGEOINC) -I$(CUDALINK) -I$(SRC_DIR) -Xcompiler -fPIC --compiler-options -Wall -std=c++11 -o $(OBJ_DIR)/reduce.o $(SRC_DIR)/reduce.cu

libMap: $(OBJ_DIR)/map.o
	$(NVCC) -L$(MR4CLIB) -L$(MR4CGEOLIB) -I$(GDALLIB) -L$(CUDALIB) -Xcompiler -rdynamic -shared -Xcompiler -fPIC $(OBJ_DIR)/map.o -o $(LIB)/libMap.so -lmr4c -lmr4cgeo -lcuda

libReduce: $(OBJ_DIR)/reduce.o
	$(NVCC) -L$(MR4CLIB) -L$(CUDALIB) -Xcompiler -rdynamic -shared -Xcompiler -fPIC -o $(LIB)/libReduce.so $(OBJ_DIR)/reduce.o -lmr4c -lcuda

.PHONY: clean, libMap, libReduce, init, all
