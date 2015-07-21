CC = g++
NVCC = nvcc
SRC_DIR =./src
LIB = ./lib
OBJ_DIR = ./objs

SRCS = $(shell find $(SRC_DIR) -name *.cu)
OBJS = $(patsubst $(SRC_DIR)%.cu, $(OBJ_DIR)%.o, $(SRCS) )

MR4C_HOME ?= /usr/local/mr4c
MR4CINC = $(MR4C_HOME)/native/include
MR4CLIB = $(MR4C_HOME)/native/lib

GDALINC = /usr/local/include
GDALLIB = /usr/local/lib

CUDALINK = /usr/local/cuda/include

#default target
all: clean init compile lib

clean:
	rm -rf $(OBJ_DIR)/*
	rm -rf $(LIB)/*
	rm -rf $(OBJ_DIR)
	rm -rf $(LIB)

init:
	mkdir -p $(OBJ_DIR)
	mkdir -p $(LIB)

compile: ./src/imageinfo.cu
	$(NVCC) -c -I$(MR4CINC) -I$(GDALINC) -I$(CUDALINC) -o ./objs/imageinfo.o -Xcompiler -fPIC --compiler-options -Wall -std=c++11 ./src/imageinfo.cu
	mv imageinfo.o ./objs/.

lib: ./objs/imageinfo.o
	$(NVCC) -L$(MR4CLIB) -lmr4c -L$(GDALLIB) -L$(CUDALINK) --compiler-options -lgdal -lcuda -Xcompiler -rdynamic -shared -Xcompiler -fPIC -std=c++11 -o ./lib/libImageInfo.so ./objs/imageinfo.o

.PHONY: clean, lib, init, all
