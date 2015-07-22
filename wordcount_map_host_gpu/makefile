CC=g++
NVCC=nvcc
SRC_DIR =./src
LIB = ./lib
OBJ_DIR = ./objs

SRCS = $(shell find $(SRC_DIR) -name *.cu)
OBJS = $(patsubst $(SRC_DIR)%.cu, $(OBJ_DIR)%.o, $(SRCS) )

MR4C_HOME ?= /usr/local/mr4c
MR4CINC = $(MR4C_HOME)/native/include
MR4CLIB = $(MR4C_HOME)/native/lib

CUDALINK = /usr/local/cuda/include
CUDALIB = /usr/local/cuda/lib64

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

compile: ./src/changeimage.cu
	$(NVCC) -c -I$(MR4CINC) -I$(CUDALINK) -o ./objs/changeimage.o -Xcompiler -fPIC --compiler-options -Wall -std=c++11 ./src/changeimage.cu

lib: ./objs/changeimage.o
	$(NVCC) -L$(MR4CLIB) -L$(CUDALIB) -lmr4c -lcuda -Xcompiler -rdynamic -shared -Xcompiler -fPIC -std=c++11 -o ./lib/libChangeImage.so ./objs/changeimage.o

.PHONY: clean, lib, init, all
