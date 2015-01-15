#################
#
# Makefile for "Kernels for computation and communication overlap"
#
# Adjust to following for your local setup:

# set path to cuda runtime library for the Fortran linker
LIB_PATH = -L/cm/shared/apps/cuda55/toolkit/current/lib64/

# select the right compute capability for your GPU
#NVCCFLAGS = -gencode arch=compute_35,code=sm_35 -Xptxas=-v
#NVCCFLAGS = -gencode arch=compute_30,code=sm_30 -Xptxas=-v
NVCCFLAGS = -gencode arch=compute_20,code=sm_20 -Xptxas=-v

#################

# Compilers
CC = g++
NVCC = nvcc -Xcompiler=-Wall
#FC = ifort $(IFLAGS)
FC = gfortran $(FFLAGS)

OPT = -O3
#OPT = -g

CFLAGS = $(OPT)

#gfortran options
FFLAGS = -fdefault-double-8 -fdefault-real-8 -Wall -Wtabs $(OPT) -fimplicit-none -ffree-line-length-none

#ifort options
IFLAGS = $(OPT) -fpconstant -r8

.PHONY: clean

all: matmul conv benchmark-pci state buoydiff spmv 

clean:
	rm -f *.o *.mod matmul conv benchmark-pci state buoydiff spmv 

benchmark-pci: benchmark-pci.cu timer.o 
	$(NVCC) $(OPT) $(NVCCFLAGS) -c benchmark-pci.cu -o benchmark-pci.o
	$(NVCC) $(OPT) $(NVCCFLAGS) benchmark-pci.o timer.o -o benchmark-pci

conv: conv.cu timer.o 
	$(NVCC) $(OPT) $(NVCCFLAGS) -c conv.cu -o conv.o
	$(NVCC) $(OPT) $(NVCCFLAGS) conv.o timer.o -o conv

matmul: matmul.cu timer.o 
	$(NVCC) $(OPT) $(NVCCFLAGS) -c matmul.cu -o matmul.o
	$(NVCC) $(OPT) $(NVCCFLAGS) matmul.o timer.o -o matmul

spmv: spmv.cu timer.o 
	$(NVCC) $(OPT) $(NVCCFLAGS) -c spmv.cu -o spmv.o
	$(NVCC) $(OPT) $(NVCCFLAGS) timer.o spmv.o -o spmv

buoydiff: buoydiff.F90 timer.o buoydiff.cu domain.h state_mod.F90
	$(NVCC) $(OPT) $(NVCCFLAGS) -c buoydiff.cu -o buoydiff_c.o
	$(FC) -c state_mod.F90 -o state_mod.o
	$(FC) -c buoydiff.F90 -o buoydiff.o
	$(FC) $(LIB_PATH) -lstdc++ -lcudart buoydiff.o buoydiff_c.o state_mod.o timer.o -o buoydiff

state: state.F90 state.cu timer.o domain.h state_mod.F90 state_init.cu
	$(NVCC) -O3 $(NVCCFLAGS) -c state.cu -o state_c.o
	$(FC) -c state_mod.F90 -o state_mod.o
	$(FC) -c state.F90 -o state.o
	$(FC) $(LIB_PATH) -lstdc++ -lcudart state.o state_c.o state_mod.o timer.o -o state

timer.o: timer.cc timer.h
	$(CC) $(CFLAGS) -c timer.cc -o timer.o

