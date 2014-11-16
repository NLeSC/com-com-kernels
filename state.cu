/*
 * Copyright 2014 Netherlands eScience Center
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


/**
 * This program benchmarks four different implementations for
 * overlapping CPU-GPU communication and GPU computation of the
 * MWJF state kernel.
 *
 * @author Ben van Werkhoven <b.vanwerkhoven@esciencecenter.nl>
 * 
 */


#include <stdio.h>
#include <math.h>
#include <unistd.h>

#include "domain.h"

//CUDA thread block size
#define BLOCK_X 32
#define BLOCK_Y 8

//CUDA nStreams
#define NSTREAMS 42
cudaStream_t stream[NSTREAMS];

int nStreams = 42;

#define ITERATIONS 5

//TMIN TMAX SMIN SMAX
#define TMIN -2.0
#define TMAX 999.0
#define SMIN 0.0
#define SMAX 0.999


#define CUDA_CHECK_ERROR(errorMessage) do {                                 \
    cudaError_t err = cudaGetLastError();                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
    err = cudaThreadSynchronize();                                           \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
        exit(EXIT_FAILURE);                                                  \
    } } while (0)


extern "C" {

//entry point for Fortran main
  void state_mwjf (double *SALTK, double *TEMPK, double *DRHODT, double *DRHODS, double *RHOFULL, double *pressz);


//Four functions that match the four implementation strategies
//discussed in the paper
  void state_mwjf_implicit (double *SALTK, double *TEMPK, double *DRHODT, double *DRHODS, double *RHOFULL, int outputs);

  void state_mwjf_explicit (double *SALTK, double *TEMPK, double *DRHODT, double *DRHODS, double *RHOFULL, int outputs);

  void state_mwjf_streams (double *SALTK, double *TEMPK, double *DRHODT, double *DRHODS, double *RHOFULL, int outputs);

  void state_mwjf_hybrid (double *SALTK, double *TEMPK, double *DRHODT, double *DRHODS, double *RHOFULL, int outputs);

  void cuda_init ();
  void my_cudamallochost (void **hostptr, int *size);

  void start_timer ();
  void stop_timer (float *);

  int compare (double *a, double *b, int N);

// kernels
  __global__ void mwjf_state_flex (double *SALTK, double *TEMPK, double *RHOFULL, double *DRHODT, double *DRHODS,
				   int jb, int je, int ib, int ie, int kb, int ke, int nx_block, int ny_block, int outputs);

  __global__ void mwjf_state_1Dprot (double *TEMPK, double *SALTK, double *RHOOUT, double *DRHODT, double *DRHODS, int n_outputs, int start_k, int end_k);

}


//include state initialization function
void cuda_state_initialize (double *pressz);
#include "state_init.cu"




int cuda_initialized = 0;

//my own Fortran entry for initializing CUDA
void
cuda_init () {
  if (cuda_initialized == 0) {
    cuda_initialized = 1;

    cudaSetDeviceFlags (cudaDeviceMapHost);
    cudaSetDevice (0);
    cudaDeviceSetCacheConfig (cudaFuncCachePreferL1);
  }
}


//my own Fortran entry for cudaMallocHost
void
my_cudamallochost (void **hostptr, int *p_size) {
  cudaError_t err;

  err = cudaHostAlloc ((void **) hostptr, (*p_size) * sizeof (double), cudaHostAllocMapped);
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaHostAlloc: %s\n", cudaGetErrorString (err));
  }

}


double *h_RHOFULL;
double *h_DRHODT;
double *h_DRHODS;


void
check_result (double *rho1, double *rho2, double *drhodt1, double *drhodt2, double *drhods1, double *drhods2, int outputs) {
  int res = 0;
  //correctness checking
  res = compare (rho1, rho2, KM * NX_BLOCK * NY_BLOCK);
  if (res > 0) {
    printf ("number of errors in RHOFULL: %d\n", res);
  }

  if (outputs == 3) {
    res = compare (drhodt1, drhodt2, KM * NX_BLOCK * NY_BLOCK);
    if (res > 0) {
      printf ("number of errors in DRHODT: %d\n", res);
    }

    res = compare (drhods1, drhods2, KM * NX_BLOCK * NY_BLOCK);
    if (res > 0) {
      printf ("number of errors in DRHODS: %d\n", res);
    }
  }
}



/* 
 * C wrapper function for calling the function that will call the GPU kernel
 *
 * This is the entry point for Fortran and calls the different CUDA implementations
 */
void
state_mwjf (double *SALTK, double *TEMPK, double *DRHODT, double *DRHODS, double *RHOFULL, double *pressz) {

  cudaError_t err;
  int k;

  //make sure cuda is initialized
  cuda_init ();

  //initialize mwjf state specific variables
  cuda_state_initialize (pressz);

  //setup streams
  for (k = 0; k < NSTREAMS; k++) {
    stream[k] = (cudaStream_t) - 1;
    err = cudaStreamCreate (&stream[k]);
    if (err != cudaSuccess) {
      fprintf (stderr, "Error doing cudaStreamCreate k=%d\n", k);
    }
  }
  CUDA_CHECK_ERROR ("After cudaStreamCreate");

  //allocate host buffers
  err = cudaHostAlloc ((void **) &h_RHOFULL, KM * NX_BLOCK * NY_BLOCK * sizeof (double), cudaHostAllocMapped);
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaHostAlloc h_RHOFULL: %s\n", cudaGetErrorString (err));
  }
  memset (h_RHOFULL, 0, KM * NX_BLOCK * NY_BLOCK * sizeof (double));

  err = cudaHostAlloc ((void **) &h_DRHODT, KM * NX_BLOCK * NY_BLOCK * sizeof (double), cudaHostAllocMapped);
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaHostAlloc h_DRHODT: %s\n", cudaGetErrorString (err));
  }
  memset (h_DRHODT, 0, KM * NX_BLOCK * NY_BLOCK * sizeof (double));

  err = cudaHostAlloc ((void **) &h_DRHODS, KM * NX_BLOCK * NY_BLOCK * sizeof (double), cudaHostAllocMapped);
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaHostAlloc h_DRHODS: %s\n", cudaGetErrorString (err));
  }
  memset (h_DRHODS, 0, KM * NX_BLOCK * NY_BLOCK * sizeof (double));

  int n_outputs = 1;

  for (k = 0; k < ITERATIONS; k++) {
    state_mwjf_explicit (SALTK, TEMPK, h_DRHODT, h_DRHODS, h_RHOFULL, n_outputs);
  }

  check_result (RHOFULL, h_RHOFULL, DRHODT, h_DRHODT, DRHODS, h_DRHODS, n_outputs);

  for (k = 0; k < ITERATIONS; k++) {
    state_mwjf_implicit (SALTK, TEMPK, h_DRHODT, h_DRHODS, h_RHOFULL, n_outputs);
  }

  check_result (RHOFULL, h_RHOFULL, DRHODT, h_DRHODT, DRHODS, h_DRHODS, n_outputs);

  for (k = 0; k < ITERATIONS; k++) {
    state_mwjf_streams (SALTK, TEMPK, h_DRHODT, h_DRHODS, h_RHOFULL, n_outputs);
  }

  check_result (RHOFULL, h_RHOFULL, DRHODT, h_DRHODT, DRHODS, h_DRHODS, n_outputs);

  for (k = 0; k < ITERATIONS; k++) {
    state_mwjf_hybrid (SALTK, TEMPK, h_DRHODT, h_DRHODS, h_RHOFULL, n_outputs);
  }

  check_result (RHOFULL, h_RHOFULL, DRHODT, h_DRHODT, DRHODS, h_DRHODS, n_outputs);

  //cleanup 
  for (int k = 0; k < NSTREAMS; k++) {
    cudaError_t err = cudaStreamDestroy (stream[k]);
    if (err != cudaSuccess) {
      fprintf (stderr, "Error doing cudaStreamDestroy k=%d\n", k);
    }
  }

  cudaFreeHost (h_RHOFULL);
  cudaFreeHost (h_DRHODT);
  cudaFreeHost (h_DRHODS);

  //this is used to tell the Nvidia Profiler to flush profiling info
  cudaDeviceReset ();

}




/*
 * Host code that invokes the state kernel
 *
 * The implicit implementation uses device-mapped host memory rather
 * than explicit memory copy statements. A different kernel is used
 * to ensure strictly coalesced access to system memory.
 *
 * outputs = either 1 or 3
 */
void
state_mwjf_implicit (double *SALTK, double *TEMPK, double *DRHODT, double *DRHODS, double *RHOFULL, int outputs) {

  float time;

  dim3 threads (256, 1);
  dim3 grid (1, 1);
  grid.x = (int) ceilf (((float) (NX_BLOCK * NY_BLOCK) / (float) 256));
  grid.y = KM;

  //MEASURE TOTAL TIME, INCLUDING PCIe TRANSFERS
  cudaDeviceSynchronize ();
  start_timer ();

  mwjf_state_1Dprot <<< grid, threads, 0, stream[1] >>> (TEMPK, SALTK, h_RHOFULL, h_DRHODT, h_DRHODS, outputs, 0, KM);

  cudaDeviceSynchronize ();
  stop_timer (&time);
  printf ("IMPLICIT:\t%.6f \n", time);

  //error checking
  cudaDeviceSynchronize ();
  CUDA_CHECK_ERROR ("After kernel execution");


}






/*
 * Host code that invokes the state kernel
 *
 * The explicit implementation uses explicit memory copy
 * statements to move all data to the GPU, executes the
 * GPU kernel, and uses memory copies to copy the output
 * data back to host memory. This implementation achieves
 * no overlap between transfers and/or computation.
 *
 * outputs = either 1 or 3
 */
void
state_mwjf_explicit (double *SALTK, double *TEMPK, double *DRHODT, double *DRHODS, double *RHOFULL, int outputs) {

  cudaError_t err;

  //device buffers
  double *d_SALTK;
  double *d_TEMPK;
  double *d_RHOFULL;
  double *d_DRHODT;
  double *d_DRHODS;

  //allocate device memory
  err = cudaMalloc ((void **) &d_TEMPK, KM * NX_BLOCK * NY_BLOCK * sizeof (double));
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMalloc d_TEMPK: %s\n", cudaGetErrorString (err));
  }

  err = cudaMalloc ((void **) &d_SALTK, KM * NX_BLOCK * NY_BLOCK * sizeof (double));
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMalloc d_SALTK: %s\n", cudaGetErrorString (err));
  }

  err = cudaMalloc ((void **) &d_RHOFULL, KM * NX_BLOCK * NY_BLOCK * sizeof (double));
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMalloc d_RHOFULL: %s\n", cudaGetErrorString (err));
  }

  if (outputs == 3) {
    err = cudaMalloc ((void **) &d_DRHODT, KM * NX_BLOCK * NY_BLOCK * sizeof (double));
    if (err != cudaSuccess) {
      fprintf (stderr, "Error in cudaMalloc d_DRHODT: %s\n", cudaGetErrorString (err));
    }

    err = cudaMalloc ((void **) &d_DRHODS, KM * NX_BLOCK * NY_BLOCK * sizeof (double));
    if (err != cudaSuccess) {
      fprintf (stderr, "Error in cudaMalloc d_DRHODS: %s\n", cudaGetErrorString (err));
    }
  }

  //error checking
  cudaDeviceSynchronize ();
  CUDA_CHECK_ERROR ("After memory setup");

  //setup execution parameters
  int jb = 0;
  int je = NY_BLOCK;
  int ib = 0;
  int ie = NX_BLOCK;

  dim3 threads (BLOCK_X, BLOCK_Y);
  dim3 grid (1, 1);
  grid.y = (int) ceilf ((float) NY_BLOCK / (float) (BLOCK_Y));
  grid.x = (int) ceilf ((float) NX_BLOCK / (float) (BLOCK_X));

  int k = 0;
  int lps = KM;
  float time;

  //TOTAL EXECUTION TIMING
  cudaDeviceSynchronize ();
  start_timer ();

  err = cudaMemcpyAsync (d_TEMPK, TEMPK, KM * NX_BLOCK * NY_BLOCK * sizeof (double), cudaMemcpyHostToDevice, stream[1]);
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMemcpy host to device d_TEMPK: %s\n", cudaGetErrorString (err));
  }

  err = cudaMemcpyAsync (d_SALTK, SALTK, KM * NX_BLOCK * NY_BLOCK * sizeof (double), cudaMemcpyHostToDevice, stream[1]);
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMemcpy host to device d_TEMPK: %s\n", cudaGetErrorString (err));
  }

  mwjf_state_flex <<< grid, threads, 0, stream[1] >>> (d_SALTK, d_TEMPK, d_RHOFULL, d_DRHODT, d_DRHODS,
						       jb, je, ib, ie, k, k + lps, NX_BLOCK, NY_BLOCK, outputs);

  err = cudaMemcpyAsync (h_RHOFULL, d_RHOFULL, KM * NX_BLOCK * NY_BLOCK * sizeof (double), cudaMemcpyDeviceToHost, stream[1]);
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMemcpy device to host d_RHOFULL: %s\n", cudaGetErrorString (err));
  }

  if (outputs == 3) {
    err = cudaMemcpyAsync (h_DRHODT, d_DRHODT, KM * NX_BLOCK * NY_BLOCK * sizeof (double), cudaMemcpyDeviceToHost, stream[1]);
    if (err != cudaSuccess) {
      fprintf (stderr, "Error in cudaMemcpy device to host d_DRHODT: %s\n", cudaGetErrorString (err));
    }

    err = cudaMemcpyAsync (h_DRHODS, d_DRHODS, KM * NX_BLOCK * NY_BLOCK * sizeof (double), cudaMemcpyDeviceToHost, stream[1]);
    if (err != cudaSuccess) {
      fprintf (stderr, "Error in cudaMemcpy device to host d_DRHODS: %s\n", cudaGetErrorString (err));
    }
  }

  //time checking
  cudaDeviceSynchronize ();
  stop_timer (&time);
  printf ("EXPLICIT:\t%.6f \n", time);

  //error checking
  cudaDeviceSynchronize ();
  CUDA_CHECK_ERROR ("After kernel execution");


  //NOW MEASURE KERNEL EXECUTION TIME ONLY
  cudaDeviceSynchronize ();
  start_timer ();

  mwjf_state_flex <<< grid, threads, 0, stream[1] >>> (d_SALTK, d_TEMPK, d_RHOFULL, d_DRHODT, d_DRHODS,
						       jb, je, ib, ie, k, k + lps, NX_BLOCK, NY_BLOCK, outputs);

  cudaDeviceSynchronize ();
  stop_timer (&time);

  //double ops = outputs == 1? 40.0 : 89.0;
  //double flops = ((NX_BLOCK*NY_BLOCK*lps*ops)/1000000000.0)/(time/1000.0)
  printf ("Kernel: \t%.6f \n", time);


  //Free device memory
  cudaFree (d_RHOFULL);
  if (outputs == 3) {
    cudaFree (d_DRHODT);
    cudaFree (d_DRHODS);
  }
  cudaFree (d_TEMPK);
  cudaFree (d_SALTK);

}




/*
 * Host code that invokes the state kernel
 *
 * The streams implementation uses CUDA streams combined
 * with explicit memory copy statements. This way transfers
 * in one stream may overlap with computation and transfers
 * in other streams.
 *
 * outputs = either 1 or 3
 */
void
state_mwjf_streams (double *SALTK, double *TEMPK, double *DRHODT, double *DRHODS, double *RHOFULL, int outputs) {

  cudaError_t err;
  int array_size = NX_BLOCK * NY_BLOCK;

  //device buffers
  double *d_SALTK;
  double *d_TEMPK;
  double *d_RHOFULL;
  double *d_DRHODT;
  double *d_DRHODS;

  //allocate device memory
  err = cudaMalloc ((void **) &d_TEMPK, KM * NX_BLOCK * NY_BLOCK * sizeof (double));
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMalloc d_TEMPK: %s\n", cudaGetErrorString (err));
  }

  err = cudaMalloc ((void **) &d_SALTK, KM * NX_BLOCK * NY_BLOCK * sizeof (double));
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMalloc d_SALTK: %s\n", cudaGetErrorString (err));
  }

  err = cudaMalloc ((void **) &d_RHOFULL, KM * NX_BLOCK * NY_BLOCK * sizeof (double));
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMalloc d_RHOFULL: %s\n", cudaGetErrorString (err));
  }

  if (outputs == 3) {
    err = cudaMalloc ((void **) &d_DRHODT, KM * NX_BLOCK * NY_BLOCK * sizeof (double));
    if (err != cudaSuccess) {
      fprintf (stderr, "Error in cudaMalloc d_DRHODT: %s\n", cudaGetErrorString (err));
    }

    err = cudaMalloc ((void **) &d_DRHODS, KM * NX_BLOCK * NY_BLOCK * sizeof (double));
    if (err != cudaSuccess) {
      fprintf (stderr, "Error in cudaMalloc d_DRHODS: %s\n", cudaGetErrorString (err));
    }
  }

  //error checking
  cudaDeviceSynchronize ();
  CUDA_CHECK_ERROR ("After memory setup");

  //setup execution parameters
  int jb = 0;
  int je = NY_BLOCK;
  int ib = 0;
  int ie = NX_BLOCK;

  dim3 threads (BLOCK_X, BLOCK_Y);
  dim3 grid (1, 1);
  grid.y = (int) ceilf ((float) NY_BLOCK / (float) (BLOCK_Y));
  grid.x = (int) ceilf ((float) NX_BLOCK / (float) (BLOCK_X));

  int lps = KM;			//levels to compute per stream
  int k = 0;

  nStreams = nStreams == -1 ? NSTREAMS : nStreams;
  k = 0;
  lps = KM / nStreams;

  //MEASURING TOTAL EXECUTION TIME
  float time;
  cudaDeviceSynchronize ();
  start_timer ();

  for (k = 0; k < KM; k += lps) {
    err = cudaMemcpyAsync (d_TEMPK + k * array_size, TEMPK + k * array_size, lps * array_size * sizeof (double), cudaMemcpyHostToDevice, stream[k]);
    if (err != cudaSuccess) {
      fprintf (stderr, "Error in cudaMemcpy host to device d_TEMPK in stream %d: %s\n", k, cudaGetErrorString (err));
    }

    err = cudaMemcpyAsync (d_SALTK + k * array_size, SALTK + k * array_size, lps * array_size * sizeof (double), cudaMemcpyHostToDevice, stream[k]);
    if (err != cudaSuccess) {
      fprintf (stderr, "Error in cudaMemcpy host to device d_SALTK in stream %d: %s\n", k, cudaGetErrorString (err));
    }
  }

  for (k = 0; k < KM; k += lps) {
    mwjf_state_flex <<< grid, threads, 0, stream[k] >>> (d_SALTK, d_TEMPK, d_RHOFULL, d_DRHODT, d_DRHODS,
							 jb, je, ib, ie, k, k + lps, NX_BLOCK, NY_BLOCK, outputs);

  }

  for (k = 0; k < KM; k += lps) {
    err = cudaMemcpyAsync (h_RHOFULL + k * array_size, d_RHOFULL + k * array_size, lps * array_size * sizeof (double), cudaMemcpyDeviceToHost, stream[k]);
    if (err != cudaSuccess) {
      fprintf (stderr, "Error in cudaMemcpy device to host d_RHOFULL in stream %d: %s\n", k, cudaGetErrorString (err));
    }

    if (outputs == 3) {
      err = cudaMemcpyAsync (h_DRHODT + k * array_size, d_DRHODT + k * array_size, lps * array_size * sizeof (double), cudaMemcpyDeviceToHost, stream[k]);
      if (err != cudaSuccess) {
	fprintf (stderr, "Error in cudaMemcpy device to host d_DRHODT in stream %d: %s\n", k, cudaGetErrorString (err));
      }

      err = cudaMemcpyAsync (h_DRHODS + k * array_size, d_DRHODS + k * array_size, lps * array_size * sizeof (double), cudaMemcpyDeviceToHost, stream[k]);
      if (err != cudaSuccess) {
	fprintf (stderr, "Error in cudaMemcpy device to host d_DRHODS in stream %d: %s\n", k, cudaGetErrorString (err));
      }
    }
  }

  //time checking
  cudaDeviceSynchronize ();
  stop_timer (&time);
  printf ("STREAMS:\t%.6f \n", time);

  //error checking
  cudaDeviceSynchronize ();
  CUDA_CHECK_ERROR ("After kernel execution");

  //Free device memory
  cudaFree (d_RHOFULL);
  if (outputs == 3) {
    cudaFree (d_DRHODT);
    cudaFree (d_DRHODS);
  }
  cudaFree (d_TEMPK);
  cudaFree (d_SALTK);

}



/*
 * Host code that invokes the state kernel
 *
 * The Hybrid implementation uses CUDA streams combined
 * with explicit memory copy statements for the input data
 * and uses device-mapped host memory to copy the output data
 * back to host memory. 
 *
 * outputs = either 1 or 3
 */
void
state_mwjf_hybrid (double *SALTK, double *TEMPK, double *DRHODT, double *DRHODS, double *RHOFULL, int outputs) {

  cudaError_t err;
  int array_size = NX_BLOCK * NY_BLOCK;

  //device buffers
  double *d_SALTK;
  double *d_TEMPK;

  //allocate device memory
  err = cudaMalloc ((void **) &d_TEMPK, KM * NX_BLOCK * NY_BLOCK * sizeof (double));
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMalloc d_TEMPK: %s\n", cudaGetErrorString (err));
  }

  err = cudaMalloc ((void **) &d_SALTK, KM * NX_BLOCK * NY_BLOCK * sizeof (double));
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMalloc d_SALTK: %s\n", cudaGetErrorString (err));
  }

  //error checking
  cudaDeviceSynchronize ();
  CUDA_CHECK_ERROR ("After memory setup");

  //setup execution parameters
  nStreams = nStreams == -1 ? NSTREAMS : nStreams;
  int lps = KM / nStreams;	//levels to compute per stream
  int k = 0;

  dim3 threads (256, 1);
  dim3 grid (1, 1);
  grid.x = (int) ceilf (((float) (NX_BLOCK * NY_BLOCK) / (float) 256));
  grid.y = lps;

  //MEASURING TOTAL EXECUTION TIME
  float time;
  cudaDeviceSynchronize ();
  start_timer ();

  for (k = 0; k < KM; k += lps) {
    err = cudaMemcpyAsync (d_TEMPK + k * array_size, TEMPK + k * array_size, lps * array_size * sizeof (double), cudaMemcpyHostToDevice, stream[k]);
    if (err != cudaSuccess) {
      fprintf (stderr, "Error in cudaMemcpy host to device d_TEMPK in stream %d: %s\n", k, cudaGetErrorString (err));
    }

    err = cudaMemcpyAsync (d_SALTK + k * array_size, SALTK + k * array_size, lps * array_size * sizeof (double), cudaMemcpyHostToDevice, stream[k]);
    if (err != cudaSuccess) {
      fprintf (stderr, "Error in cudaMemcpy host to device d_SALTK in stream %d: %s\n", k, cudaGetErrorString (err));
    }
  }

  for (k = 0; k < KM; k += lps) {
    mwjf_state_1Dprot <<< grid, threads, 0, stream[k] >>> (d_TEMPK, d_SALTK, h_RHOFULL, h_DRHODT, h_DRHODS, outputs, k, k + lps);
  }

  //time checking
  cudaDeviceSynchronize ();
  stop_timer (&time);
  printf ("HYBRID: \t%.6f \n", time);

  //error checking
  cudaDeviceSynchronize ();
  CUDA_CHECK_ERROR ("After kernel execution");

  //Free device memory
  cudaFree (d_TEMPK);
  cudaFree (d_SALTK);

}




/*
 * kernel for MWJF State
 *
 * This kernel is adapted from state_mod.F90 of the Parallel Ocean
 * Program.
 *
 * This kernel uses 1-dimensional thread indexing. Thread block
 * and thread indexes are recomputed to an index i and vertical
 * level k. This is to enforce strictly coalesced memory accesses.
 * 
 */
__global__ void
mwjf_state_1Dprot (double *TEMPK, double *SALTK, double *RHOOUT, double *DRHODT, double *DRHODS, int n_outputs, int start_k, int end_k) {

  //obtain global id
  int i = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
  int k = start_k + (i / (NX_BLOCK * NY_BLOCK));
  //obtain array index
  int index = i + start_k * NX_BLOCK * NY_BLOCK;

  double tq, sq, sqr, work1, work2, work3, work4, denomk;

  if (i < NX_BLOCK * NY_BLOCK * (end_k - start_k)) {

    tq = min (TEMPK[index], TMAX);
    tq = max (tq, TMIN);
    sq = min (SALTK[index], SMAX);
    sq = 1000.0 * max (sq, SMIN);

    sqr = sqrt (sq);

    work1 = d_mwjfnums0t0[k] + tq * (d_mwjfnums0t1 + tq * (d_mwjfnums0t2[k] + d_mwjfnums0t3 * tq)) +
      sq * (d_mwjfnums1t0[k] + d_mwjfnums1t1 * tq + d_mwjfnums2t0 * sq);

    work2 = d_mwjfdens0t0[k] + tq * (d_mwjfdens0t1[k] + tq * (d_mwjfdens0t2 +
							      tq * (d_mwjfdens0t3[k] + d_mwjfdens0t4 * tq))) +
      sq * (d_mwjfdens1t0 + tq * (d_mwjfdens1t1 + tq * tq * d_mwjfdens1t3) + sqr * (d_mwjfdensqt0 + tq * tq * d_mwjfdensqt2));

    denomk = 1.0 / work2;
    RHOOUT[index] = work1 * denomk;

    if (n_outputs == 3) {
      work3 =			// dP_1/dT
	d_mwjfnums0t1 + tq * (2.0 * d_mwjfnums0t2[k] + 3.0 * d_mwjfnums0t3 * tq) + d_mwjfnums1t1 * sq;

      work4 =			// dP_2/dT
	d_mwjfdens0t1[k] + sq * d_mwjfdens1t1 +
	tq * (2.0 * (d_mwjfdens0t2 + sq * sqr * d_mwjfdensqt2) + tq * (3.0 * (d_mwjfdens0t3[k] + sq * d_mwjfdens1t3) + tq * 4.0 * d_mwjfdens0t4));

      DRHODT[index] = (work3 - work1 * denomk * work4) * denomk;

      work3 =			// dP_1/dS
	d_mwjfnums1t0[k] + d_mwjfnums1t1 * tq + 2.0 * d_mwjfnums2t0 * sq;

      work4 = d_mwjfdens1t0 +	// dP_2/dS
	tq * (d_mwjfdens1t1 + tq * tq * d_mwjfdens1t3) + 1.5 * sqr * (d_mwjfdensqt0 + tq * tq * d_mwjfdensqt2);

      DRHODS[index] = (work3 - work1 * denomk * work4) * denomk * 1000.0;
    }

  }

}



/*
 * kernel for MWJF State
 *
 * This kernel is adapted from state_mod.F90 of the Parallel Ocean
 * Program.
 *
 * This kernel flexible in that it does not make compile time
 * assumptions about the domain sizes. As such it uses more
 * parameters to mark the start and end of the computed range 
 * within the domain.
 *
 */
__global__ void
mwjf_state_flex (double *SALTK, double *TEMPK, double *RHOFULL, double *DRHODT, double *DRHODS,
		 int jb, int je, int ib, int ie, int kb, int ke, int nx_block, int ny_block, int outputs) {

  //obtain global ids
  int j = jb + threadIdx.y + blockIdx.y * BLOCK_Y;
  int i = ib + threadIdx.x + blockIdx.x * BLOCK_X;
  int k;

  double tq, sq, sqr, work1, work2, work3, work4, denomk;

  if (j < je && i < ie) {

    for (k = kb; k < ke; k++) {

      tq = min (TEMPK[i + j * nx_block + k * ny_block * nx_block], TMAX);
      tq = max (tq, TMIN);

      sq = min (SALTK[i + j * nx_block + k * ny_block * nx_block], SMAX);
      sq = 1000.0 * max (sq, SMIN);

      sqr = sqrt (sq);

      work1 = d_mwjfnums0t0[k] + tq * (d_mwjfnums0t1 + tq * (d_mwjfnums0t2[k] + d_mwjfnums0t3 * tq)) +
	sq * (d_mwjfnums1t0[k] + d_mwjfnums1t1 * tq + d_mwjfnums2t0 * sq);

      work2 = d_mwjfdens0t0[k] + tq * (d_mwjfdens0t1[k] + tq * (d_mwjfdens0t2 +
								tq * (d_mwjfdens0t3[k] + d_mwjfdens0t4 * tq))) +
	sq * (d_mwjfdens1t0 + tq * (d_mwjfdens1t1 + tq * tq * d_mwjfdens1t3) + sqr * (d_mwjfdensqt0 + tq * tq * d_mwjfdensqt2));

      denomk = 1.0 / work2;

      RHOFULL[i + j * nx_block + k * ny_block * nx_block] = work1 * denomk;

      if (outputs == 3) {

	work3 =			// dP_1/dT
	  d_mwjfnums0t1 + tq * (2.0 * d_mwjfnums0t2[k] + 3.0 * d_mwjfnums0t3 * tq) + d_mwjfnums1t1 * sq;

	work4 =			// dP_2/dT
	  d_mwjfdens0t1[k] + sq * d_mwjfdens1t1 +
	  tq * (2.0 * (d_mwjfdens0t2 + sq * sqr * d_mwjfdensqt2) + tq * (3.0 * (d_mwjfdens0t3[k] + sq * d_mwjfdens1t3) + tq * 4.0 * d_mwjfdens0t4));

	DRHODT[i + j * nx_block + k * ny_block * nx_block] = (work3 - work1 * denomk * work4) * denomk;

	work3 =			// dP_1/dS
	  d_mwjfnums1t0[k] + d_mwjfnums1t1 * tq + 2.0 * d_mwjfnums2t0 * sq;

	work4 = d_mwjfdens1t0 +	// dP_2/dS
	  tq * (d_mwjfdens1t1 + tq * tq * d_mwjfdens1t3) + 1.5 * sqr * (d_mwjfdensqt0 + tq * tq * d_mwjfdensqt2);

	DRHODS[i + j * nx_block + k * ny_block * nx_block] = (work3 - work1 * denomk * work4) * denomk * 1000.0;

      }

    }
  }

}







int
compare (double *a1, double *a2, int N) {
  int i = 0, res = 0;
  int print = 0;
  int zero_one = 0;
  int zero_two = 0;

  double eps = 0.0000000000001;

  for (i = 0; i < N; i++) {

    if (a1[i] < eps && a1[i] > -eps) {
      zero_one++;
    }
    if (a2[i] < eps && a2[i] > -eps) {
      zero_two++;
    }

    if (isnan (a1[i]) || isnan (a2[i])) {
      res++;
      if (print < 10) {
	print++;
	fprintf (stderr, "Error detected isnan at i=%d, a1= %30.27e a2= %30.27e\n", i, a1[i], a2[i]);
      }
    }

    double diff = a1[i] - a2[i];
    if (diff > eps || diff < -eps) {
      res++;
      if (print < 10) {
	print++;
	fprintf (stderr, "Error detected at i=%d, \t a1= \t %30.27e \t a2= \t %30.27e\n", i, a1[i], a2[i]);
      }
    }

  }

  if (zero_one > 3 * (N / 4)) {
    fprintf (stderr, "Error: array1 contains %d zeros\n", zero_one);
  }
  if (zero_two > 3 * (N / 4)) {
    fprintf (stderr, "Error: array2 contains %d zeros\n", zero_two);
  }

  if (zero_one != zero_two) {
    fprintf (stderr, "Error: number of zeros in arrays dont correspond zero1=%d, zero2=%d\n", zero_one, zero_two);
  }

  //fprintf(stdout,"Number of errors in GPU result: %d\n",res);

  return res;
}
