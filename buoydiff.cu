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
 * overlapping CPU-GPU communication and GPU computation of a
 * buoydiff kernel.
 * 
 * @author Ben van Werkhoven <b.vanwerkhoven@esciencecenter.nl>
 * 
 */

#include <stdio.h>
#include "domain.h"

//CUDA nStreams
#define NSTREAMS 42
int nStreams = -1;
cudaStream_t stream[KM];
cudaEvent_t event_htod[KM];
cudaEvent_t event_comp[KM];
cudaEvent_t event_dtoh[KM];

#define ITERATIONS 1


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


  void buoydiff_entry (double *DBLOC, double *DBSFC, double *TRCR, int *kmt, double *pressz);

  void buoydiff_implicit (double *DBLOC, double *DBSFC, double *TRCR);
  void buoydiff_explicit (double *DBLOC, double *DBSFC, double *TRCR);
  void buoydiff_streams (double *DBLOC, double *DBSFC, double *TRCR);
  void buoydiff_hybrid (double *DBLOC, double *DBSFC, double *TRCR);

  void devsync () {
    cudaDeviceSynchronize ();
  }

  void cuda_init ();
  void my_cudamallochost (void **hostptr, int *size, int *type);

  void start_timer ();
  void stop_timer (float *);

  int compare (double *a, double *b, int N);

  __global__ void buoydiff_kernel1D (double *DBLOC, double *DBSFC, double *TEMP, double *SALT, int *KMT, int start_k, int end_k);
  __global__ void buoydiff_kernel (double *DBLOC, double *DBSFC, double *TEMP, double *SALT, int *KMT, int start_k, int end_k);

}

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

//Fortran entry for cudaMallocHost
//type denotes the number of bytes per value in size
void
my_cudamallochost (void **hostptr, int *p_size, int *p_type) {
  cudaError_t err;

  err = cudaHostAlloc ((void **) hostptr, (*p_size) * (*p_type), cudaHostAllocMapped);
  if (err != cudaSuccess)
    fprintf (stderr, "Error in cudaHostAlloc: %s\n", cudaGetErrorString (err));

}

//device pointer for kmt in GPU global memory
int *d_kmt;


void
check_results (double *dbloc1, double *dbloc2, double *dbsfc1, double *dbsfc2) {
  int res;

  res = compare (dbloc1, dbloc2, NX_BLOCK * NY_BLOCK * KM);
  if (res > 0)
    printf ("number of errors in DBLOC: %d\n", res);
  res = compare (dbsfc1, dbsfc2, NX_BLOCK * NY_BLOCK * KM);
  if (res > 0)
    printf ("number of errors in DBSFC: %d\n", res);
}


void
buoydiff_entry (double *DBLOC, double *DBSFC, double *TRCR, int *kmt, double *pressz) {

  cudaError_t err;
  int k;

  cuda_init ();

  //initialize mwjf state specific variables
  cuda_state_initialize (pressz);

  //setup kmt info on the GPU
  err = cudaMalloc (&d_kmt, NX_BLOCK * NY_BLOCK * sizeof (int));
  if (err != cudaSuccess)
    fprintf (stderr, "Error doing cudaMalloc d_kmt\n");
  err = cudaMemcpy (d_kmt, kmt, NX_BLOCK * NY_BLOCK * sizeof (int), cudaMemcpyHostToDevice);
  if (err != cudaSuccess)
    fprintf (stderr, "Error doing cudaMemcpyHostToDevice KMT\n");

  //setup streams
  for (k = 0; k < NSTREAMS; k++) {
    err = cudaStreamCreate (&stream[k]);
    if (err != cudaSuccess)
      fprintf (stderr, "Error in cudaStreamCreate: %s\n", cudaGetErrorString (err));
  }

  //create cuda events
  for (k = 0; k < KM; k++) {
    err = cudaEventCreate (&event_htod[k]);
    if (err != cudaSuccess)
      fprintf (stderr, "Error in cudaEventCreate htod: %s\n", cudaGetErrorString (err));
    err = cudaEventCreate (&event_comp[k]);
    if (err != cudaSuccess)
      fprintf (stderr, "Error in cudaEventCreate comp: %s\n", cudaGetErrorString (err));
    err = cudaEventCreate (&event_dtoh[k]);
    if (err != cudaSuccess)
      fprintf (stderr, "Error in cudaEventCreate dtoh: %s\n", cudaGetErrorString (err));
  }

  //host arrays to store GPU result for comparision
  double *h_DBLOC;
  double *h_DBSFC;
  err = cudaHostAlloc ((void **) &h_DBLOC, NX_BLOCK * NY_BLOCK * KM * sizeof (double), cudaHostAllocMapped);
  if (err != cudaSuccess)
    fprintf (stderr, "Error in cudaHostAlloc h_DBLOC: %s\n", cudaGetErrorString (err));
  err = cudaHostAlloc ((void **) &h_DBSFC, NX_BLOCK * NY_BLOCK * KM * sizeof (double), cudaHostAllocMapped);
  if (err != cudaSuccess)
    fprintf (stderr, "Error in cudaHostAlloc h_DBSFC: %s\n", cudaGetErrorString (err));
  memset (h_DBLOC, 0, NX_BLOCK * NY_BLOCK * KM * sizeof (double));
  memset (h_DBSFC, 0, NX_BLOCK * NY_BLOCK * KM * sizeof (double));

  for (k = 0; k < ITERATIONS; k++) {
    buoydiff_explicit (h_DBLOC, h_DBSFC, TRCR);
  }

  check_results (h_DBLOC, DBLOC, h_DBSFC, DBSFC);

  for (k = 0; k < ITERATIONS; k++) {
    buoydiff_implicit (h_DBLOC, h_DBSFC, TRCR);
  }

  check_results (h_DBLOC, DBLOC, h_DBSFC, DBSFC);

  for (k = 0; k < ITERATIONS; k++) {
    buoydiff_streams (h_DBLOC, h_DBSFC, TRCR);
  }

  check_results (h_DBLOC, DBLOC, h_DBSFC, DBSFC);

  for (k = 0; k < ITERATIONS; k++) {
    buoydiff_hybrid (h_DBLOC, h_DBSFC, TRCR);
  }

  check_results (h_DBLOC, DBLOC, h_DBSFC, DBSFC);

  //free host memory
  cudaFreeHost (h_DBLOC);
  cudaFreeHost (h_DBSFC);

  //this is used to tell the Nvidia Profiler to flush profiling info
  cudaDeviceReset ();

}


/*
 * Host code that invokes the buoydiff kernel
 *
 * The implicit implementation uses device-mapped host memory rather
 * than explicit memory copy statements. A different kernel is used
 * to ensure strictly coalesced access to system memory.
 *
 */
void
buoydiff_implicit (double *DBLOC, double *DBSFC, double *TRCR) {

  float time;

  //setup execution parameters, assuming 1D for now
  dim3 threads (256, 1);
  dim3 grid (1, 1);
  grid.x = (int) ceilf (((float) (NX_BLOCK * NY_BLOCK) / (float) threads.x));
  grid.y = (KM);

  //host to device copies, assuming mapped mem for now
  cudaDeviceSynchronize ();
  CUDA_CHECK_ERROR ("Before buoydiff_gpu kernel execution");

  //start measurement
  cudaDeviceSynchronize ();
  start_timer ();

  buoydiff_kernel1D <<< grid, threads, 0, stream[1] >>> (DBLOC, DBSFC, TRCR, TRCR + (NX_BLOCK * NY_BLOCK * KM), d_kmt, 0, KM);

  cudaDeviceSynchronize ();
  stop_timer (&time);
  printf ("IMPLICIT: \t%.6f \n", time);

  cudaDeviceSynchronize ();
  CUDA_CHECK_ERROR ("After buoydiff implicit");

}

/*
 * Host code that invokes the buoydiff kernel
 *
 * The explicit implementation uses explicit memory copy
 * statements to move all data to the GPU, executes the
 * GPU kernel, and uses memory copies to copy the output
 * data back to host memory. This implementation achieves
 * no overlap between transfers and/or computation.
 *
 */
void
buoydiff_explicit (double *DBLOC, double *DBSFC, double *TRCR) {

  cudaError_t err;

  //allocate device memory
  double *d_DBLOC;
  double *d_DBSFC;
  double *d_TRCR;

  err = cudaMalloc ((void **) &d_DBLOC, NX_BLOCK * NY_BLOCK * KM * sizeof (double));
  if (err != cudaSuccess)
    fprintf (stderr, "Error in popMalloc d_DBLOC: %s\n", cudaGetErrorString (err));
  err = cudaMalloc ((void **) &d_DBSFC, NX_BLOCK * NY_BLOCK * KM * sizeof (double));
  if (err != cudaSuccess)
    fprintf (stderr, "Error in popMalloc d_DBSFC: %s\n", cudaGetErrorString (err));
  err = cudaMalloc ((void **) &d_TRCR, NX_BLOCK * NY_BLOCK * KM * 2 * sizeof (double));
  if (err != cudaSuccess)
    fprintf (stderr, "Error in popMalloc d_TRCR %s\n", cudaGetErrorString (err));

  cudaDeviceSynchronize ();
  CUDA_CHECK_ERROR ("After memory setup");

  //setup execution parameters
  dim3 threads (16, 16);
  dim3 grid (1, 1);
  grid.y = (int) ceilf ((float) NY_BLOCK / (float) (threads.y));
  grid.x = (int) ceilf ((float) NX_BLOCK / (float) (threads.x));

  //setup timers
  float time;
  cudaDeviceSynchronize ();
  start_timer ();

  err = cudaMemcpyAsync (d_TRCR, TRCR, NX_BLOCK * NY_BLOCK * KM * 2 * sizeof (double), cudaMemcpyHostToDevice, stream[1]);
  if (err != cudaSuccess)
    fprintf (stderr, "Error in cudaMemcpy host to device TRCR: %s\n", cudaGetErrorString (err));

  buoydiff_kernel <<< grid, threads, 0, stream[1] >>> (d_DBLOC, d_DBSFC, d_TRCR, d_TRCR + (NX_BLOCK * NY_BLOCK * KM), d_kmt, 0, KM);

  err = cudaMemcpyAsync (DBLOC, d_DBLOC, NX_BLOCK * NY_BLOCK * KM * sizeof (double), cudaMemcpyDeviceToHost, stream[1]);
  if (err != cudaSuccess)
    fprintf (stderr, "Error in cudaMemcpy device to host DBLOC: %s\n", cudaGetErrorString (err));

  err = cudaMemcpyAsync (DBSFC, d_DBSFC, NX_BLOCK * NY_BLOCK * KM * sizeof (double), cudaMemcpyDeviceToHost, stream[1]);
  if (err != cudaSuccess)
    fprintf (stderr, "Error in cudaMemcpy device to host DBSFC: %s\n", cudaGetErrorString (err));

  cudaDeviceSynchronize ();
  stop_timer (&time);
  printf ("EXPLICIT:\t%.6f \n", time);

  cudaDeviceSynchronize ();
  CUDA_CHECK_ERROR ("After buoydiff explicit");

  //setup timers
  cudaDeviceSynchronize ();
  start_timer ();

  buoydiff_kernel <<< grid, threads, 0, stream[1] >>> (d_DBLOC, d_DBSFC, d_TRCR, d_TRCR + (NX_BLOCK * NY_BLOCK * KM), d_kmt, 0, KM);

  cudaDeviceSynchronize ();
  stop_timer (&time);
  printf ("Kernel:\t%.6f \n", time);

  cudaFree (d_DBLOC);
  cudaFree (d_DBSFC);
  cudaFree (d_TRCR);

}

/*
 * Host code that invokes the buoydiff kernel
 *
 * The streams implementation uses CUDA streams combined
 * with explicit memory copy statements. This way transfers
 * in one stream may overlap with computation and transfers
 * in other streams.
 *
 */
void
buoydiff_streams (double *DBLOC, double *DBSFC, double *TRCR) {

  cudaError_t err;

  //allocate device memory
  double *d_DBLOC;
  double *d_DBSFC;
  double *d_TRCR;
  double *d_SALT;
  double *SALT = TRCR + NX_BLOCK * NY_BLOCK * KM;

  err = cudaMalloc ((void **) &d_DBLOC, NX_BLOCK * NY_BLOCK * KM * sizeof (double));
  if (err != cudaSuccess)
    fprintf (stderr, "Error in popMalloc d_DBLOC: %s\n", cudaGetErrorString (err));
  err = cudaMalloc ((void **) &d_DBSFC, NX_BLOCK * NY_BLOCK * KM * sizeof (double));
  if (err != cudaSuccess)
    fprintf (stderr, "Error in popMalloc d_DBSFC: %s\n", cudaGetErrorString (err));
  err = cudaMalloc ((void **) &d_TRCR, NX_BLOCK * NY_BLOCK * KM * 2 * sizeof (double));
  if (err != cudaSuccess)
    fprintf (stderr, "Error in popMalloc d_TRCR %s\n", cudaGetErrorString (err));
  d_SALT = d_TRCR + NX_BLOCK * NY_BLOCK * KM;

  cudaDeviceSynchronize ();
  CUDA_CHECK_ERROR ("After memory setup");

  //setup execution parameters
  dim3 threads (32, 8);
  dim3 grid (1, 1);
  grid.y = (int) ceilf ((float) NY_BLOCK / (float) (threads.y));
  grid.x = (int) ceilf ((float) NX_BLOCK / (float) (threads.x));

  int array_size = NX_BLOCK * NY_BLOCK;
  int lps = 1;			//levels to compute per stream
  int k = 0;
  nStreams = nStreams == -1 ? NSTREAMS : nStreams;
  k = 0;
  lps = KM / nStreams;
  if (lps * nStreams != KM) {
    fprintf (stderr, "Error: Levels Per Stream is not a divisor of nStreams!\n");
  }

  //setup timers
  float time;
  cudaDeviceSynchronize ();
  start_timer ();

  for (k = 0; k < KM; k += lps) {

    //doing sync before transfers
    if (k > 0) {
      //wait for memcpy in stream k=0 to be complete
      err = cudaStreamWaitEvent (stream[k], event_htod[0], 0);
      if (err != cudaSuccess)
	fprintf (stderr, "Error in cudaStreamWaitEvent htod k=0: %s\n", cudaGetErrorString (err));
      //wait for memcpy in stream k-1 to be complete
      err = cudaStreamWaitEvent (stream[k], event_htod[k - lps], 0);
      if (err != cudaSuccess)
	fprintf (stderr, "Error in cudaStreamWaitEvent htod k-1: %s\n", cudaGetErrorString (err));
    }

    err = cudaMemcpyAsync (d_TRCR + k * array_size, TRCR + k * array_size, lps * array_size * sizeof (double), cudaMemcpyHostToDevice, stream[k]);
    if (err != cudaSuccess)
      fprintf (stderr, "Error in cudaMemcpy host to device TRCR: %s\n", cudaGetErrorString (err));
    err = cudaMemcpyAsync (d_SALT + k * array_size, SALT + k * array_size, lps * array_size * sizeof (double), cudaMemcpyHostToDevice, stream[k]);
    if (err != cudaSuccess)
      fprintf (stderr, "Error in cudaMemcpy host to device TRCR: %s\n", cudaGetErrorString (err));

    //record cuda event
    err = cudaEventRecord (event_htod[k], stream[k]);
    if (err != cudaSuccess)
      fprintf (stderr, "Error in cudaEventRecord htod: %s\n", cudaGetErrorString (err));
  }

  for (k = 0; k < KM; k += lps) {

    buoydiff_kernel <<< grid, threads, 0, stream[k] >>> (d_DBLOC, d_DBSFC, d_TRCR, d_SALT, d_kmt, k, k + lps);

    err = cudaEventRecord (event_comp[k], stream[k]);
    if (err != cudaSuccess)
      fprintf (stderr, "Error in cudaEventRecord htod: %s\n", cudaGetErrorString (err));
  }

  for (k = 0; k < KM; k += lps) {
    err = cudaMemcpyAsync (DBSFC + k * array_size, d_DBSFC + k * array_size, lps * array_size * sizeof (double), cudaMemcpyDeviceToHost, stream[k]);
    if (err != cudaSuccess)
      fprintf (stderr, "Error in cudaMemcpy device to host DBSFC: %s\n", cudaGetErrorString (err));

    if (k < KM - lps) {
      //wait for computation in stream k+1 to be complete
      err = cudaStreamWaitEvent (stream[k], event_comp[k + lps], 0);
      if (err != cudaSuccess)
	fprintf (stderr, "Error in cudaStreamWaitEvent comp k+1: %s\n", cudaGetErrorString (err));
    }
    err = cudaMemcpyAsync (DBLOC + k * array_size, d_DBLOC + k * array_size, lps * array_size * sizeof (double), cudaMemcpyDeviceToHost, stream[k]);
    if (err != cudaSuccess)
      fprintf (stderr, "Error in cudaMemcpy device to host DBLOC: %s\n", cudaGetErrorString (err));
  }
  //time checking
  cudaDeviceSynchronize ();
  stop_timer (&time);
  printf ("Streams:\t%.6f \n", time);

  cudaDeviceSynchronize ();
  CUDA_CHECK_ERROR ("After buoydiff streams");

  cudaFree (d_DBLOC);
  cudaFree (d_DBSFC);
  cudaFree (d_TRCR);

}



/*
 * Host code that invokes the buoydiff kernel
 *
 * The Hybrid implementation uses CUDA streams combined
 * with explicit memory copy statements for the input data
 * and uses device-mapped host memory to copy the output data
 * back to host memory. 
 *
 */
void
buoydiff_hybrid (double *DBLOC, double *DBSFC, double *TRCR) {

  cudaError_t err;

  //allocate device memory
  double *d_TRCR;
  double *d_SALT;
  double *SALT = TRCR + NX_BLOCK * NY_BLOCK * KM;

  err = cudaMalloc ((void **) &d_TRCR, NX_BLOCK * NY_BLOCK * KM * 2 * sizeof (double));
  if (err != cudaSuccess)
    fprintf (stderr, "Error in popMalloc d_TRCR %s\n", cudaGetErrorString (err));
  d_SALT = d_TRCR + NX_BLOCK * NY_BLOCK * KM;

  cudaDeviceSynchronize ();
  CUDA_CHECK_ERROR ("After memory setup");

  //setup execution parameters
  dim3 threads (256, 1);
  dim3 grid (1, 1);
  grid.x = (int) ceilf (((float) (NX_BLOCK * NY_BLOCK) / (float) threads.x));
  grid.y = 1;

  int array_size = NX_BLOCK * NY_BLOCK;
  int lps = 1;			//levels to compute per stream
  int k = 0;
  nStreams = nStreams == -1 ? NSTREAMS : nStreams;
  k = 0;
  lps = KM / nStreams;
  if (lps * nStreams != KM) {
    fprintf (stderr, "Error: Levels Per Stream is not a divisor of nStreams!\n");
  }
  grid.y = lps;

  //setup timers
  float time;
  cudaDeviceSynchronize ();
  start_timer ();

  for (k = 0; k < KM; k += lps) {

    //doing sync before transfers
    if (k > 0) {
      //wait for memcpy in stream k=0 to be complete
      err = cudaStreamWaitEvent (stream[k], event_htod[0], 0);
      if (err != cudaSuccess)
	fprintf (stderr, "Error in cudaStreamWaitEvent htod k=0: %s\n", cudaGetErrorString (err));
      //wait for memcpy in stream k-1 to be complete
      err = cudaStreamWaitEvent (stream[k], event_htod[k - lps], 0);
      if (err != cudaSuccess)
	fprintf (stderr, "Error in cudaStreamWaitEvent htod k-1: %s\n", cudaGetErrorString (err));
    }

    err = cudaMemcpyAsync (d_TRCR + k * array_size, TRCR + k * array_size, lps * array_size * sizeof (double), cudaMemcpyHostToDevice, stream[k]);
    if (err != cudaSuccess)
      fprintf (stderr, "Error in cudaMemcpy host to device TRCR: %s\n", cudaGetErrorString (err));
    err = cudaMemcpyAsync (d_SALT + k * array_size, SALT + k * array_size, lps * array_size * sizeof (double), cudaMemcpyHostToDevice, stream[k]);
    if (err != cudaSuccess)
      fprintf (stderr, "Error in cudaMemcpy host to device TRCR: %s\n", cudaGetErrorString (err));

    //record cuda event
    err = cudaEventRecord (event_htod[k], stream[k]);
    if (err != cudaSuccess)
      fprintf (stderr, "Error in cudaEventRecord htod: %s\n", cudaGetErrorString (err));
  }

  for (k = 0; k < KM; k += lps) {

    buoydiff_kernel1D <<< grid, threads, 0, stream[k] >>> (DBLOC, DBSFC, d_TRCR, d_SALT, d_kmt, k, k + lps);

  }

  //time checking
  cudaDeviceSynchronize ();
  stop_timer (&time);
  printf ("HYBRID: \t%.6f \n", time);

  cudaDeviceSynchronize ();
  CUDA_CHECK_ERROR ("After buoydiff hybrid");

  cudaFree (d_TRCR);

}




__device__ double
state (double temp, double salt, int k) {
  double tq, sq, sqr, work1, work2, denomk;

  tq = min (temp, TMAX);
  tq = max (tq, TMIN);

  sq = min (salt, SMAX);
  sq = 1000.0 * max (sq, SMIN);

  sqr = sqrt (sq);

  work1 = d_mwjfnums0t0[k] + tq * (d_mwjfnums0t1 + tq * (d_mwjfnums0t2[k] + d_mwjfnums0t3 * tq)) +
    sq * (d_mwjfnums1t0[k] + d_mwjfnums1t1 * tq + d_mwjfnums2t0 * sq);

  work2 = d_mwjfdens0t0[k] + tq * (d_mwjfdens0t1[k] + tq * (d_mwjfdens0t2 +
							    tq * (d_mwjfdens0t3[k] + d_mwjfdens0t4 * tq))) +
    sq * (d_mwjfdens1t0 + tq * (d_mwjfdens1t1 + tq * tq * d_mwjfdens1t3) + sqr * (d_mwjfdensqt0 + tq * tq * d_mwjfdensqt2));

  denomk = 1.0 / work2;

  return work1 * denomk;
}



/*
 * kernel for buoydiff  
 *
 * This kernel is adapted from the buoydiff subroutine in 
 * vmix_kpp.F90 of the Parallel Ocean Program.
 *
 * This kernel computes the buoyancy differences between different vertical
 * levels in the ocean based on temperature and salinity. Internally, the
 * state kernel is used to compute the potential density. The computation
 * of buoyancy differences at level k requires the density of both the
 * surface level and k-1 displaced to level k, as well as the water density
 * at level k.
 *
 * This implementation uses 1-dimensional indexing to ensure strict coalescing
 * when transferring data over PCIe. 
 *
 */
__global__ void
buoydiff_kernel1D (double *DBLOC, double *DBSFC, double *TEMP, double *SALT, int *KMT, int start_k, int end_k) {

  int i = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
  int k = start_k + (i / (NX_BLOCK * NY_BLOCK));

  //compute index at surface level
  int sfci = i - (i / (NX_BLOCK * NY_BLOCK)) * (NX_BLOCK * NY_BLOCK);

  //obtain array indexes at level k and k-1
  int index = i + start_k * NX_BLOCK * NY_BLOCK;
  int indexmk = index - (NX_BLOCK * NY_BLOCK);

  double rho1, rhokm, rhok;
  double dbloc;

  if (i < NX_BLOCK * NY_BLOCK * (end_k - start_k)) {

    //if k==0 we write DBSFC=0 and exit, as DBLOC for k=0 is computed by thread at k=1
    if (k == 0) {
      DBSFC[index] = 0.0;
      return;
    }

    //!  calculate DBLOC and DBSFC for all other levels
    rho1 = state (TEMP[sfci], SALT[sfci], k);
    rhokm = state (TEMP[indexmk], SALT[indexmk], k);
    rhok = state (TEMP[index], SALT[index], k);

    if (rhok != 0.0) {		//prevent div by zero
      DBSFC[index] = d_grav * (1.0 - rho1 / rhok);
      dbloc = d_grav * (1.0 - rhokm / rhok);
    }
    else {
      DBSFC[index] = 0.0;
      dbloc = 0.0;
    }

    //zero if on land
    //why DBSFC isnt zeroed here is a mystery to me -Ben
    if (k >= KMT[sfci]) {	//-1 removed because FORTRAN array index starts at 1
      dbloc = 0.0;
    }
    if (k == KM - 1) {
      DBLOC[index] = 0.0;
    }

    DBLOC[indexmk] = dbloc;

  }
}


/*
 * kernel for buoydiff
 *
 * This kernel is adapted from the buoydiff subroutine in 
 * vmix_kpp.F90 of the Parallel Ocean Program.
 *
 * This kernel computes the buoyancy differences between different vertical
 * levels in the ocean based on temperature and salinity. Internally, the
 * state kernel is used to compute the potential density. The computation
 * of buoyancy differences at level k requires the density of both the
 * surface level and k-1 displaced to level k, as well as the water density
 * at level k.
 *
 */
__global__ void
buoydiff_kernel (double *DBLOC, double *DBSFC, double *TEMP, double *SALT, int *KMT, int start_k, int end_k) {

  //obtain indices
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int k = start_k;

  double rho1, rhokm, rhok;

  if (j < NY_BLOCK && i < NX_BLOCK) {
    //prefetch values from sfc
    double tempsfc = TEMP[i + j * NX_BLOCK];
    double saltsfc = SALT[i + j * NX_BLOCK];
    int kmt = KMT[i + j * NX_BLOCK];

    for (k = start_k; k < end_k; k++) {

      if (k == 0) {
	DBSFC[i + j * NX_BLOCK] = 0.0;
      }
      else {

	if (k == KM - 1) {
	  DBLOC[i + j * NX_BLOCK + k * NY_BLOCK * NX_BLOCK] = 0.0;
	}

	rho1 = state (tempsfc, saltsfc, k);

	rhokm = state (TEMP[i + j * NX_BLOCK + (k - 1) * NY_BLOCK * NX_BLOCK], SALT[i + j * NX_BLOCK + (k - 1) * NY_BLOCK * NX_BLOCK], k);

	rhok = state (TEMP[i + j * NX_BLOCK + k * NY_BLOCK * NX_BLOCK], SALT[i + j * NX_BLOCK + k * NY_BLOCK * NX_BLOCK], k);

	if (rhok != 0.0) {	//prevent div by zero
	  DBSFC[i + j * NX_BLOCK + k * NY_BLOCK * NX_BLOCK] = d_grav * (1.0 - rho1 / rhok);
	  DBLOC[i + j * NX_BLOCK + (k - 1) * NY_BLOCK * NX_BLOCK] = d_grav * (1.0 - rhokm / rhok);
	}
	else {
	  DBSFC[i + j * NX_BLOCK + k * NY_BLOCK * NX_BLOCK] = 0.0;
	  DBLOC[i + j * NX_BLOCK + (k - 1) * NY_BLOCK * NX_BLOCK] = 0.0;
	}

	//zero if on land
	//why DBSFC isnt zeroed here is a mystery to me -Ben
	if (k >= kmt) {		//-1 removed because FORTRAN array index starts at 1
	  DBLOC[i + j * NX_BLOCK + (k - 1) * NY_BLOCK * NX_BLOCK] = 0.0;
	}

      }
    }
  }

}


/*
 * Compare function that compares two arrays of length N for similarity
 * 
 * This function performs a number of different tests, for example the number of
 * values at an epsilon from 0.0 should be similar in both arrays and may not
 * be greater than 3/4th of the array. Additionally NaN values are treated as
 * errors.
 *
 * The value of eps should be adjusted to something reasonable given the
 * fact that CPU and GPU do not produce exactly the same numerical results. 
 */
int
compare (double *a1, double *a2, int N) {
  int i = 0, res = 0;
  int print = 0;
  int zero_one = 0;
  int zero_two = 0;
  double eps = 0.00000000001;

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
	fprintf (stderr, "Error detected at i=%d, a1= %30.27e a2= %30.27e\n", i, a1[i], a2[i]);
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
