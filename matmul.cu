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
 * matrix multiplication kernel.
 *
 * The kernel is assumed to be tuned to each device by selecting
 * the best performing combination of thread block dimensions 
 * and tiling factors in X and Y. In this implementation tiling
 * in X increases the amount of work per thread block and tiling
 * in Y increases the amount of work per thread. 
 * 
 * @author Ben van Werkhoven <b.vanwerkhoven@esciencecenter.nl>
 * 
 */

#include <stdio.h>
#include <stdlib.h>

#define WIDTH 4096
#define HEIGHT 4096

//Select best kernel configuration for your device

//Tesla K20
//#define BLOCK_X 32
//#define BLOCK_Y 8
//#define TILE_Y 4
//#define TILE_X 8

//GTX 480 
#define BLOCK_X 32
#define BLOCK_Y 8
#define TILE_Y 4
#define TILE_X 8

//GTX Titan
//#define BLOCK_X 64
//#define BLOCK_Y 8
//#define TILE_Y 8
//#define TILE_X 2



#define NSTREAMS (WIDTH/BLOCK_X)

#define ITERATIONS 5

//for naive
#define BLOCK_SIZE 32

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


/* "" */


extern "C" {

  void matmul (float *res, float *mat, float *vec);
  void matmul_explicit (float *res, float *mat, float *vec);
  void matmul_implicit (float *C, float *A, float *B);
  void matmul_streams (float *C, float *A, float *B);
  void matmul_hybrid (float *C, float *A, float *B);

  void matmul_naive (float *C, float *A, float *B);

  void start_timer ();
  void stop_timer (float *);

  int compare (float *a, float *b, int N);

  __global__ void matmul_kernel_shared (float *C, float *A, float *B);
  __global__ void matmul_kernel_opt (float *C, float *A, float *B);

}


int nStreams = -1;
cudaStream_t stream[NSTREAMS];
cudaEvent_t event_htod[NSTREAMS];

float *h_A;
float *h_B;
float *h_C;
float *h_Cref;

float *d_A;
float *d_B;
float *d_C;

int
main () {
  cudaError_t err;

  cudaSetDeviceFlags (cudaDeviceMapHost);
  cudaSetDevice (0);

  cudaDeviceSetCacheConfig (cudaFuncCachePreferShared);
  cudaDeviceSetSharedMemConfig (cudaSharedMemBankSizeFourByte);

  //setup streams
  for (int k = 0; k < NSTREAMS; k++) {
    err = cudaStreamCreate (&stream[k]);
    if (err != cudaSuccess) {
      fprintf (stderr, "Error in cudaStreamCreate: %s\n", cudaGetErrorString (err));
    }
    err = cudaEventCreate (&event_htod[k]);
    if (err != cudaSuccess) {
      fprintf (stderr, "Error in cudaEventCreate htod: %s\n", cudaGetErrorString (err));
    }

  }

  //setup memory
  err = cudaHostAlloc ((void **) &h_A, WIDTH * HEIGHT * sizeof (float), cudaHostAllocMapped);
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaHostAlloc: %s\n", cudaGetErrorString (err));
  }

  err = cudaHostAlloc ((void **) &h_B, WIDTH * HEIGHT * sizeof (float), cudaHostAllocMapped);
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaHostAlloc: %s\n", cudaGetErrorString (err));
  }

  err = cudaHostAlloc ((void **) &h_C, WIDTH * HEIGHT * sizeof (float), cudaHostAllocMapped);
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaHostAlloc: %s\n", cudaGetErrorString (err));
  }

  err = cudaHostAlloc ((void **) &h_Cref, WIDTH * HEIGHT * sizeof (float), cudaHostAllocMapped);
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaHostAlloc: %s\n", cudaGetErrorString (err));
  }

  for (int y = 0; y < HEIGHT; y++) {
    for (int x = 0; x < WIDTH; x++) {
      int r = rand ();
      h_A[y * (WIDTH) + x] = 0.000001 + (r % 999) / 1000.0;
      r = rand ();
      h_B[y * (WIDTH) + x] = 0.000001 + (r % 500) / 5000.0;
    }
  }

  //error checking
  cudaDeviceSynchronize ();
  CUDA_CHECK_ERROR ("After setup");

  //create reference answer for correctness checks
  memset (h_Cref, 0, WIDTH * HEIGHT * sizeof (float));
  memset (h_C, 0, WIDTH * HEIGHT * sizeof (float));
  matmul_naive (h_Cref, h_A, h_B);

  //run four different implementations

  for (int k = 0; k < ITERATIONS; k++) {
    matmul_explicit (h_C, h_A, h_B);
  }
  compare (h_Cref, h_C, WIDTH * HEIGHT);

  for (int k = 0; k < ITERATIONS; k++) {
    matmul_implicit (h_C, h_A, h_B);
  }
  compare (h_Cref, h_C, WIDTH * HEIGHT);

  for (int k = 0; k < ITERATIONS; k++) {
    matmul_streams (h_C, h_A, h_B);
  }
  compare (h_Cref, h_C, WIDTH * HEIGHT);

  for (int k = 0; k < ITERATIONS; k++) {
    matmul_hybrid (h_C, h_A, h_B);
  }
  compare (h_Cref, h_C, WIDTH * HEIGHT);

  return 0;
}


//reference implementation, not called by this program
void
matmul (float *C, float *A, float *B) {

  int x, y, k;
  float sum = 0.0f;

  for (y = 0; y < HEIGHT; y++) {
    for (x = 0; x < WIDTH; x++) {
      sum = 0.0f;
      for (k = 0; k < WIDTH; k++) {
	sum += A[y * WIDTH + k] * B[k * WIDTH + x];
      }
      C[y * WIDTH + x] = sum;
    }
  }

}


/*
 * Naive CUDA kernel for matrix multiplication
 *
 * not called in this program, included for completeness and clarity.
 */
__global__ void
matmul_kernel (float *C, float *A, float *B) {

  int x = blockIdx.x * BLOCK_X + threadIdx.x;
  int y = blockIdx.y * BLOCK_Y + threadIdx.y;
  int k;
  float sum = 0.0f;

  if ((x < WIDTH) && (y < HEIGHT)) {

    for (k = 0; k < WIDTH; k++) {
      sum += A[y * WIDTH + k] * B[k * WIDTH + x];
    }
    C[y * WIDTH + x] = sum;

  }

}


/*
 * Slightly less naive CUDA kernel for matrix multiplication
 *
 * This implementation is used to compare the results of the produced
 * by the optimized and different schemes for overlapping communication
 * and computation. The main reason to use another kernel to compare
 * results is that a naive CPU version would take forever.
 *
 * In this kernel a thread block uses a tile of shared memory to
 * cooperatively load and store the values required for each computation
 * step.
 *
 */
__global__ void
matmul_kernel_shared (float *C, float *A, float *B) {

  __shared__ float sA[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float sB[BLOCK_SIZE][BLOCK_SIZE];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;
  int k, kb;
  float sum = 0.0f;

  if ((x < WIDTH) && (y < HEIGHT)) {

    for (k = 0; k < WIDTH; k += BLOCK_SIZE) {

      __syncthreads ();
      sA[ty][tx] = A[y * WIDTH + k + tx];
      sB[ty][tx] = B[(k + ty) * WIDTH + x];
      __syncthreads ();

      for (kb = 0; kb < BLOCK_SIZE; kb++) {
	sum += sA[ty][kb] * sB[kb][tx];
      }

    }
    C[y * WIDTH + x] = sum;

  }

}


/*
 * Optimized CUDA kernel for matrix multiplication
 *
 * This kernel is optimized and tuned according to the directions given
 * in: "Better performance at lower occupancy" by V. Volkov,
 * GPU Technology Conference, GTC 2010.
 *
 * The thread block dimensions as well as tiling factors are tuned to-
 * wards each GPU used as part of our evaluation.
 *
 */
__global__ void
matmul_kernel_opt (float *C, float *A, float *B) {

  __shared__ float sA[BLOCK_X][BLOCK_X];
  __shared__ float sB[BLOCK_X][BLOCK_X * TILE_X];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x = blockIdx.x * BLOCK_X * TILE_X + threadIdx.x;
  int y = blockIdx.y * BLOCK_Y * TILE_Y + threadIdx.y;
  int k, kb;

#if(TILE_X == 8)
#if(TILE_Y == 4)
  float sum[TILE_X][TILE_Y] = { {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0} };
#endif
#elif(TILE_X == 2)
#if(TILE_Y == 8)
  float sum[TILE_X][TILE_Y] = { {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0} };
#endif
#endif

  for (k = 0; k < WIDTH; k += BLOCK_X) {

    __syncthreads ();
#pragma unroll
    for (int i = 0; i < TILE_Y; i++) {
      sA[ty + BLOCK_Y * i][tx] = A[y * WIDTH + BLOCK_Y * i * WIDTH + k + tx];
#pragma unroll
      for (int j = 0; j < TILE_X; j++) {
	sB[ty + BLOCK_Y * i][tx + j * BLOCK_X] = B[(k + ty + BLOCK_Y * i) * WIDTH + x + j * BLOCK_X];
      }
    }
    __syncthreads ();

    //compute
#pragma unroll
    for (kb = 0; kb < BLOCK_X; kb++) {

#pragma unroll
      for (int i = 0; i < TILE_Y; i++) {
#pragma unroll
	for (int j = 0; j < TILE_X; j++) {
	  sum[j][i] += sA[ty + BLOCK_Y * i][kb] * sB[kb][tx + j * BLOCK_X];
	}
      }

    }

  }

  //store result
#pragma unroll
  for (int i = 0; i < TILE_Y; i++) {
#pragma unroll
    for (int j = 0; j < TILE_X; j++) {
      C[y * WIDTH + x + BLOCK_Y * i * WIDTH + j * BLOCK_X] = sum[j][i];
    }
  }

}




/*
 * Host code that invokes the matrix multiplication kernel
 *
 * The explicit implementation uses explicit memory copy
 * statements to move all data to the GPU, executes the
 * GPU kernel, and uses memory copies to copy the output
 * data back to host memory. This implementation achieves
 * no overlap between transfers and/or computation.
 *
 */
void
matmul_explicit (float *C, float *A, float *B) {
  cudaError_t err;

  err = cudaMalloc ((void **) &d_A, WIDTH * HEIGHT * sizeof (float));
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMalloc d_A: %s\n", cudaGetErrorString (err));
  }

  err = cudaMalloc ((void **) &d_B, WIDTH * HEIGHT * sizeof (float));
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMalloc d_B: %s\n", cudaGetErrorString (err));
  }

  err = cudaMalloc ((void **) &d_C, WIDTH * HEIGHT * sizeof (float));
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMalloc d_C: %s\n", cudaGetErrorString (err));
  }

  err = cudaMemset (d_C, 0, WIDTH * HEIGHT * sizeof (float));
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMemset d_C: %s\n", cudaGetErrorString (err));
  }

  cudaDeviceSynchronize ();
  CUDA_CHECK_ERROR ("After memory setup");

  dim3 threads (BLOCK_X, BLOCK_Y);
  dim3 grid ((int) ceilf ((float) WIDTH / (float) (BLOCK_X * TILE_X)), (int) ceilf ((float) HEIGHT / (float) (BLOCK_Y * TILE_Y)));

  float time;
  cudaDeviceSynchronize ();
  start_timer ();

  err = cudaMemcpyAsync (d_B, B, WIDTH * HEIGHT * sizeof (float), cudaMemcpyHostToDevice, stream[1]);
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMemcpy host to device B: %s\n", cudaGetErrorString (err));
  }

  err = cudaMemcpyAsync (d_A, A, WIDTH * HEIGHT * sizeof (float), cudaMemcpyHostToDevice, stream[1]);
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMemcpy host to device A: %s\n", cudaGetErrorString (err));
  }

  matmul_kernel_opt <<< grid, threads, 0, stream[1] >>> (d_C, d_A, d_B);

  err = cudaMemcpyAsync (C, d_C, WIDTH * HEIGHT * sizeof (float), cudaMemcpyDeviceToHost, stream[1]);
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMemcpy device to host C: %s\n", cudaGetErrorString (err));
  }

  cudaDeviceSynchronize ();
  stop_timer (&time);

  cudaDeviceSynchronize ();
  CUDA_CHECK_ERROR ("After Explicit4");

  printf ("EXPLICIT: %.6f ms\n", time);

  cudaDeviceSynchronize ();
  start_timer ();

  matmul_kernel_opt <<< grid, threads, 0, stream[1] >>> (d_C, d_A, d_B);

  cudaDeviceSynchronize ();
  stop_timer (&time);
  float flops = 2.0 * (WIDTH * HEIGHT) * (WIDTH);
  float giga = 1000000000.0;
  printf ("EXPLICIT kernel: %.6f ms\t %.3f GFLOP/s \n", time, (flops / giga) / (time / 1000.0));

  cudaDeviceSynchronize ();
  CUDA_CHECK_ERROR ("After kernel explicit");

  cudaFree (d_C);
  cudaFree (d_B);
  cudaFree (d_A);

}




/*
 * Host code that invokes the matrix multiplication kernel
 *
 * The implicit implementation uses device-mapped host memory rather
 * than explicit memory copy statements. A different kernel is used
 * to ensure strictly coalesced access to system memory.
 *
 */
void
matmul_implicit (float *C, float *A, float *B) {

  cudaDeviceSynchronize ();
  CUDA_CHECK_ERROR ("before execution");

  dim3 threads (BLOCK_X, BLOCK_Y);
  dim3 grid ((int) ceilf ((float) WIDTH / (float) (BLOCK_X * TILE_X)), (int) ceilf ((float) HEIGHT / (float) (BLOCK_Y * TILE_Y)));

  float time;
  cudaDeviceSynchronize ();
  start_timer ();

  matmul_kernel_opt <<< grid, threads, 0, stream[1] >>> (C, A, B);

  cudaDeviceSynchronize ();
  stop_timer (&time);
  printf ("IMPLICIT: %.6f ms\n", time);

  cudaDeviceSynchronize ();
  CUDA_CHECK_ERROR ("After kernel");

}




/*
 * Host code that invokes the matrix multiplication kernel
 *
 * The streams implementation uses CUDA streams combined
 * with explicit memory copy statements. This way transfers
 * in one stream may overlap with computation and transfers
 * in other streams.
 *
 */
void
matmul_streams (float *C, float *A, float *B) {
  cudaError_t err;
  int k;

  err = cudaMalloc ((void **) &d_A, WIDTH * HEIGHT * sizeof (float));
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMalloc d_A: %s\n", cudaGetErrorString (err));
  }

  err = cudaMalloc ((void **) &d_B, WIDTH * HEIGHT * sizeof (float));
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMalloc d_B: %s\n", cudaGetErrorString (err));
  }

  err = cudaMalloc ((void **) &d_C, WIDTH * HEIGHT * sizeof (float));
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMalloc d_C: %s\n", cudaGetErrorString (err));
  }

  err = cudaMemset (d_C, 0, WIDTH * HEIGHT * sizeof (float));
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMemset d_C: %s\n", cudaGetErrorString (err));
  }

  cudaDeviceSynchronize ();
  CUDA_CHECK_ERROR ("After memory setup");

  dim3 threads (BLOCK_X, BLOCK_Y);
//  dim3 grid( (int)ceilf((float)WIDTH / (float)(BLOCK_X)) , (int)ceilf((float)HEIGHT / (float)(BLOCK_Y)));

  dim3 grid ((int) ceilf ((float) WIDTH / (float) (BLOCK_X * TILE_X)), 1);
//  dim3 grid( (int)ceilf((float)WIDTH / (float)(BLOCK_X)) , 1);

  int lps = WIDTH * BLOCK_Y * TILE_Y;

  float time;
  cudaDeviceSynchronize ();
  start_timer ();

  err = cudaMemcpyAsync (d_B, B, WIDTH * HEIGHT * sizeof (float), cudaMemcpyHostToDevice, stream[1]);
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMemcpy host to device B: %s\n", cudaGetErrorString (err));
  }

  err = cudaEventRecord (event_htod[1], stream[1]);
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaEventRecord htod: %s\n", cudaGetErrorString (err));
  }

  for (k = 0; k < NSTREAMS; k++) {
    err = cudaMemcpyAsync (d_A + k * lps, A + k * lps, lps * sizeof (float), cudaMemcpyHostToDevice, stream[k]);
    if (err != cudaSuccess) {
      fprintf (stderr, "Error in cudaMemcpy host to device scratch: %s\n", cudaGetErrorString (err));
    }

  }

  for (k = 0; k < NSTREAMS; k++) {
    //wait for memcpy in stream 1 to be complete
    err = cudaStreamWaitEvent (stream[k], event_htod[1], 0);
    if (err != cudaSuccess) {
      fprintf (stderr, "Error in cudaStreamWaitEvent htod 1: %s\n", cudaGetErrorString (err));
    }

    matmul_kernel_opt <<< grid, threads, 0, stream[k] >>> (d_C + k * lps, d_A + k * lps, d_B);
  }

  for (k = 0; k < NSTREAMS; k++) {
    err = cudaMemcpyAsync (C + k * lps, d_C + k * lps, lps * sizeof (float), cudaMemcpyDeviceToHost, stream[k]);
    if (err != cudaSuccess) {
      fprintf (stderr, "Error in cudaMemcpy device to host C: %s\n", cudaGetErrorString (err));
    }
  }

  cudaDeviceSynchronize ();
  stop_timer (&time);
  printf ("STREAMS: %.6f ms\n", time);

  cudaDeviceSynchronize ();
  CUDA_CHECK_ERROR ("After kernel");

  cudaFree (d_C);
  cudaFree (d_B);
  cudaFree (d_A);

}


/*
 * Host code that invokes the matrix multiplication kernel
 *
 * The Hybrid implementation uses CUDA streams combined
 * with explicit memory copy statements for the input data
 * and uses device-mapped host memory to copy the output data
 * back to host memory. 
 *
 */
void
matmul_hybrid (float *C, float *A, float *B) {
  cudaError_t err;
  int k;

  err = cudaMalloc ((void **) &d_A, WIDTH * HEIGHT * sizeof (float));
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMalloc d_A: %s\n", cudaGetErrorString (err));
  }

  err = cudaMalloc ((void **) &d_B, WIDTH * HEIGHT * sizeof (float));
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMalloc d_B: %s\n", cudaGetErrorString (err));
  }

  cudaDeviceSynchronize ();
  CUDA_CHECK_ERROR ("After memory setup");

  dim3 threads (BLOCK_X, BLOCK_Y);
  dim3 grid ((int) ceilf ((float) WIDTH / (float) (BLOCK_X * TILE_X)), 1);

  int lps = WIDTH * BLOCK_Y * TILE_Y;

  float time;
  cudaDeviceSynchronize ();
  start_timer ();

  err = cudaMemcpyAsync (d_B, B, WIDTH * HEIGHT * sizeof (float), cudaMemcpyHostToDevice, stream[1]);
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMemcpy host to device B: %s\n", cudaGetErrorString (err));
  }

  err = cudaEventRecord (event_htod[1], stream[1]);
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaEventRecord htod: %s\n", cudaGetErrorString (err));
  }

  for (k = 0; k < NSTREAMS; k++) {
    err = cudaMemcpyAsync (d_A + k * lps, A + k * lps, lps * sizeof (float), cudaMemcpyHostToDevice, stream[k]);
    if (err != cudaSuccess) {
      fprintf (stderr, "Error in cudaMemcpy host to device scratch: %s\n", cudaGetErrorString (err));
    }

  }

  for (k = 0; k < NSTREAMS; k++) {
    //wait for memcpy in stream 1 to be complete
    err = cudaStreamWaitEvent (stream[k], event_htod[1], 0);
    if (err != cudaSuccess) {
      fprintf (stderr, "Error in cudaStreamWaitEvent htod 1: %s\n", cudaGetErrorString (err));
    }

    matmul_kernel_opt <<< grid, threads, 0, stream[k] >>> (C + k * lps, d_A + k * lps, d_B);
  }

  cudaDeviceSynchronize ();
  stop_timer (&time);
  printf ("HYBRID: %.6f ms\n", time);

  cudaDeviceSynchronize ();
  CUDA_CHECK_ERROR ("After kernel");

  cudaFree (d_B);
  cudaFree (d_A);

}



/*
 * Host code that invokes the naive matrix multiplication kernel
 *
 * The naive kernel is used to verify results from the other
 * implementations. It uses explicit memory copy
 * statements to move all data to the GPU, executes the
 * naive kernel, and uses memory copies to copy the output
 * data back to host memory. This implementation achieves
 * no overlap between transfers and/or computation.
 *
 */
void
matmul_naive (float *C, float *A, float *B) {
  cudaError_t err;

  err = cudaMalloc ((void **) &d_A, WIDTH * HEIGHT * sizeof (float));
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMalloc d_A: %s\n", cudaGetErrorString (err));
  }

  err = cudaMalloc ((void **) &d_B, WIDTH * HEIGHT * sizeof (float));
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMalloc d_B: %s\n", cudaGetErrorString (err));
  }

  err = cudaMalloc ((void **) &d_C, WIDTH * HEIGHT * sizeof (float));
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMalloc d_C: %s\n", cudaGetErrorString (err));
  }

  err = cudaMemset (d_C, 0, WIDTH * HEIGHT * sizeof (float));
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMemset d_C: %s\n", cudaGetErrorString (err));
  }

  cudaDeviceSynchronize ();
  CUDA_CHECK_ERROR ("After memory setup");

  //NAIVE - DO NOT CHANGE -
  dim3 threads (BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid ((int) ceilf ((float) WIDTH / (float) (BLOCK_SIZE)), (int) ceilf ((float) HEIGHT / (float) (BLOCK_SIZE)));

  float time;
  cudaDeviceSynchronize ();
  start_timer ();

  err = cudaMemcpyAsync (d_B, B, WIDTH * HEIGHT * sizeof (float), cudaMemcpyHostToDevice, stream[1]);
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMemcpy host to device B: %s\n", cudaGetErrorString (err));
  }

  err = cudaMemcpyAsync (d_A, A, WIDTH * HEIGHT * sizeof (float), cudaMemcpyHostToDevice, stream[1]);
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMemcpy host to device A: %s\n", cudaGetErrorString (err));
  }

  //NAIVE - DO NOT CHANGE -
  matmul_kernel_shared <<< grid, threads, 0, stream[1] >>> (d_C, d_A, d_B);

  err = cudaMemcpyAsync (C, d_C, WIDTH * HEIGHT * sizeof (float), cudaMemcpyDeviceToHost, stream[1]);
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMemcpy device to host C: %s\n", cudaGetErrorString (err));
  }

  cudaDeviceSynchronize ();
  stop_timer (&time);

  cudaDeviceSynchronize ();
  CUDA_CHECK_ERROR ("After Naive");

  printf ("NAIVE: %.6f ms\n", time);

  cudaDeviceSynchronize ();
  start_timer ();

  //NAIVE - DO NOT CHANGE -
  matmul_kernel_shared <<< grid, threads, 0, stream[1] >>> (d_C, d_A, d_B);

  cudaDeviceSynchronize ();
  stop_timer (&time);
  float flops = 2.0 * (WIDTH * HEIGHT) * (WIDTH);
  float giga = 1000000000.0;
  printf ("NAIVE kernel: %.6f ms\t %.3f GFLOP/s \n", time, (flops / giga) / (time / 1000.0));

  cudaDeviceSynchronize ();
  CUDA_CHECK_ERROR ("After kernel explicit");

  cudaFree (d_C);
  cudaFree (d_B);
  cudaFree (d_A);

}




int
compare (float *a1, float *a2, int N) {
  int i = 0, res = 0;
  int print = 0;
  int zero_one = 0;
  int zero_two = 0;
  float eps = 0.000001;

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
	fprintf (stderr, "Error detected at i=%d,\t a1= %10.7e \t a2= \t %10.7e\n", i, a1[i], a2[i]);
      }
    }

    float diff = a1[i] - a2[i];
    if (diff > eps || diff < -eps) {
      res++;
      if (print < 10) {
	print++;
	fprintf (stderr, "Error detected at i=%d,\t a1= \t %10.7e \t a2= \t %10.7e\n", i, a1[i], a2[i]);
      }
    }

  }

  if (zero_one > (N / 4)) {
    fprintf (stderr, "Error: array1 contains %d zeros\n", zero_one);
  }
  if (zero_two > (N / 4)) {
    fprintf (stderr, "Error: array2 contains %d zeros\n", zero_two);
  }

  if (zero_one != zero_two) {
    fprintf (stderr, "Error: number of zeros in arrays dont correspond zero1=%d, zero2=%d\n", zero_one, zero_two);
  }

  if (res > 0) {
    fprintf (stdout, "Number of errors in GPU result: %d\n", res);
  }

  return res;
}
