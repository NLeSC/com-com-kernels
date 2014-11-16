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
 * 2D Convolution kernel.
 *
 * For more information on how the kernel in this program was generated
 * please see:
 *  Optimizing convolution operations on GPUs using adaptive tiling
 *  B. van Werkhoven, J. Maassen, F.J. Seinstra, H.E Bal
 *  Future Generation Computer Systems, Volume 30, 2014
 * 
 * @author Ben van Werkhoven <b.vanwerkhoven@esciencecenter.nl>
 * 
 */
#include <stdio.h>
#include <stdlib.h>

#define WIDTH 4096
#define HEIGHT 4096

#define FW 17
#define FH 17

#define BLOCK_X 16
#define BLOCK_Y 16

#define TILE_X 4

#define NSTREAMS 64

#define ITERATIONS 5

//#define USE_READ_ONLY_CACHE 
#ifdef USE_READ_ONLY_CACHE
#define LDG(x) __ldg(x)
#else
#define LDG(x) *(x)
#endif

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

  void convolvution2d (float *image, float *scratch, float *filter, int scratchWidth, int scratchHeight, float filterWeight);
  void convolvution2d_explicit (float *image, float *scratch, int scratchWidth, int scratchHeight, float filterWeight);
  void convolvution2d_implicit (float *image, float *scratch, int scratchWidth, int scratchHeight, float filterWeight);
  void convolvution2d_stream (float *image, float *scratch, int scratchWidth, int scratchHeight, float filterWeight);
  void convolvution2d_hybrid (float *image, float *scratch, int scratchWidth, int scratchHeight, float filterWeight);

  void start_timer ();
  void stop_timer (float *);

  int compare (float *a, float *b, int N);

  __global__ void convolvution2d_kernel_naive (float *image, float *scratch, int scratchWidth, int scratchHeight, float filterWeight);
  __global__ void convolvution2d_kernel (float *__restrict__ iPtr, const float *__restrict__ sPtr, int totalWidth, int scratchHeight, float divisor);

}

int nStreams = -1;
cudaStream_t stream[NSTREAMS];
cudaEvent_t event_htod[NSTREAMS];

float *h_filter;
float *h_image;
float *h_imageref;
float *h_scratch;
float filterWeight = 0.0;

__constant__ float d_filter[FW * FH];
float *d_image;
float *d_scratch;

int
main () {
  cudaError_t err;

  cudaSetDeviceFlags (cudaDeviceMapHost);
  cudaSetDevice (0);
  cudaDeviceSynchronize ();

  printf ("Starting Conv..\n");

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
  h_imageref = (float *) malloc (WIDTH * HEIGHT * sizeof (float));
  if (!h_imageref)
    fprintf (stderr, "Error in malloc: h_imageref\n");

  err = cudaHostAlloc ((void **) &h_filter, FW * FH * sizeof (float), cudaHostAllocMapped);
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaHostAlloc: %s\n", cudaGetErrorString (err));
  }

  err = cudaHostAlloc ((void **) &h_image, WIDTH * HEIGHT * sizeof (float), cudaHostAllocMapped);
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaHostAlloc: %s\n", cudaGetErrorString (err));
  }
  memset (h_image, 0, WIDTH * HEIGHT * sizeof (float));

  err = cudaHostAlloc ((void **) &h_scratch, (WIDTH + FW - 1) * (HEIGHT + FH - 1) * sizeof (float), cudaHostAllocMapped);
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaHostAlloc: %s\n", cudaGetErrorString (err));
  }

  for (int y = 0; y < HEIGHT + FH - 1; y++) {
    for (int x = 0; x < WIDTH + FW - 1; x++) {
      int r = rand ();
      h_scratch[y * (WIDTH + FW - 1) + x] = 1.0 + r % 254;
    }
  }

  for (int y = 0; y < FH; y++) {
    for (int x = 0; x < FW; x++) {
      int r = rand ();
      float w = 0.001 + (r % 999) / 1000.0;
      h_filter[y * FW + x] = w;
      filterWeight += w;
    }
  }

  err = cudaMemcpyToSymbolAsync (d_filter, h_filter, FW * FH * sizeof (float), 0, cudaMemcpyHostToDevice, 0);
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMemcpyToSymbolAsync: %s\n", cudaGetErrorString (err));
  }

  //error checking
  cudaDeviceSynchronize ();
  CUDA_CHECK_ERROR ("After setup");

  printf ("Starting CPU routine..\n");
  fflush (stdout);

  float time;
  start_timer ();

  convolvution2d (h_imageref, h_scratch, h_filter, WIDTH + FW - 1, HEIGHT + FH - 1, filterWeight);

  stop_timer (&time);
  printf ("2D Convolution on CPU took: %.6f ms\n", time);

  for (int i = 0; i < ITERATIONS; i++) {
    convolvution2d_explicit (h_image, h_scratch, WIDTH + FW - 1, HEIGHT + FH - 1, filterWeight);
  }
  compare (h_imageref, h_image, WIDTH * HEIGHT);

  for (int i = 0; i < ITERATIONS; i++) {
    convolvution2d_implicit (h_image, h_scratch, WIDTH + FW - 1, HEIGHT + FH - 1, filterWeight);
  }
  compare (h_imageref, h_image, WIDTH * HEIGHT);

  for (int i = 0; i < ITERATIONS; i++) {
    convolvution2d_stream (h_image, h_scratch, WIDTH + FW - 1, HEIGHT + FH - 1, filterWeight);
  }
  compare (h_imageref, h_image, WIDTH * HEIGHT);

  for (int i = 0; i < ITERATIONS; i++) {
    convolvution2d_hybrid (h_image, h_scratch, WIDTH + FW - 1, HEIGHT + FH - 1, filterWeight);
  }
  compare (h_imageref, h_image, WIDTH * HEIGHT);

  return 0;
}



void
convolvution2d (float *image, float *scratch, float *filter, int scratchWidth, int scratchHeight, float filterWeight) {

  int x, y;
  int i, j;
  float sum = 0.0;

  for (y = 0; y < HEIGHT; y++) {
    for (x = 0; x < WIDTH; x++) {
      sum = 0.0;

      for (j = 0; j < FH; j++) {
	for (i = 0; i < FW; i++) {
	  sum += scratch[(y + j) * scratchWidth + (x + i)] * filter[j * FH + i];
	}
      }

      image[y * WIDTH + x] = sum / filterWeight;

    }
  }

}

__global__ void
convolvution2d_kernel_naive (float *image, float *scratch, int scratchWidth, int scratchHeight, float filterWeight) {

  int x = blockIdx.x * BLOCK_X + threadIdx.x;
  int y = blockIdx.y * BLOCK_Y + threadIdx.y;
  int i, j;
  float sum = 0.0;

  if (y * x < HEIGHT * WIDTH) {

    for (j = 0; j < FH; j++) {
      for (i = 0; i < FW; i++) {
	sum += scratch[(y + j) * scratchWidth + (x + i)] * d_filter[j * FH + i];
      }
    }

    image[y * WIDTH + x] = sum / filterWeight;

  }
}

#define SHWIDTH (4*BLOCK_X+FW-1)
#define SHMEMSIZE (SHWIDTH*(BLOCK_Y+FH-1))

__shared__ float shared_scratch[SHMEMSIZE];

/*
 * The following 2D Convolution kernel is adapted from a kernel
 * specifically generated for filter size 17x17 and thread block
 * size 16x16 using 1x4 tiling.
 *
 * For more information on how this kernel was generated see:
 *  Optimizing convolution operations on GPUs using adaptive tiling
 *  B. van Werkhoven, J. Maassen, F.J. Seinstra, H.E Bal
 *  Future Generation Computer Systems, Volume 30, 2014
 */
__global__ void
convolvution2d_kernel (float *__restrict__ iPtr, const float *__restrict__ sPtr, int totalWidth, int scratchHeight, float divisor) {

  float sum0 = 0;
  float sum1 = 0;
  float sum2 = 0;
  float sum3 = 0;

  int sindex = 0;
  int ty = threadIdx.y;
  int tx = threadIdx.x;

  //set scratch to point to start of scratch for this block
  sPtr += (ty + blockIdx.y * BLOCK_Y) * totalWidth + 4 * blockIdx.x * BLOCK_X + tx;
  iPtr += (ty + blockIdx.y * BLOCK_Y) * WIDTH + 4 * blockIdx.x * BLOCK_X + tx;

  //coalsced global memory loads
  //since there are more elements than threads there is some branching here
  sindex = ty * SHWIDTH + tx;

  shared_scratch[sindex] = LDG (sPtr);
  shared_scratch[sindex + 1 * BLOCK_X] = LDG (sPtr + 1 * BLOCK_X);
  shared_scratch[sindex + 2 * BLOCK_X] = LDG (sPtr + 2 * BLOCK_X);
  shared_scratch[sindex + 3 * BLOCK_X] = LDG (sPtr + 3 * BLOCK_X);
  shared_scratch[sindex + 4 * BLOCK_X] = LDG (sPtr + 4 * BLOCK_X);

  sindex += BLOCK_Y * SHWIDTH;
  sPtr += BLOCK_Y * totalWidth;
  shared_scratch[sindex] = LDG (sPtr);
  shared_scratch[sindex + 1 * BLOCK_X] = LDG (sPtr + 1 * BLOCK_X);
  shared_scratch[sindex + 2 * BLOCK_X] = LDG (sPtr + 2 * BLOCK_X);
  shared_scratch[sindex + 3 * BLOCK_X] = LDG (sPtr + 3 * BLOCK_X);
  shared_scratch[sindex + 4 * BLOCK_X] = LDG (sPtr + 4 * BLOCK_X);

//    int jEnd = 16*1 + kerHeight-1;
//    int iEnd = 16*1 + kerWidth-1;

/*
    for (j=ty; j<jEnd; j+= 16) {
	for (i=tx; i<iEnd; i+= 64) {
	    shared_scratch[j*SHWIDTH + i] = sPtr[j*totalWidth + i].x;
	}
    }
*/

  __syncthreads ();
  sindex = ty * SHWIDTH + tx;

  //kernel computation
  int kindex = 0;
  int i = 0;
  int j = 0;
#pragma unroll
  for (j = 0; j < FH; j++) {
#pragma unroll
    for (i = 0; i < FW; i++) {
      sum0 += shared_scratch[sindex] * d_filter[kindex];
      sum1 += shared_scratch[sindex + 1 * BLOCK_X] * d_filter[kindex];
      sum2 += shared_scratch[sindex + 2 * BLOCK_X] * d_filter[kindex];
      sum3 += shared_scratch[sindex + 3 * BLOCK_X] * d_filter[kindex];
      sindex++;
      kindex++;
    }
    sindex = sindex - FW + SHWIDTH;
  }

/*    int jEnd = ty + kerHeight;
    int iEnd = tx + kerWidth;
    int i,j,kindex=0;

    for (j=ty; j<jEnd; j++) {
        for (i=tx; i<iEnd; i++) {
            sum0 += shared_scratch[j*SHWIDTH+i] * filter[kindex].x;
            sum1 += shared_scratch[j*SHWIDTH+i+64] * filter[kindex].x;
            kindex++;
        }
    }
*/

  //global memory store
  *iPtr = sum0 / divisor;
  iPtr += BLOCK_X;
  *iPtr = sum1 / divisor;
  iPtr += BLOCK_X;
  *iPtr = sum2 / divisor;
  iPtr += BLOCK_X;
  *iPtr = sum3 / divisor;

}





/*
 * Host code that invokes the 2D Convolution kernel
 *
 * The explicit implementation uses explicit memory copy
 * statements to move all data to the GPU, executes the
 * GPU kernel, and uses memory copies to copy the output
 * data back to host memory. This implementation achieves
 * no overlap between transfers and/or computation.
 *
 */
void
convolvution2d_explicit (float *image, float *scratch, int scratchWidth, int scratchHeight, float filterWeight) {
  cudaError_t err;

  err = cudaMalloc ((void **) &d_image, WIDTH * HEIGHT * sizeof (float));
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMalloc d_image: %s\n", cudaGetErrorString (err));
  }

  err = cudaMemset (d_image, 0, WIDTH * HEIGHT * sizeof (float));
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMemset d_image: %s\n", cudaGetErrorString (err));
  }

  err = cudaMalloc ((void **) &d_scratch, scratchWidth * scratchHeight * sizeof (float));
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMalloc d_scratch: %s\n", cudaGetErrorString (err));
  }


  cudaDeviceSynchronize ();
  CUDA_CHECK_ERROR ("After memory setup");

  dim3 threads (BLOCK_X, BLOCK_Y);
  // for naive kernel use this grid
  //  dim3 grid( (int)ceilf((float)WIDTH / (float)(BLOCK_X)) , (int)ceilf((float)HEIGHT / (float)(BLOCK_Y))); 
  // for tiled kernel use this grid
  dim3 grid ((int) ceilf ((float) WIDTH / (float) (TILE_X * BLOCK_X)), (int) ceilf ((float) HEIGHT / (float) (BLOCK_Y)));

  float time;
  cudaDeviceSynchronize ();
  start_timer ();

  err = cudaMemcpyAsync (d_scratch, scratch, scratchWidth * scratchHeight * sizeof (float), cudaMemcpyHostToDevice, stream[1]);
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMemcpy host to device scratch: %s\n", cudaGetErrorString (err));
  }

  convolvution2d_kernel <<< grid, threads, 0, stream[1] >>> (d_image, d_scratch, scratchWidth, scratchHeight, filterWeight);

  err = cudaMemcpyAsync (image, d_image, WIDTH * HEIGHT * sizeof (float), cudaMemcpyDeviceToHost, stream[1]);
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMemcpy device to host image: %s\n", cudaGetErrorString (err));
  }

  cudaDeviceSynchronize ();
  stop_timer (&time);
  printf ("EXPLICIT: %.6f ms\n", time);

  cudaDeviceSynchronize ();
  CUDA_CHECK_ERROR ("After explicit");


  cudaDeviceSynchronize ();
  start_timer ();

  convolvution2d_kernel <<< grid, threads, 0, stream[1] >>> (d_image, d_scratch, scratchWidth, scratchHeight, filterWeight);


  cudaDeviceSynchronize ();
  stop_timer (&time);
  float flops = 2.0 * WIDTH * HEIGHT * FH * FW + WIDTH + HEIGHT;
  float giga = 1000000000.0;
  printf ("EXPLICIT kernel: %.6f ms\t %.3f GFLOP/s \n", time, (flops / giga) / (time / 1000.0));

  cudaDeviceSynchronize ();
  CUDA_CHECK_ERROR ("After kernel");

  cudaFree (d_image);
  cudaFree (d_scratch);

}


/*
 * Host code that invokes the 2D Convolution kernel
 *
 * The implicit implementation uses device-mapped host memory rather
 * than explicit memory copy statements. A different kernel is used
 * to ensure strictly coalesced access to system memory.
 *
 */
void
convolvution2d_implicit (float *image, float *scratch, int scratchWidth, int scratchHeight, float filterWeight) {

  dim3 threads (BLOCK_X, BLOCK_Y);
  dim3 grid ((int) ceilf ((float) WIDTH / (float) (TILE_X * BLOCK_X)), (int) ceilf ((float) HEIGHT / (float) (BLOCK_Y)));

  float time;
  cudaDeviceSynchronize ();
  start_timer ();

  convolvution2d_kernel <<< grid, threads, 0, stream[1] >>> (image, scratch, scratchWidth, scratchHeight, filterWeight);

  cudaDeviceSynchronize ();
  stop_timer (&time);
  printf ("IMPLICIT: %.6f ms\n", time);

  cudaDeviceSynchronize ();
  CUDA_CHECK_ERROR ("After kernel");

}




/*
 * Host code that invokes the 2D Convolution kernel
 *
 * The streams implementation uses CUDA streams combined
 * with explicit memory copy statements. This way transfers
 * in one stream may overlap with computation and transfers
 * in other streams.
 *
 */
void
convolvution2d_stream (float *image, float *scratch, int scratchWidth, int scratchHeight, float filterWeight) {
  cudaError_t err;
  int k = 0;

  err = cudaMalloc ((void **) &d_image, WIDTH * HEIGHT * sizeof (float));
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMalloc d_image: %s\n", cudaGetErrorString (err));
  }

  err = cudaMemset (d_image, 0, WIDTH * HEIGHT * sizeof (float));
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMemset d_image: %s\n", cudaGetErrorString (err));
  }

  err = cudaMalloc ((void **) &d_scratch, scratchWidth * scratchHeight * sizeof (float));
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMalloc d_scratch: %s\n", cudaGetErrorString (err));
  }

  cudaDeviceSynchronize ();
  CUDA_CHECK_ERROR ("After memory setup");

  dim3 threads (BLOCK_X, BLOCK_Y);
  // for naive kernel use this grid
  //  dim3 grid( (int)ceilf((float)WIDTH / (float)(BLOCK_X)) , (int)ceilf((float)HEIGHT / (float)(BLOCK_Y))); 
  // for tiled kernel use this grid
  dim3 grid ((int) ceilf ((float) WIDTH / (float) (TILE_X * BLOCK_X)), (HEIGHT / NSTREAMS) / BLOCK_Y);

  int lps = scratchWidth * (HEIGHT / NSTREAMS);
  int border = scratchWidth * (FH - 1);

  float time;
  cudaDeviceSynchronize ();
  start_timer ();

  for (k = 0; k < NSTREAMS; k++) {
    if (k == 0) {
      err = cudaMemcpyAsync (d_scratch, scratch, (border + lps) * sizeof (float), cudaMemcpyHostToDevice, stream[k]);
    }
    else {
      err = cudaStreamWaitEvent (stream[k], event_htod[k - 1], 0);
      if (err != cudaSuccess) {
	fprintf (stderr, "Error in cudaStreamWaitEvent htod k-1: %s\n", cudaGetErrorString (err));
      }

      err = cudaMemcpyAsync (d_scratch + border + k * lps, scratch + border + k * lps, lps * sizeof (float), cudaMemcpyHostToDevice, stream[k]);
    }
    if (err != cudaSuccess) {
      fprintf (stderr, "Error in cudaMemcpy host to device scratch: %s\n", cudaGetErrorString (err));
    }

    err = cudaEventRecord (event_htod[k], stream[k]);
    if (err != cudaSuccess) {
      fprintf (stderr, "Error in cudaEventRecord htod: %s\n", cudaGetErrorString (err));
    }

    // depending on whether you have implicit synchronization or not it may be beneficial to either split or
    // merge these loops
    //  }
    //  for (k=0; k<NSTREAMS; k++) {

    convolvution2d_kernel <<< grid, threads, 0, stream[k] >>> (d_image + k * (WIDTH * (HEIGHT / NSTREAMS)), d_scratch + k * lps, scratchWidth, scratchHeight,
							       filterWeight);
  }

  for (k = 0; k < NSTREAMS; k++) {

    err =
      cudaMemcpyAsync (image + k * (WIDTH * (HEIGHT / NSTREAMS)), d_image + k * (WIDTH * (HEIGHT / NSTREAMS)), (WIDTH * (HEIGHT / NSTREAMS)) * sizeof (float),
		       cudaMemcpyDeviceToHost, stream[k]);
    if (err != cudaSuccess) {
      fprintf (stderr, "Error in cudaMemcpy device to host image: %s\n", cudaGetErrorString (err));
    }

  }

  cudaDeviceSynchronize ();
  stop_timer (&time);
  printf ("STREAMS: %.6f ms\n", time);


  cudaDeviceSynchronize ();
  CUDA_CHECK_ERROR ("After kernel");

  cudaFree (d_image);
  cudaFree (d_scratch);

}



/*
 * Host code that invokes the 2D Convolution kernel
 *
 * The Hybrid implementation uses CUDA streams combined
 * with explicit memory copy statements for the input data
 * and uses device-mapped host memory to copy the output data
 * back to host memory. 
 *
 */
void
convolvution2d_hybrid (float *image, float *scratch, int scratchWidth, int scratchHeight, float filterWeight) {
  cudaError_t err;
  int k = 0;

  err = cudaMalloc ((void **) &d_scratch, scratchWidth * scratchHeight * sizeof (float));
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMalloc d_scratch: %s\n", cudaGetErrorString (err));
  }

  cudaDeviceSynchronize ();
  CUDA_CHECK_ERROR ("After memory setup");

  dim3 threads (BLOCK_X, BLOCK_Y);
  // for naive kernel use this grid
  //  dim3 grid( (int)ceilf((float)WIDTH / (float)(BLOCK_X)) , (int)ceilf((float)HEIGHT / (float)(BLOCK_Y))); 
  // for tiled kernel use this grid
  dim3 grid ((int) ceilf ((float) WIDTH / (float) (TILE_X * BLOCK_X)), (HEIGHT / NSTREAMS) / BLOCK_Y);

  int lps = scratchWidth * (HEIGHT / NSTREAMS);
  int border = scratchWidth * (FH - 1);

  float time;
  cudaDeviceSynchronize ();
  start_timer ();

  for (k = 0; k < NSTREAMS; k++) {
    if (k == 0) {
      err = cudaMemcpyAsync (d_scratch, scratch, (border + lps) * sizeof (float), cudaMemcpyHostToDevice, stream[k]);
    }
    else {

      err = cudaStreamWaitEvent (stream[k], event_htod[k - 1], 0);
      if (err != cudaSuccess) {
	fprintf (stderr, "Error in cudaStreamWaitEvent htod k-1: %s\n", cudaGetErrorString (err));
      }

      err = cudaMemcpyAsync (d_scratch + border + k * lps, scratch + border + k * lps, lps * sizeof (float), cudaMemcpyHostToDevice, stream[k]);
    }
    if (err != cudaSuccess) {
      fprintf (stderr, "Error in cudaMemcpy host to device scratch: %s\n", cudaGetErrorString (err));
    }

    err = cudaEventRecord (event_htod[k], stream[k]);
    if (err != cudaSuccess) {
      fprintf (stderr, "Error in cudaEventRecord htod: %s\n", cudaGetErrorString (err));
    }
  }

  for (k = 0; k < NSTREAMS; k++) {

    convolvution2d_kernel <<< grid, threads, 0, stream[k] >>> (image + k * (WIDTH * (HEIGHT / NSTREAMS)), d_scratch + k * lps, scratchWidth, scratchHeight,
							       filterWeight);
  }

  //mapped memory is used to transfer result to main memory in this implementation

  cudaDeviceSynchronize ();
  stop_timer (&time);
  printf ("HYBRID: %.6f ms\n", time);

  cudaDeviceSynchronize ();
  CUDA_CHECK_ERROR ("After kernel");

  cudaFree (d_scratch);

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
compare (float *a1, float *a2, int N) {
  int i = 0, res = 0;
  int print = 0;
  int zero_one = 0;
  int zero_two = 0;
  float eps = 1e-4f;

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
