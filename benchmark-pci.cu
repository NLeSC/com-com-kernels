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

#include <stdio.h>
#include <math.h>

#define NSTREAMS            256	// 256

#define TOTAL_SIZE          (1 << 27)	//1 GB of doubles

#define ITERATIONS	    10

//modes for testing
#define HOSTTODEVICE	1
#define DEVICETOHOST	2
#define HYBRID		3


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
  void start_timer ();
  void stop_timer (float *);

  void measure (int size, int mode);

  __global__ void mappedMemoryCopy (double *dst, double *src, int n);

}

cudaStream_t stream[NSTREAMS];

double *hostptr = 0;
double *devptr = 0;

double Lo;
double G;
double g;

int
main () {
  cudaError_t err;
  int k;

  cudaSetDeviceFlags (cudaDeviceMapHost);
  cudaSetDevice (0);

  cudaDeviceSynchronize ();
  CUDA_CHECK_ERROR ("After device initialization");

  //setup streams
  for (k = 0; k < NSTREAMS; k++) {
    err = cudaStreamCreate (&stream[k]);
    if (err != cudaSuccess) {
      fprintf (stderr, "Error in cudaStreamCreate: %s\n", cudaGetErrorString (err));
    }
  }

  err = cudaHostAlloc ((void **) &hostptr, TOTAL_SIZE * sizeof (double), cudaHostAllocMapped);
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaHostAlloc: %s\n", cudaGetErrorString (err));
  }

  //fill the memory
  for (int i = 0; i < TOTAL_SIZE; i++) {
    hostptr[i] = (i & 0xff) * 1.0;
  }

  err = cudaMalloc ((void **) &devptr, TOTAL_SIZE * sizeof (double));
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMalloc: %s\n", cudaGetErrorString (err));
  }

  cudaDeviceSynchronize ();
  CUDA_CHECK_ERROR ("After memory setup");

  printf ("Measure HostToDevice\n");
  measure (TOTAL_SIZE, HOSTTODEVICE);

  printf ("Measure DeviceToHost\n");
  measure (TOTAL_SIZE, DEVICETOHOST);

  printf ("Measure Hybrid\n");
  measure (TOTAL_SIZE, HYBRID);

  //clean up
  cudaFreeHost (hostptr);
  cudaFree (devptr);

}

void
measure (int size, int mode) {
  cudaError_t err;
  int i, k;

  //setup timers and parameters
  float time;
  float total_time = 0.0f;
  int nStreams;
  int elementsPerStream;

  //setup grids for kernel launches
  dim3 single_block (1, 1);
  dim3 single_thread (1, 1);
  dim3 threads (256, 1);
  dim3 grid (1, 1);
  int max_blocks = (64 * 1024) - 1;
  int blocks = (int) ceilf ((float) size / (float) 256.0);
  grid.x = min (blocks, max_blocks);
  grid.y = (int) ceilf ((float) blocks / (float) max_blocks);

  cudaDeviceSynchronize ();
  CUDA_CHECK_ERROR ("Before starting new measurement");

  //First do some transfers to wake up the device
  nStreams = 8;
  elementsPerStream = TOTAL_SIZE / nStreams;

  cudaDeviceSynchronize ();
  for (i = 0; i < ITERATIONS; i++) {

    for (k = 0; k < nStreams; k++) {
      err =
	cudaMemcpyAsync (devptr + k * elementsPerStream, hostptr + k * elementsPerStream, elementsPerStream * sizeof (double), cudaMemcpyHostToDevice,
			 stream[k]);
      if (err != cudaSuccess) {
	fprintf (stderr, "Error in cudaMemcpy host to device: %s\n", cudaGetErrorString (err));
      }

      err =
	cudaMemcpyAsync (hostptr + k * elementsPerStream, devptr + k * elementsPerStream, elementsPerStream * sizeof (double), cudaMemcpyDeviceToHost,
			 stream[k + 1]);
      if (err != cudaSuccess) {
	fprintf (stderr, "Error in cudaMemcpy host to device: %s\n", cudaGetErrorString (err));
      }
    }

  }
  cudaDeviceSynchronize ();

  //Now estimate model parameters  L+o, G, and g

  //estimate L+o using a very small message, thus assuming kG = 0.
  for (i = 0; i < ITERATIONS; i++) {
    cudaDeviceSynchronize ();
    start_timer ();

    if (mode == HOSTTODEVICE) {
      cudaMemcpyAsync (devptr, hostptr, 1 * sizeof (double), cudaMemcpyHostToDevice, stream[1]);
    }
    else if (mode == DEVICETOHOST) {
      cudaMemcpyAsync (hostptr, devptr, 1 * sizeof (double), cudaMemcpyDeviceToHost, stream[1]);
    }
    else if (mode == HYBRID) {
      mappedMemoryCopy <<< single_block, single_thread, 0, stream[1] >>> (hostptr, devptr, 1);
    }

    cudaDeviceSynchronize ();
    stop_timer (&time);

    total_time += time;
  }

  Lo = (double) total_time / (ITERATIONS);
  printf ("L+o=%.10f\n", Lo);
  total_time = 0.0;

  //Now estimate G
  nStreams = 1;
  if (mode == HYBRID) {
    nStreams = 128;

    max_blocks = (64 * 1024) - 1;
    blocks = (int) ceilf ((int) ceilf ((float) size / (float) nStreams) / (float) 256.0);

    grid.x = min (blocks, max_blocks);
    grid.y = (int) ceilf ((float) blocks / (float) max_blocks);

  }
  long long totalElements = (long long) 0;
  double sum_time = 0.0;

  for (int j = 0; j < 10; j++) {
    elementsPerStream = size / nStreams;
    totalElements += elementsPerStream * nStreams;

    for (i = 0; i < ITERATIONS; i++) {
      cudaDeviceSynchronize ();
      start_timer ();

      for (k = 0; k < nStreams; k++) {
	if (mode == HOSTTODEVICE) {
	  err =
	    cudaMemcpyAsync (devptr + k * elementsPerStream, hostptr + k * elementsPerStream, elementsPerStream * sizeof (double), cudaMemcpyHostToDevice,
			     stream[k]);
	  if (err != cudaSuccess) {
	    fprintf (stderr, "Error in cudaMemcpy host to device: %s\n", cudaGetErrorString (err));
	  }
	}
	else if (mode == DEVICETOHOST) {
	  err =
	    cudaMemcpyAsync (hostptr + k * elementsPerStream, devptr + k * elementsPerStream, elementsPerStream * sizeof (double), cudaMemcpyDeviceToHost,
			     stream[k]);
	  if (err != cudaSuccess) {
	    fprintf (stderr, "Error in cudaMemcpy host to device: %s\n", cudaGetErrorString (err));
	  }
	}
	else if (mode == HYBRID) {
	  err =
	    cudaMemcpyAsync (devptr + k * elementsPerStream, hostptr + k * elementsPerStream, elementsPerStream * sizeof (double), cudaMemcpyHostToDevice,
			     stream[k + 1]);
	  if (err != cudaSuccess)
	    fprintf (stderr, "Error in cudaMemcpy host to device: %s\n", cudaGetErrorString (err));
	  mappedMemoryCopy <<< grid, threads, 0, stream[k] >>> (hostptr + k * elementsPerStream, devptr + k * elementsPerStream, elementsPerStream);
	}

      }

      cudaDeviceSynchronize ();
      stop_timer (&time);

      total_time += time;
    }

    sum_time += (double) total_time / (ITERATIONS);
    total_time = 0.0;
  }

  double wG = sum_time - (10 * ITERATIONS) * Lo;
  if (mode == HYBRID) {
    wG = wG - (10 * ITERATIONS) * (nStreams - 1) * g;
  }

  double w = (double) totalElements * (double) sizeof (double);

  G = wG / w;

  printf ("G=%20.17e, G=%.10f, BW=%.6f MB/s\n", G, G, (1000.0 / G) / (1 << 20));

  //Now estimate g

  if (mode == HYBRID) {
    return;
  }				//don't measure g for hybrid

  nStreams = 32;
  elementsPerStream = size / nStreams;
  g = 0.0;

  for (i = 0; i < ITERATIONS; i++) {
    cudaDeviceSynchronize ();
    start_timer ();

    for (k = 0; k < nStreams; k++) {
      if (mode == HOSTTODEVICE) {
	err =
	  cudaMemcpyAsync (devptr + k * elementsPerStream, hostptr + k * elementsPerStream, elementsPerStream * sizeof (double), cudaMemcpyHostToDevice,
			   stream[k]);
	if (err != cudaSuccess) {
	  fprintf (stderr, "Error in cudaMemcpy host to device: %s\n", cudaGetErrorString (err));
	}
      }
      else if (mode == DEVICETOHOST) {
	err =
	  cudaMemcpyAsync (hostptr + k * elementsPerStream, devptr + k * elementsPerStream, elementsPerStream * sizeof (double), cudaMemcpyDeviceToHost,
			   stream[k]);
	if (err != cudaSuccess) {
	  fprintf (stderr, "Error in cudaMemcpy host to device: %s\n", cudaGetErrorString (err));
	}
      }

    }

    cudaDeviceSynchronize ();
    stop_timer (&time);

    total_time += time;
  }

  //L+o+(wG*nStreams)+(g*(nStreams-1)) = 
  float g_time = total_time / (float) (ITERATIONS);
  total_time = 0.0;

  //-(L+o)
  double tmp = (double) g_time - Lo;

  //-(wG*nStreams)
  tmp = tmp - (G * (double) (size * sizeof (double)));

  //= (g*(nStreams-1))
  g = tmp / (double) (nStreams - 1);
  printf ("g=%f\n", g);


}


__global__ void
mappedMemoryCopy (double *dst, double *src, int n) {

  //obtain index
  int i = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {

    double temp = src[i];
    dst[i] = temp;

  }

}
