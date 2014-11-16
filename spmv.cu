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
 * sparse matrix vector multiplication kernel.
 * 
 * @author Ben van Werkhoven <b.vanwerkhoven@esciencecenter.nl>
 * 
 */

#include <stdio.h>
#include <stdlib.h>

#include <math.h>
#include <time.h>

//CSR representation
#define M 128*1024		//number of rows

//used for filling matrix
#define N 128*1024		//number of cols

#define NNZ (int)(((long)N * (long)M)/(long)1000)	//number of non-zero elements

#define BLOCK_X 128

#define NSTREAMS 128

#define ITERATIONS 5


extern "C" {

  void print_CSR (int *row_start, int *col_idx, float *values, int elem);

  void print_vector (float *x, int n);

  void spmv_kernel (float *y, int *row_start, int *col_idx, float *values, float *x);

  void spmv_explicit (float *y, int *row_start, int *col_idx, float *values, float *x);
  void spmv_implicit (float *y, int *row_start, int *col_idx, float *values, float *x);

  void spmv_streams (float *y, int *row_start, int *col_idx, float *values, float *x);

  void spmv_hybrid (float *y, int *row_start, int *col_idx, float *values, float *x);

  void start_timer ();
  void stop_timer (float *);

  int compare (float *a1, float *a2, int n);

  __global__ void spmv_gpukernel (float *y, int *row_start, int *col_idx, float *values, float *x);

  __global__ void spmv_offsetkernel (float *y, int *row_start, int *col_idx, float *values, float *x, int offset);

}

//this number specifies the actual number of streams used at this point
int nStreams = 32;

cudaStream_t stream[NSTREAMS];
cudaEvent_t event_htod[NSTREAMS];


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
  int *row_start;
  int *col_idx;
  float *values;
  float *y;
  float *y_ref;
  float *x;

  err = cudaHostAlloc ((void **) &row_start, (M + 1) * sizeof (int), cudaHostAllocMapped);
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaHostAlloc: %s\n", cudaGetErrorString (err));
  }

  err = cudaHostAlloc ((void **) &col_idx, NNZ * sizeof (int), cudaHostAllocMapped);
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaHostAlloc: %s\n", cudaGetErrorString (err));
  }

  err = cudaHostAlloc ((void **) &values, NNZ * sizeof (float), cudaHostAllocMapped);
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaHostAlloc: %s\n", cudaGetErrorString (err));
  }

  err = cudaHostAlloc ((void **) &x, N * sizeof (float), cudaHostAllocMapped);
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaHostAlloc: %s\n", cudaGetErrorString (err));
  }

  err = cudaHostAlloc ((void **) &y_ref, M * sizeof (float), cudaHostAllocMapped);
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaHostAlloc: %s\n", cudaGetErrorString (err));
  }

  err = cudaHostAlloc ((void **) &y, M * sizeof (float), cudaHostAllocMapped);
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaHostAlloc: %s\n", cudaGetErrorString (err));
  }

  int i, j, k;
  for (i = 0; i < M; i++) {
    y[i] = 0.0;
    y_ref[i] = 0.0;
  }
  for (i = 0; i < N; i++) {
    x[i] = 0.00001 + (rand () % 10000) / 10000.0;
  }

  srand (time (NULL));
  //srand(13337);

  float density = (float) (NNZ) / (float) ((long) M * N);
  //int chance = (int)ceilf(1.0 / density);  //used only in more random matrix generation
  int elem = 0;
  int first_on_row = 1;

  int n_per_col = (int) ceilf (N * density);
  int *col_indexes = (int *) malloc (n_per_col * sizeof (int));

  float time;
  start_timer ();

  //Generate the sparse matrix
  //using a less random generation scheme because more random took forever
  for (j = 0; j < M; j++) {
    row_start[j] = elem;
    first_on_row = 1;

    //faster less-random matrix generation
    //generate column indexes
    for (i = 0; i < n_per_col; i++) {
      int sub_range = N / n_per_col;
      col_indexes[i] = i * sub_range + rand () % sub_range;
    }

    int min = N;
    int min_idx = -1;
    for (i = 0; i < n_per_col && elem < NNZ; i++) {

      //search lowest column idx
      for (k = 0; k < n_per_col; k++) {
	if (col_indexes[k] < min) {
	  min = col_indexes[k];
	  min_idx = k;
	}
	//if duplicate delete it
	if (col_indexes[k] == min) {
	  col_indexes[k] = N + 1;
	}
      }

      //sanity checks
      if (min < N) {
	if (elem >= NNZ) {
	  fprintf (stderr, "error: elem=%d > NNZ=%d", elem, NNZ);
	}

	//add value
	values[elem] = 0.0001 + ((rand () % 1000) / 1000.0);
	col_idx[elem] = min;

	if (first_on_row == 1) {
	  first_on_row = 0;
	  row_start[j] = elem;
	}
	elem++;
      }

      //for next search
      col_indexes[min_idx] = N + 1;
      min = N;
    }


    /* the more randomly generated way of matrix generation
       for (i=0; i<N && elem < NNZ; i++) {

       if ((rand() % chance) == 0) {
       //create non-zero
       values[elem] = 1.0 + ((rand() % 1000) / 100.0) ;
       col_idx[elem] = i;

       if (first_on_row == 1) { 
       first_on_row = 0;
       row_start[row_idx++] = elem; 
       }

       elem++;
       }

       }
     */

    //check for empty row and add same index
    if (first_on_row == 1) {
      row_start[j] = elem;
    }

  }
  //last element of row_start[] points to last element
  //in values[] and col_idx[] by definition
  row_start[M] = elem;
  free (col_indexes);

  stop_timer (&time);
  printf ("Matrix generated in: %.6f ms\n", time);
  printf ("elements in sparse matrix: %d\n", elem);
  printf ("target density=%f, achieved density=%f\n", density, (float) elem / ((float) M * (float) N));
  printf ("target NNZ=%d, achieved NNZ=%d\n", NNZ, elem);

  printf ("\n");

  print_CSR (row_start, col_idx, values, elem);

  printf ("finished generating sparse matrix, starting computation ... \n");
  fflush (stdout);
  start_timer ();

  spmv_kernel (y_ref, row_start, col_idx, values, x);

  stop_timer (&time);
  printf ("SPMV CPU: %.6f ms\n", time);

  // now run the four implementations

  for (i = 0; i < ITERATIONS; i++) {
    spmv_explicit (y, row_start, col_idx, values, x);
  }
  compare (y_ref, y, M);

  for (i = 0; i < ITERATIONS; i++) {
    spmv_implicit (y, row_start, col_idx, values, x);
  }
  compare (y_ref, y, M);

  for (i = 0; i < ITERATIONS; i++) {
    spmv_streams (y, row_start, col_idx, values, x);
  }
  compare (y_ref, y, M);

  for (i = 0; i < ITERATIONS; i++) {
    spmv_hybrid (y, row_start, col_idx, values, x);
  }
  compare (y_ref, y, M);

  //clean up
  cudaFreeHost (row_start);
  cudaFreeHost (col_idx);
  cudaFreeHost (values);
  cudaFreeHost (x);
  cudaFreeHost (y);
  cudaFreeHost (y_ref);

  //flush info for profiling
  cudaDeviceSynchronize ();
  cudaDeviceReset ();

  return 0;
}


/*
 * Utility function for printing the sparse matrix in CSR representation
 */
void
print_CSR (int *row_start, int *col_idx, float *values, int elem) {
  int i;

  //sanity check
  if (elem > 100)
    return;

  printf ("NNZ=%d, M=%d\n", elem, M);

  printf ("row_start[]=\n {");
  for (i = 0; i < M; i++) {
    printf ("%d", row_start[i]);
    if (i < M - 1)
      printf (", ");
  }
  printf ("}\n");

  printf ("col_idx[]=\n {");
  for (i = 0; i < elem; i++) {
    printf ("%d", col_idx[i]);
    if (i < elem - 1)
      printf (", ");
  }
  printf ("}\n");

  printf ("values[]=\n {");
  for (i = 0; i < elem; i++) {
    printf ("%.2f", values[i]);
    if (i < elem - 1)
      printf (", ");
  }
  printf ("}\n");


}

/*
 * Utility function for printing a small vector
 */
void
print_vector (float *x, int n) {
  printf ("x=\n {");
  for (int i = 0; i < n; i++) {
    printf ("%.2f", x[i]);
    if (i < n - 1)
      printf (", ");
  }
  printf ("}\n");
}



/*
 * Simple kernel for performing a sparse matrix vector multiplication
 * for a sparse matrix in CSR representation
 *
 * y = A * x
 */
void
spmv_kernel (float *y, int *row_start, int *col_idx, float *values, float *x) {
  int i, j;
  //for each row
  for (i = 0; i < M; i++) {
    //for each element on row
    for (j = row_start[i]; j < row_start[i + 1]; j++) {
      y[i] += values[j] * x[col_idx[j]];
    }
  }
}


/*
 * Simple CUDA kernel for performing a sparse matrix vector multiplication
 * for a sparse matrix in CSR representation
 *
 * y = A * x
 */
__global__ void
spmv_gpukernel (float *y, int *row_start, int *col_idx, float *values, float *x) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float ly = 0.0;
  if (i < M) {
    //for each element on row
    int start = row_start[i];
    int end = row_start[i + 1];
    for (int j = start; j < end; j++) {
      int col = col_idx[j];
      float val = values[j];
      float lx = x[col];
      ly += val * lx;
    }
    y[i] = ly;
  }
}

/*
 * Simple CUDA kernel for performing a sparse matrix vector multiplication
 * for a sparse matrix in CSR representation
 *
 * y = A * x
 * 
 * The offset kernel allows an offset to be specified that directs the threads towards
 * a row in the sparse matrix from which to start the computation. This is used
 * when the computation is split across different streams.
 * 
 */
__global__ void
spmv_offsetkernel (float *y, int *row_start, int *col_idx, float *values, float *x, int offset) {
  int i = offset + blockIdx.x * blockDim.x + threadIdx.x;
  float ly = 0.0;
  if (i < M) {
    //for each element on row
    for (int j = row_start[i]; j < row_start[i + 1]; j++) {
      ly += values[j] * x[col_idx[j]];
    }
    y[i] = ly;
  }
}



/*
 * Host code that invokes the sparse matrix vector multiplication kernel
 *
 * The explicit implementation uses explicit memory copy
 * statements to move all data to the GPU, executes the
 * GPU kernel, and uses memory copies to copy the output
 * data back to host memory. This implementation achieves
 * no overlap between transfers and/or computation.
 *
 */
void
spmv_explicit (float *y, int *row_start, int *col_idx, float *values, float *x) {

  cudaError_t err;
  int *d_row_start;
  int *d_col_idx;
  float *d_values;
  float *d_x;
  float *d_y;

  err = cudaMalloc ((void **) &d_row_start, (M + 1) * sizeof (int));
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMalloc d_row_start: %s\n", cudaGetErrorString (err));
  }

  err = cudaMalloc ((void **) &d_col_idx, NNZ * sizeof (int));
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMalloc d_col_idx: %s\n", cudaGetErrorString (err));
  }

  err = cudaMalloc ((void **) &d_values, NNZ * sizeof (float));
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMalloc d_values: %s\n", cudaGetErrorString (err));
  }

  err = cudaMalloc ((void **) &d_x, N * sizeof (float));
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMalloc d_x: %s\n", cudaGetErrorString (err));
  }

  err = cudaMalloc ((void **) &d_y, M * sizeof (float));
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMalloc d_y: %s\n", cudaGetErrorString (err));
  }

  dim3 threads (BLOCK_X, 1);
  dim3 grid ((int) ceilf ((float) M / (float) (BLOCK_X)), 1);

  cudaDeviceSynchronize ();
  err = cudaGetLastError ();
  if (err != cudaSuccess) {
    fprintf (stderr, "Error occured: %s\n", cudaGetErrorString (err));
  }

  float time;

  //Measure total execution time
  cudaDeviceSynchronize ();
  start_timer ();

  err = cudaMemcpyAsync (d_row_start, row_start, (M + 1) * sizeof (int), cudaMemcpyHostToDevice, stream[1]);
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMemcpy host to device row_start: %s\n", cudaGetErrorString (err));
  }

  err = cudaMemcpyAsync (d_col_idx, col_idx, NNZ * sizeof (int), cudaMemcpyHostToDevice, stream[1]);
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMemcpy host to device col_idx: %s\n", cudaGetErrorString (err));
  }

  err = cudaMemcpyAsync (d_values, values, NNZ * sizeof (float), cudaMemcpyHostToDevice, stream[1]);
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMemcpy host to device values: %s\n", cudaGetErrorString (err));
  }

  err = cudaMemcpyAsync (d_x, x, N * sizeof (float), cudaMemcpyHostToDevice, stream[1]);
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMemcpy host to device x: %s\n", cudaGetErrorString (err));
  }

  spmv_gpukernel <<< grid, threads, 0, stream[1] >>> (d_y, d_row_start, d_col_idx, d_values, d_x);

  err = cudaMemcpyAsync (y, d_y, M * sizeof (float), cudaMemcpyDeviceToHost, stream[1]);
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMemcpy device to host y: %s\n", cudaGetErrorString (err));
  }

  cudaDeviceSynchronize ();
  stop_timer (&time);
  printf ("EXPLICIT: %.6f ms\n", time);

  //Measure kernel execution time
  cudaDeviceSynchronize ();
  start_timer ();

  spmv_gpukernel <<< grid, threads, 0, stream[1] >>> (d_y, d_row_start, d_col_idx, d_values, d_x);

  cudaDeviceSynchronize ();
  stop_timer (&time);
  printf ("EXPLICIT kernel:\t %.6f ms\n", time);

  cudaFree (d_row_start);
  cudaFree (d_col_idx);
  cudaFree (d_values);
  cudaFree (d_x);
  cudaFree (d_y);

}


/*
 * Host code that invokes the sparse matrix vector multiplication kernel
 *
 * The implicit implementation uses device-mapped host memory rather
 * than explicit memory copy statements. A different kernel is used
 * to ensure strictly coalesced access to system memory.
 *
 */
void
spmv_implicit (float *y, int *row_start, int *col_idx, float *values, float *x) {

  cudaError_t err;

  dim3 threads (BLOCK_X, 1);
  dim3 grid ((int) ceilf ((float) M / (float) (BLOCK_X)), 1);

  cudaDeviceSynchronize ();
  err = cudaGetLastError ();
  if (err != cudaSuccess) {
    fprintf (stderr, "Error occured: %s\n", cudaGetErrorString (err));
  }

  float time;
  cudaDeviceSynchronize ();
  start_timer ();

  spmv_gpukernel <<< grid, threads, 0, stream[1] >>> (y, row_start, col_idx, values, x);

  cudaDeviceSynchronize ();
  stop_timer (&time);
  printf ("IMPLICIT: %.6f ms\n", time);

}


/*
 * Host code that invokes the sparse matrix vector multiplication kernel
 *
 * The streams implementation uses CUDA streams combined
 * with explicit memory copy statements. This way transfers
 * in one stream may overlap with computation and transfers
 * in other streams.
 *
 */
void
spmv_streams (float *y, int *row_start, int *col_idx, float *values, float *x) {

  int k;

  cudaError_t err;
  int *d_row_start;
  int *d_col_idx;
  float *d_values;
  float *d_x;
  float *d_y;

  err = cudaMalloc ((void **) &d_row_start, (M + 1) * sizeof (int));
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMalloc d_row_start: %s\n", cudaGetErrorString (err));
  }

  err = cudaMalloc ((void **) &d_col_idx, NNZ * sizeof (int));
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMalloc d_col_idx: %s\n", cudaGetErrorString (err));
  }

  err = cudaMalloc ((void **) &d_values, NNZ * sizeof (float));
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMalloc d_values: %s\n", cudaGetErrorString (err));
  }

  err = cudaMalloc ((void **) &d_x, N * sizeof (float));
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMalloc d_x: %s\n", cudaGetErrorString (err));
  }

  err = cudaMalloc ((void **) &d_y, M * sizeof (float));
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMalloc d_y: %s\n", cudaGetErrorString (err));
  }

  dim3 threads (BLOCK_X, 1);
  dim3 grid ((int) ceilf ((float) M / (float) (BLOCK_X)), 1);

  cudaDeviceSynchronize ();
  err = cudaGetLastError ();
  if (err != cudaSuccess) {
    fprintf (stderr, "Error occured: %s\n", cudaGetErrorString (err));
  }

  //determine rows per stream
  int nstr = NSTREAMS;
  if (nStreams != -1)
    nstr = nStreams;

  //setup for spmv_offsetkernel
  int tb = M / BLOCK_X;
  int rps = (tb / nstr) * BLOCK_X;
  grid.x = tb / nstr;
  if (tb % nstr != 0) {
    fprintf (stderr, "Error nStreams=%d not a divisor of the number of thread blocks=%d\n", nstr, M);
  }

  float time;
  cudaDeviceSynchronize ();
  start_timer ();

  //all streams need x to be on the device
  err = cudaMemcpyAsync (d_x, x, N * sizeof (float), cudaMemcpyHostToDevice, stream[0]);
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMemcpy host to device x: %s\n", cudaGetErrorString (err));
  }

  //copy first element in row_start, copy rest as needed by stream
  err = cudaMemcpyAsync (d_row_start, row_start, sizeof (int), cudaMemcpyHostToDevice, stream[0]);
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMemcpy host to device row_start: %s\n", cudaGetErrorString (err));
  }

  err = cudaEventRecord (event_htod[0], stream[0]);
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaEventRecord htod: %s\n", cudaGetErrorString (err));
  }

  for (k = 0; k < nstr; k++) {
    int start = row_start[rps * k];
    int end = row_start[rps * (k + 1)];
    //printf("stream %d: start=%d, end=%d\n", k, start, end);

    err = cudaStreamWaitEvent (stream[k], event_htod[0], 0);
    if (err != cudaSuccess) {
      fprintf (stderr, "Error in cudaStreamWaitEvent htod 1: %s\n", cudaGetErrorString (err));
    }

    //enforce strict ordering of copy operations per stream
    if (k > 0) {
      err = cudaStreamWaitEvent (stream[k], event_htod[k - 1], 0);
      if (err != cudaSuccess) {
	fprintf (stderr, "Error in cudaStreamWaitEvent htod 1: %s\n", cudaGetErrorString (err));
      }
    }

    err = cudaMemcpyAsync (d_row_start + 1 + rps * k, row_start + 1 + rps * k, rps * sizeof (int), cudaMemcpyHostToDevice, stream[k]);
    if (err != cudaSuccess) {
      fprintf (stderr, "Stream %d: Error in cudaMemcpy host to device row_start: %s\n", k, cudaGetErrorString (err));
    }

    err = cudaMemcpyAsync (d_col_idx + start, col_idx + start, (end - start) * sizeof (int), cudaMemcpyHostToDevice, stream[k]);
    if (err != cudaSuccess) {
      fprintf (stderr, "Stream %d: Error in cudaMemcpy host to device col_idx: %s\n", k, cudaGetErrorString (err));
    }

    err = cudaMemcpyAsync (d_values + start, values + start, (end - start) * sizeof (float), cudaMemcpyHostToDevice, stream[k]);
    if (err != cudaSuccess) {
      fprintf (stderr, "Stream %d: Error in cudaMemcpy host to device values: %s\n", k, cudaGetErrorString (err));
    }

    //enforce strict ordering of copy operations per stream
    err = cudaEventRecord (event_htod[k], stream[k]);
    if (err != cudaSuccess) {
      fprintf (stderr, "Error in cudaEventRecord htod: %s\n", cudaGetErrorString (err));
    }

//  }

//  for (k=0; k<nstr; k++) {
    spmv_offsetkernel <<< grid, threads, 0, stream[k] >>> (d_y, d_row_start, d_col_idx, d_values, d_x, k * rps);
  }

  for (k = 0; k < nstr; k++) {
    err = cudaMemcpyAsync (y + k * rps, d_y + k * rps, rps * sizeof (float), cudaMemcpyDeviceToHost, stream[k]);
    if (err != cudaSuccess) {
      fprintf (stderr, "Error in cudaMemcpy device to host y: %s\n", cudaGetErrorString (err));
    }
  }

  cudaDeviceSynchronize ();
  stop_timer (&time);
  printf ("STREAMS: %.6f ms\n", time);


  cudaFree (d_row_start);
  cudaFree (d_col_idx);
  cudaFree (d_values);
  cudaFree (d_x);
  cudaFree (d_y);

}


/*
 * Host code that invokes the sparse matrix vector multiplication kernel
 *
 * The Hybrid implementation uses CUDA streams combined
 * with explicit memory copy statements for the input data
 * and uses device-mapped host memory to copy the output data
 * back to host memory. 
 *
 */
void
spmv_hybrid (float *y, int *row_start, int *col_idx, float *values, float *x) {

  int k;

  cudaError_t err;
  int *d_row_start;
  int *d_col_idx;
  float *d_values;
  float *d_x;

  err = cudaMalloc ((void **) &d_row_start, (M + 1) * sizeof (int));
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMalloc d_row_start: %s\n", cudaGetErrorString (err));
  }

  err = cudaMalloc ((void **) &d_col_idx, NNZ * sizeof (int));
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMalloc d_col_idx: %s\n", cudaGetErrorString (err));
  }

  err = cudaMalloc ((void **) &d_values, NNZ * sizeof (float));
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMalloc d_values: %s\n", cudaGetErrorString (err));
  }

  err = cudaMalloc ((void **) &d_x, N * sizeof (float));
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMalloc d_x: %s\n", cudaGetErrorString (err));
  }

  dim3 threads (BLOCK_X, 1);
  dim3 grid ((int) ceilf ((float) M / (float) (BLOCK_X)), 1);

  cudaDeviceSynchronize ();
  err = cudaGetLastError ();
  if (err != cudaSuccess) {
    fprintf (stderr, "Error occured: %s\n", cudaGetErrorString (err));
  }

  //determine rows per stream
  int nstr = NSTREAMS;
  if (nStreams != -1)
    nstr = nStreams;

  //setup for spmv_offsetkernel
  int tb = M / BLOCK_X;
  int rps = (tb / nstr) * BLOCK_X;
  grid.x = tb / nstr;
  if (tb % nstr != 0) {
    fprintf (stderr, "Error nStreams=%d not a divisor of the number of thread blocks=%d\n", nstr, M);
  }

  float time;
  cudaDeviceSynchronize ();
  start_timer ();

  //all streams need x to be on the device
  err = cudaMemcpyAsync (d_x, x, N * sizeof (float), cudaMemcpyHostToDevice, stream[0]);
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMemcpy host to device x: %s\n", cudaGetErrorString (err));
  }

  //copy first element in row_start, copy rest as needed by stream
  err = cudaMemcpyAsync (d_row_start, row_start, sizeof (int), cudaMemcpyHostToDevice, stream[0]);
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaMemcpy host to device row_start: %s\n", cudaGetErrorString (err));
  }

  err = cudaEventRecord (event_htod[0], stream[0]);
  if (err != cudaSuccess) {
    fprintf (stderr, "Error in cudaEventRecord htod: %s\n", cudaGetErrorString (err));
  }

  for (k = 0; k < nstr; k++) {
    int start = row_start[rps * k];
    int end = row_start[rps * (k + 1)];

    err = cudaStreamWaitEvent (stream[k], event_htod[0], 0);
    if (err != cudaSuccess) {
      fprintf (stderr, "Error in cudaStreamWaitEvent htod 1: %s\n", cudaGetErrorString (err));
    }

    //enforce strict ordering of copy operations per stream
    if (k > 0) {
      err = cudaStreamWaitEvent (stream[k], event_htod[k - 1], 0);
      if (err != cudaSuccess) {
	fprintf (stderr, "Error in cudaStreamWaitEvent htod 1: %s\n", cudaGetErrorString (err));
      }
    }

    err = cudaMemcpyAsync (d_row_start + 1 + rps * k, row_start + 1 + rps * k, rps * sizeof (int), cudaMemcpyHostToDevice, stream[k]);
    if (err != cudaSuccess) {
      fprintf (stderr, "Stream %d: Error in cudaMemcpy host to device row_start: %s\n", k, cudaGetErrorString (err));
    }

    err = cudaMemcpyAsync (d_col_idx + start, col_idx + start, (end - start) * sizeof (int), cudaMemcpyHostToDevice, stream[k]);
    if (err != cudaSuccess) {
      fprintf (stderr, "Stream %d: Error in cudaMemcpy host to device col_idx: %s\n", k, cudaGetErrorString (err));
    }

    err = cudaMemcpyAsync (d_values + start, values + start, (end - start) * sizeof (float), cudaMemcpyHostToDevice, stream[k]);
    if (err != cudaSuccess) {
      fprintf (stderr, "Stream %d: Error in cudaMemcpy host to device values: %s\n", k, cudaGetErrorString (err));
    }

    //enforce strict ordering of copy operations per stream
    err = cudaEventRecord (event_htod[k], stream[k]);
    if (err != cudaSuccess) {
      fprintf (stderr, "Error in cudaEventRecord htod: %s\n", cudaGetErrorString (err));
    }

//  }

//  for (k=0; k<nstr; k++) {
    spmv_offsetkernel <<< grid, threads, 0, stream[k] >>> (y, d_row_start, d_col_idx, d_values, d_x, k * rps);
//  }

  }

  cudaDeviceSynchronize ();
  stop_timer (&time);
  printf ("HYBRID: %.6f ms\n", time);

  cudaFree (d_row_start);
  cudaFree (d_col_idx);
  cudaFree (d_values);
  cudaFree (d_x);

}




/*
 * Compare function that compares two arrays of length N for similarity
 * 
 * This function performs a number of different tests, for example the number of
 * values at an epsilon from 0.0 should be similar in both arrays and may not
 * be greater than 1/4th of the array. Additionally NaN values are treated as
 * errors.
 *
 * The value of eps should be adjusted to something reasonable given the
 * fact that CPU and GPU do not produce exactly the same numerical results. 
 */
int
compare (float *a1, float *a2, int n) {
  int i = 0, res = 0;
  int print = 0;
  int zero_one = 0;
  int zero_two = 0;
  float eps = 0.0001;

  for (i = 0; i < n; i++) {

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

  if (zero_one > (n / 4)) {
    fprintf (stderr, "Error: array1 contains %d zeros\n", zero_one);
  }
  if (zero_two > (n / 4)) {
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
