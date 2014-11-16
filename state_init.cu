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

/*
 * This file should be included into files that would like to use the state kernel.
 *
 * @author Ben van Werkhoven <b.vanwerkhoven@esciencecenter.nl>
 */

//bounds checkers
#define TMIN -2.0
#define TMAX 999.0
#define SMIN 0.0
#define SMAX 0.999

//all constants needed on the GPU
__constant__ double d_mwjfnums0t1;
__constant__ double d_mwjfnums0t3;
__constant__ double d_mwjfnums1t1;
__constant__ double d_mwjfnums2t0;
__constant__ double d_mwjfdens0t2;
__constant__ double d_mwjfdens0t4;
__constant__ double d_mwjfdens1t0;
__constant__ double d_mwjfdens1t1;
__constant__ double d_mwjfdens1t3;
__constant__ double d_mwjfdensqt0;
__constant__ double d_mwjfdensqt2;

__constant__ double d_mwjfnums0t0[KM];
__constant__ double d_mwjfnums0t2[KM];
__constant__ double d_mwjfnums1t0[KM];

__constant__ double d_mwjfdens0t0[KM];
__constant__ double d_mwjfdens0t1[KM];
__constant__ double d_mwjfdens0t3[KM];

__constant__ double d_grav;

//constants used in POP copied from the Parallel Ocean Program
//from the Fortran 90 file constants.F90
double c0 = 0.0;
double c1 = 1.0;
double c2 = 2.0;
double c3 = 3.0;
double c4 = 4.0;
double c5 = 5.0;
double c8 = 8.0;
double c10 = 10.0;
double c16 = 16.0;
double c1000 = 1000.0;
double c10000 = 10000.0;
double c1p5 = 1.5;
double p33 = c1 / c3;
double p5 = 0.500;
double p25 = 0.250;
double p125 = 0.125;
double p001 = 0.001;
double eps = 1.0e-10;
double eps2 = 1.0e-20;
double bignum = 1.0e+30;
double grav = 980.6;

double mwjfnp0s0t0 = 9.9984369900000e+2 * 0.001;
double mwjfnp0s0t1 = 7.3521284000000e+0 * 0.001;
double mwjfnp0s0t2 = -5.4592821100000e-2 * 0.001;
double mwjfnp0s0t3 = 3.9847670400000e-4 * 0.001;
double mwjfnp0s1t0 = 2.9693823900000e+0 * 0.001;
double mwjfnp0s1t1 = -7.2326881300000e-3 * 0.001;
double mwjfnp0s2t0 = 2.1238234100000e-3 * 0.001;
double mwjfnp1s0t0 = 1.0400459100000e-2 * 0.001;
double mwjfnp1s0t2 = 1.0397052900000e-7 * 0.001;
double mwjfnp1s1t0 = 5.1876188000000e-6 * 0.001;
double mwjfnp2s0t0 = -3.2404182500000e-8 * 0.001;
double mwjfnp2s0t2 = -1.2386936000000e-11 * 0.001;

double mwjfdp0s0t0 = 1.0e+0;
double mwjfdp0s0t1 = 7.2860673900000e-3;
double mwjfdp0s0t2 = -4.6083554200000e-5;
double mwjfdp0s0t3 = 3.6839057300000e-7;
double mwjfdp0s0t4 = 1.8080918600000e-10;
double mwjfdp0s1t0 = 2.1469170800000e-3;
double mwjfdp0s1t1 = -9.2706248400000e-6;
double mwjfdp0s1t3 = -1.7834364300000e-10;
double mwjfdp0sqt0 = 4.7653412200000e-6;
double mwjfdp0sqt2 = 1.6341073600000e-9;
double mwjfdp1s0t0 = 5.3084887500000e-6;
double mwjfdp2s0t3 = -3.0317512800000e-16;
double mwjfdp3s0t1 = -1.2793413700000e-17;

int cuda_state_initialized = 0;

/*
 * This function passes all constants required by the state kernel to the GPU
 */
void
cuda_state_initialize (double *pressz) {
  cudaError_t err;

  if (cuda_state_initialized == 0) {
    cuda_state_initialized = 1;

    err = cudaMemcpyToSymbol (d_mwjfnums0t1, &mwjfnp0s0t1, sizeof (double), 0, cudaMemcpyHostToDevice);	//= mwjfnp0s0t1
    if (err != cudaSuccess) {
      fprintf (stderr, "Error doing cudaMemcpyToSymbol d_mwjfnums0t1 %s\n", cudaGetErrorString (err));
    }
    err = cudaMemcpyToSymbol (d_mwjfnums0t3, &mwjfnp0s0t3, sizeof (double), 0, cudaMemcpyHostToDevice);	//= mwjfnp0s0t3
    if (err != cudaSuccess) {
      fprintf (stderr, "Error doing cudaMemcpyToSymbol d_mwjfnums0t3 %s\n", cudaGetErrorString (err));
    }
    err = cudaMemcpyToSymbol (d_mwjfnums1t1, &mwjfnp0s1t1, sizeof (double), 0, cudaMemcpyHostToDevice);	//= mwjfnp0s1t1
    if (err != cudaSuccess) {
      fprintf (stderr, "Error doing cudaMemcpyToSymbol d_mwjfnums1t1 %s\n", cudaGetErrorString (err));
    }
    err = cudaMemcpyToSymbol (d_mwjfnums2t0, &mwjfnp0s2t0, sizeof (double), 0, cudaMemcpyHostToDevice);	//= mwjfnp0s2t0
    if (err != cudaSuccess) {
      fprintf (stderr, "Error doing cudaMemcpyToSymbol d_mwjfnums2t0 %s\n", cudaGetErrorString (err));
    }
    err = cudaMemcpyToSymbol (d_mwjfdens0t2, &mwjfdp0s0t2, sizeof (double), 0, cudaMemcpyHostToDevice);	//= mwjfdp0s0t2
    if (err != cudaSuccess) {
      fprintf (stderr, "Error doing cudaMemcpyToSymbol d_mwjfdens0t2 %s\n", cudaGetErrorString (err));
    }
    err = cudaMemcpyToSymbol (d_mwjfdens0t4, &mwjfdp0s0t4, sizeof (double), 0, cudaMemcpyHostToDevice);	//= mwjfdp0s0t4
    if (err != cudaSuccess) {
      fprintf (stderr, "Error doing cudaMemcpyToSymbol d_mwjfdens0t4 %s\n", cudaGetErrorString (err));
    }
    err = cudaMemcpyToSymbol (d_mwjfdens1t0, &mwjfdp0s1t0, sizeof (double), 0, cudaMemcpyHostToDevice);	//= mwjfdp0s1t0
    if (err != cudaSuccess) {
      fprintf (stderr, "Error doing cudaMemcpyToSymbol d_mwjfdens1t0 %s\n", cudaGetErrorString (err));
    }
    err = cudaMemcpyToSymbol (d_mwjfdens1t1, &mwjfdp0s1t1, sizeof (double), 0, cudaMemcpyHostToDevice);	//= mwjfdp0s1t1
    if (err != cudaSuccess) {
      fprintf (stderr, "Error doing cudaMemcpyToSymbol d_mwjfdens1t1 %s\n", cudaGetErrorString (err));
    }
    err = cudaMemcpyToSymbol (d_mwjfdens1t3, &mwjfdp0s1t3, sizeof (double), 0, cudaMemcpyHostToDevice);	//= mwjfdp0s1t3
    if (err != cudaSuccess) {
      fprintf (stderr, "Error doing cudaMemcpyToSymbol d_mwjfdens1t3 %s\n", cudaGetErrorString (err));
    }
    err = cudaMemcpyToSymbol (d_mwjfdensqt0, &mwjfdp0sqt0, sizeof (double), 0, cudaMemcpyHostToDevice);	//= mwjfdp0sqt0
    if (err != cudaSuccess) {
      fprintf (stderr, "Error doing cudaMemcpyToSymbol d_mwjfdensqt0 %s\n", cudaGetErrorString (err));
    }
    err = cudaMemcpyToSymbol (d_mwjfdensqt2, &mwjfdp0sqt2, sizeof (double), 0, cudaMemcpyHostToDevice);	//= mwjfdp0sqt2
    if (err != cudaSuccess) {
      fprintf (stderr, "Error doing cudaMemcpyToSymbol d_mwjfdensqt2 %s\n", cudaGetErrorString (err));
    }

    err = cudaMemcpyToSymbol (d_grav, &grav, sizeof (double), 0, cudaMemcpyHostToDevice);	//= grav
    if (err != cudaSuccess) {
      fprintf (stderr, "Error doing cudaMemcpyToSymbol d_grav %s\n", cudaGetErrorString (err));
    }

    double p;

    //initialize all constant arrays to be stored in constant memory on the GPU
    double h_mwjfnums0t0[KM];
    double h_mwjfnums0t2[KM];
    double h_mwjfnums1t0[KM];
    double h_mwjfdens0t0[KM];
    double h_mwjfdens0t1[KM];
    double h_mwjfdens0t3[KM];

    int k;
    for (k = 0; k < KM; k++) {
      p = 10.0 * pressz[k];

      // first calculate numerator of MWJF density [P_1(S,T,p)]
      h_mwjfnums0t0[k] = mwjfnp0s0t0 + p * (mwjfnp1s0t0 + p * mwjfnp2s0t0);
      h_mwjfnums0t2[k] = mwjfnp0s0t2 + p * (mwjfnp1s0t2 + p * mwjfnp2s0t2);
      h_mwjfnums1t0[k] = mwjfnp0s1t0 + p * mwjfnp1s1t0;

      // now calculate denominator of MWJF density [P_2(S,T,p)]
      h_mwjfdens0t0[k] = mwjfdp0s0t0 + p * mwjfdp1s0t0;
      h_mwjfdens0t1[k] = mwjfdp0s0t1 + (p * p * p) * mwjfdp3s0t1;	//used to be p**3 in FORTRAN
      h_mwjfdens0t3[k] = mwjfdp0s0t3 + (p * p) * mwjfdp2s0t3;	//used to be p**2 in FORTRAN
    }

    err = cudaMemcpyToSymbol (d_mwjfnums0t0, h_mwjfnums0t0, KM * sizeof (double), 0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      fprintf (stderr, "Error doing cudaMemcpyToSymbol d_mwjfnums0t0 %s\n", cudaGetErrorString (err));
    }
    err = cudaMemcpyToSymbol (d_mwjfnums0t2, h_mwjfnums0t2, KM * sizeof (double), 0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      fprintf (stderr, "Error doing cudaMemcpyToSymbol d_mwjfnums0t2 %s\n", cudaGetErrorString (err));
    }
    err = cudaMemcpyToSymbol (d_mwjfnums1t0, h_mwjfnums1t0, KM * sizeof (double), 0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      fprintf (stderr, "Error doing cudaMemcpyToSymbol d_mwjfnums1t0 %s\n", cudaGetErrorString (err));
    }

    cudaMemcpyToSymbol (d_mwjfdens0t0, h_mwjfdens0t0, KM * sizeof (double), 0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      fprintf (stderr, "Error doing cudaMemcpyToSymbol d_mwjfdens0t0 %s\n", cudaGetErrorString (err));
    }
    cudaMemcpyToSymbol (d_mwjfdens0t1, h_mwjfdens0t1, KM * sizeof (double), 0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      fprintf (stderr, "Error doing cudaMemcpyToSymbol d_mwjfdens0t1 %s\n", cudaGetErrorString (err));
    }
    cudaMemcpyToSymbol (d_mwjfdens0t3, h_mwjfdens0t3, KM * sizeof (double), 0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      fprintf (stderr, "Error doing cudaMemcpyToSymbol d_mwjfdens0t3 %s\n", cudaGetErrorString (err));
    }

    //error checking
    cudaDeviceSynchronize ();
    CUDA_CHECK_ERROR ("After cudaMemcpyToSymbols");

  }
}
