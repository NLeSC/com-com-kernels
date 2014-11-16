!
! This file is a stripped down version of state_mod.F90 that is
! included in the Parallel Ocean Program. This file was created
! for the purpose of checking the correctness of the routines
! that have been translated to the GPU.
!
! You can obtain the complete source code from
! https://github.com/NLeSC/eSalsa-POP
!
! For more details about the state algorithm see:
!  Accurate and computationally efficient algorithms for potential temperature and density of seawater
!  T.J. McDougall, D.R. Jackett, D.G. Wright, and R. Feistel
!  In: Journal of Atmospheric and Oceanic Technology, Volume 20, Issue 5, 2003
!
! author Ben van Werkhoven



#include "domain.h"

module state_mod
use iso_c_binding

real (c_double), parameter, dimension(KM), public :: &
     tmin =  -2.0  ,&! limited   on the low  end
     tmax = 999.0  ,&! unlimited on the high end
     smin =   0.0  ,&! limited   on the low  end
     smax = 0.999  ! unlimited on the high end

real (c_double), parameter, dimension(KM), public :: &
      pressz = (/ 0.49742665690022175_c_double, 1.4996943388141275_c_double, 2.5198349974506398_c_double, 3.5681874348325011_c_double, 4.6563131421470700_c_double, &
5.7975076357076150_c_double, 7.0074707475945281_c_double, 8.3052109766044691_c_double, 9.7142948214301175_c_double, 11.264610295008854_c_double, 12.994919658051220_c_double, &
14.956625620051417_c_double, 17.219460861774902_c_double, 19.880251522649793_c_double, 23.076620232875573_c_double, 27.008540490962417_c_double, 31.971586653370469_c_double, &
38.404203108579246_c_double, 46.937984588968739_c_double, 58.391325463973445_c_double, 73.555512971246401_c_double, 92.673550914816900_c_double, 115.02456046808886_c_double, &
139.28770333919411_c_double, 164.35919282177287_c_double, 189.69029381044083_c_double, 215.10059181140176_c_double, 240.54785719324397_c_double, 266.02474625986446_c_double, &
291.53029287211439_c_double, 317.06440030587697_c_double, 342.62705836463948_c_double, 368.21826704840208_c_double, 393.83802635716467_c_double, 419.48633629092717_c_double, &
445.16319684968971_c_double, 470.86860803345229_c_double, 496.60256984221479_c_double, 522.36508227597733_c_double, 548.15614533473990_c_double, 550.34523433473990_c_double, &
555.12345673473990_c_double /)

real (c_double), parameter, public :: &
      c0     =    0.0   ,&
      c1     =    1.0   ,&
      c2     =    2.0   ,&
      c3     =    3.0   ,&
      c4     =    4.0   ,&
      c5     =    5.0   ,&
      c8     =    8.0   ,&
      c10    =   10.0   ,&
      c16    =   16.0   ,&
      c1000  = 1000.0   ,&
      c10000 =10000.0   ,&
      c1p5   =    1.5   ,&
      p33    = c1/c3    ,&
      p5     = 0.500    ,&
      p25    = 0.250    ,&
      p125   = 0.125    ,&
      p001   = 0.001    ,&
      eps    = 1.0e-10  ,&
      eps2   = 1.0e-20  ,&
      bignum = 1.0e+30  ,&
      grav   = 980.6


real (c_double), parameter, public ::                     &
      mwjfnp0s0t0 =   9.9984369900000e+2 * p001, &
      mwjfnp0s0t1 =   7.3521284000000e+0 * p001, &
      mwjfnp0s0t2 =  -5.4592821100000e-2 * p001, &
      mwjfnp0s0t3 =   3.9847670400000e-4 * p001, &
      mwjfnp0s1t0 =   2.9693823900000e+0 * p001, &
      mwjfnp0s1t1 =  -7.2326881300000e-3 * p001, &
      mwjfnp0s2t0 =   2.1238234100000e-3 * p001, &
      mwjfnp1s0t0 =   1.0400459100000e-2 * p001, &
      mwjfnp1s0t2 =   1.0397052900000e-7 * p001, &
      mwjfnp1s1t0 =   5.1876188000000e-6 * p001, &
      mwjfnp2s0t0 =  -3.2404182500000e-8 * p001, &
      mwjfnp2s0t2 =  -1.2386936000000e-11* p001

   !*** these constants will be used to construct the denominator

real (c_double), parameter, public ::          &
      mwjfdp0s0t0 =   1.0e+0,         &
      mwjfdp0s0t1 =   7.2860673900000e-3,  &
      mwjfdp0s0t2 =  -4.6083554200000e-5,  &
      mwjfdp0s0t3 =   3.6839057300000e-7,  &
      mwjfdp0s0t4 =   1.8080918600000e-10, &
      mwjfdp0s1t0 =   2.1469170800000e-3,  &
      mwjfdp0s1t1 =  -9.2706248400000e-6,  &
      mwjfdp0s1t3 =  -1.7834364300000e-10, &
      mwjfdp0sqt0 =   4.7653412200000e-6,  &
      mwjfdp0sqt2 =   1.6341073600000e-9,  &
      mwjfdp1s0t0 =   5.3084887500000e-6,  &
      mwjfdp2s0t3 =  -3.0317512800000e-16, &
      mwjfdp3s0t1 =  -1.2793413700000e-17

!*** MWJF numerator coefficients including pressure
real (c_double) ::                                                     &
      mwjfnums0t0, mwjfnums0t1, mwjfnums0t2, mwjfnums0t3,              &
      mwjfnums1t0, mwjfnums1t1, mwjfnums2t0,                           &
      mwjfdens0t0, mwjfdens0t1, mwjfdens0t2, mwjfdens0t3, mwjfdens0t4, &
      mwjfdens1t0, mwjfdens1t1, mwjfdens1t3,                           &
      mwjfdensqt0, mwjfdensqt2

public state

contains 

subroutine state(k, kk, TEMPK, SALTK, RHOOUT, RHOFULL, DRHODT, DRHODS)
  use iso_c_binding

   integer (c_int), intent(in) :: &
      k,                    &! depth level index
      kk                     ! level to which water is adiabatically 
                            ! displaced

   real (c_double), dimension(NX_BLOCK,NY_BLOCK), intent(in) :: & 
      TEMPK,             &! temperature at level k
      SALTK               ! salinity    at level k

! !OUTPUT PARAMETERS:

   real (c_double), dimension(NX_BLOCK,NY_BLOCK), optional, intent(out) :: & 
      RHOOUT,  &! perturbation density of water
      RHOFULL, &! full density of water
      DRHODT,  &! derivative of density with respect to temperature
      DRHODS    ! derivative of density with respect to salinity

  real (c_double), dimension(NX_BLOCK,NY_BLOCK) :: & 
    TQ,SQ,       &! adjusted T,S
    SQR,DENOMK,  &! work arrays
    WORK1, WORK2, WORK3, WORK4

  real (c_double) :: p

      TQ = min(TEMPK,tmax(k))
      TQ = max(TQ,tmin(k))

      SQ = min(SALTK,smax(kk))
      SQ = max(SQ,smin(kk))

!-----------------------------------------------------------------------
!
!  McDougall, Wright, Jackett, and Feistel EOS
!  test value : rho = 1.033213242 for
!  S = 35.0 PSU, theta = 20.0, pressz = 200.0
!
!-----------------------------------------------------------------------

      p   = c10*pressz(kk)

      SQ  = c1000*SQ
      SQR = sqrt(SQ)

      !***
      !*** first calculate numerator of MWJF density [P_1(S,T,p)]
      !***

      mwjfnums0t0 = mwjfnp0s0t0 + p*(mwjfnp1s0t0 + p*mwjfnp2s0t0)
      mwjfnums0t1 = mwjfnp0s0t1 
      mwjfnums0t2 = mwjfnp0s0t2 + p*(mwjfnp1s0t2 + p*mwjfnp2s0t2)
      mwjfnums0t3 = mwjfnp0s0t3
      mwjfnums1t0 = mwjfnp0s1t0 + p*mwjfnp1s1t0
      mwjfnums1t1 = mwjfnp0s1t1
      mwjfnums2t0 = mwjfnp0s2t0

      WORK1 = mwjfnums0t0 + TQ * (mwjfnums0t1 + TQ * (mwjfnums0t2 + &
              mwjfnums0t3 * TQ)) + SQ * (mwjfnums1t0 +              &
              mwjfnums1t1 * TQ + mwjfnums2t0 * SQ)

      !***
      !*** now calculate denominator of MWJF density [P_2(S,T,p)]
      !***

      mwjfdens0t0 = mwjfdp0s0t0 + p*mwjfdp1s0t0
      mwjfdens0t1 = mwjfdp0s0t1 + p**3 * mwjfdp3s0t1
      mwjfdens0t2 = mwjfdp0s0t2
      mwjfdens0t3 = mwjfdp0s0t3 + p**2 * mwjfdp2s0t3
      mwjfdens0t4 = mwjfdp0s0t4
      mwjfdens1t0 = mwjfdp0s1t0
      mwjfdens1t1 = mwjfdp0s1t1
      mwjfdens1t3 = mwjfdp0s1t3
      mwjfdensqt0 = mwjfdp0sqt0
      mwjfdensqt2 = mwjfdp0sqt2

      WORK2 = mwjfdens0t0 + TQ * (mwjfdens0t1 + TQ * (mwjfdens0t2 +    &
           TQ * (mwjfdens0t3 + mwjfdens0t4 * TQ))) +                   &
           SQ * (mwjfdens1t0 + TQ * (mwjfdens1t1 + TQ*TQ*mwjfdens1t3)+ &
           SQR * (mwjfdensqt0 + TQ*TQ*mwjfdensqt2))

      DENOMK = c1/WORK2

      if (present(RHOOUT)) then
        RHOOUT  = WORK1*DENOMK
      endif

      if (present(RHOFULL)) then
        RHOFULL = WORK1*DENOMK
      endif

      if (present(DRHODT)) then
         WORK3 = &! dP_1/dT
                 mwjfnums0t1 + TQ * (c2*mwjfnums0t2 +    &
                 c3*mwjfnums0t3 * TQ) + mwjfnums1t1 * SQ

         WORK4 = &! dP_2/dT
                 mwjfdens0t1 + SQ * mwjfdens1t1 +               &
                 TQ * (c2*(mwjfdens0t2 + SQ*SQR*mwjfdensqt2) +  &
                 TQ * (c3*(mwjfdens0t3 + SQ * mwjfdens1t3) +    &
                 TQ *  c4*mwjfdens0t4))

         DRHODT = (WORK3 - WORK1*DENOMK*WORK4)*DENOMK
      endif

      if (present(DRHODS)) then
         WORK3 = &! dP_1/dS
                 mwjfnums1t0 + mwjfnums1t1 * TQ + c2*mwjfnums2t0 * SQ

         WORK4 = mwjfdens1t0 +   &! dP_2/dS
                 TQ * (mwjfdens1t1 + TQ*TQ*mwjfdens1t3) +   &
                 c1p5*SQR*(mwjfdensqt0 + TQ*TQ*mwjfdensqt2)

         DRHODS = (WORK3 - WORK1*DENOMK*WORK4)*DENOMK * c1000
      endif

end subroutine state


end module state_mod

