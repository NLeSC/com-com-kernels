!
! Copyright 2014 Netherlands eScience Center
!
! Licensed under the Apache License, Version 2.0 (the "License");
! you may not use this file except in compliance with the License.
! You may obtain a copy of the License at
!
!      http://www.apache.org/licenses/LICENSE-2.0
!
! Unless required by applicable law or agreed to in writing, software
! distributed under the License is distributed on an "AS IS" BASIS,
! WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
! See the License for the specific language governing permissions and
! limitations under the License.
!


!
! This program sets up everything to call the original implementation
! of the buoydiff kernel, which is why this is in Fortran 90.
!
! At the bottom of this file the C function is called which measures
! the performance of four different implementations for
! overlapping CPU-GPU communication and GPU computation of the
! state kernel.
!
! @author Ben van Werkhoven <b.vanwerkhoven@esciencecenter.nl>
!
!

#include "domain.h"


module buoydiff_mod
!
! This module is a stripped down version of vmix_kpp.F90 that is
! included in the Parallel Ocean Program. This file was created
! for the purpose of checking the correctness of the routines
! that have been translated to the GPU.
!
! You can obtain the complete source code from
! https://github.com/NLeSC/eSalsa-POP
!
! For more details about the KPP parameterization see:
!  Oceanic vertical mixing: A review and a model with a nonlocal boundary layer parameterization
!  W.G. Large, J.C. McWilliams, and S.C. Doney
!  In: Reviews of Geophysics, Volume 32, Issue 4, 1994
!

use iso_c_binding
use state_mod

contains

 subroutine buoydiff(DBLOC, DBSFC, TRCR, KMT)

! !DESCRIPTION:
!  This routine calculates the buoyancy differences at model levels.
!
! !REVISION HISTORY:
!  same as module

! !INPUT PARAMETERS:

   real (c_double), dimension(NX_BLOCK,NY_BLOCK,KM,NT), intent(in) :: &
      TRCR                ! tracers at current time

!added, orignally in grid.F90 in POP
   integer (c_int), dimension(NX_BLOCK,NY_BLOCK), intent(in) :: &
        KMT             ! k index of deepest grid cell on T grid

!removed
!   type (block), intent(in) :: &
!      this_block          ! block information for current block

! !OUTPUT PARAMETERS:

   real (c_double), dimension(NX_BLOCK,NY_BLOCK,KM), intent(out) :: &
      DBLOC,         &! buoyancy difference between adjacent levels
      DBSFC           ! buoyancy difference between level and surface

!EOP
!BOC
!-----------------------------------------------------------------------
!
!  local variables
!
!-----------------------------------------------------------------------

   integer (c_int) :: &
      k,                 &! vertical level index
      i,j,               &! horizontal indices
      kprev, klvl, ktmp  ! indices for 2-level TEMPK array

   real (c_double), dimension(NX_BLOCK,NY_BLOCK) :: &
      RHO1,              &! density of sfc t,s displaced to k
      RHOKM,             &! density of t(k-1),s(k-1) displaced to k
      RHOK,              &! density at level k
      TEMPSFC           ! adjusted temperature at surface
!      TALPHA,            &! temperature expansion coefficient
!      SBETA               ! salinity    expansion coefficient

   real (c_double), dimension(NX_BLOCK,NY_BLOCK,2) :: &
      TEMPK               ! temp adjusted for freeze at levels k,k-1

!-----------------------------------------------------------------------
!
!  calculate density and buoyancy differences at surface
!  
!-----------------------------------------------------------------------

   TEMPSFC = merge(-c2,TRCR(:,:,1,1),TRCR(:,:,1,1) < -c2)

   klvl  = 2
   kprev = 1

   TEMPK(:,:,kprev) = TEMPSFC
   DBSFC(:,:,1) = c0

!-----------------------------------------------------------------------
!   
!  calculate DBLOC and DBSFC for all other levels
!
!-----------------------------------------------------------------------

   do k = 2,KM

      TEMPK(:,:,klvl) = merge(-c2,TRCR(:,:,k,1),TRCR(:,:,k,1) < -c2)

      call state(k, k, TEMPSFC,          TRCR(:,:,1  ,2), &
                       RHOFULL=RHO1)
      call state(k, k, TEMPK(:,:,kprev), TRCR(:,:,k-1,2), &
                       RHOFULL=RHOKM)
      call state(k, k, TEMPK(:,:,klvl),  TRCR(:,:,k  ,2), &
                       RHOFULL=RHOK)

      do j=1,NY_BLOCK
      do i=1,NX_BLOCK
         if (RHOK(i,j) /= c0) then
            DBSFC(i,j,k)   = grav*(c1 - RHO1 (i,j)/RHOK(i,j))
            DBLOC(i,j,k-1) = grav*(c1 - RHOKM(i,j)/RHOK(i,j))
         else
            DBSFC(i,j,k)   = c0
            DBLOC(i,j,k-1) = c0
         endif

         if (k-1 >= KMT(i,j)) DBLOC(i,j,k-1) = c0
      end do
      end do

      ktmp  = klvl
      klvl  = kprev
      kprev = ktmp

   enddo

   DBLOC(:,:,KM) = c0

!-----------------------------------------------------------------------
!EOC

 end subroutine buoydiff


end module buoydiff_mod






program main
use iso_c_binding
use state_mod
use buoydiff_mod

!implicit none

#include "timer.fh"

!interface block to C functionality
interface

subroutine my_cudamallochost(hostptr, size, type) bind (c)
  use iso_c_binding
  type(C_PTR), intent(out) :: hostptr
  integer (c_int), intent(in) :: size, type
end subroutine my_cudaMallocHost

subroutine cuda_init() bind (c)
  use iso_c_binding
end subroutine cuda_init

subroutine buoydiff_entry(DBLOC, DBSFC, TRCR, KMT, pressz) bind (c)
  use iso_c_binding
  real (c_double), dimension(NX_BLOCK,NY_BLOCK,KM,NT) :: &
    TRCR                ! tracers at current time
  integer (c_int), dimension(NX_BLOCK,NY_BLOCK) :: &
    KMT             ! k index of deepest grid cell on T grid
  real (c_double), dimension(NX_BLOCK,NY_BLOCK,KM) :: &
    DBLOC,         &! buoyancy difference between adjacent levels
    DBSFC           ! buoyancy difference between level and surface
  real (c_double), dimension(KM) :: &
    pressz              ! ref pressure (bars) at each level
end subroutine buoydiff_entry

end interface

!public vars

  real (c_double), pointer :: &
    TRCR(:,:,:,:)          ! tracers at current time
  integer (c_int), pointer :: &
    KMT(:,:)             ! k index of deepest grid cell on T grid
  real (c_double), pointer :: &
    DBLOC(:,:,:),         &! buoyancy difference between adjacent levels
    DBSFC(:,:,:)           ! buoyancy difference between level and surface


integer (c_int) :: i,j,k
real (c_float) :: time
type(c_ptr) :: cptr

real (c_double) :: temp

 
  allocate ( 	KMT(NX_BLOCK, NY_BLOCK) )






call cuda_init

call my_cudaMallocHost(cptr, NX_BLOCK*NY_BLOCK*KM*NT, 8);
call c_f_pointer(cptr, TRCR, (/ NX_BLOCK, NY_BLOCK, KM, NT /) );

!call my_cudaMallocHost(cptr, NX_BLOCK*NY_BLOCK, 4);
!call c_f_pointer(cptr, KMT, (/ NX_BLOCK, NY_BLOCK /) )

call my_cudaMallocHost(cptr, NX_BLOCK*NY_BLOCK*KM, 8);
call c_f_pointer(cptr, DBLOC, (/ NX_BLOCK, NY_BLOCK, KM /) );

call my_cudaMallocHost(cptr, NX_BLOCK*NY_BLOCK*KM, 8);
call c_f_pointer(cptr, DBSFC, (/ NX_BLOCK, NY_BLOCK, KM /) );


!write(*,*) 'Filling arrays with random values.'

do i=1,NX_BLOCK
  do j=1,NY_BLOCK
    KMT(i,j) = 35 + MOD(i*j,7);
    do k=1,KM
      call random_number(temp)
      TRCR(i,j,k,1)  = -2.0+temp*32.0
      call random_number(temp)
      TRCR(i,j,k,2)  = -1.0+temp*2.2
    enddo
  enddo
enddo



!write(*,*) 'In FORTRAN:'

time = 0.0
call start_timer()

call buoydiff(DBLOC, DBSFC, TRCR, KMT)

call stop_timer(time)

write(*,*) 'Fortran took: ', time

!write(*,*) 'Going to C:'

!call to CUDA/C function
call buoydiff_entry(DBLOC, DBSFC, TRCR, KMT, pressz)

!write(*,*) 'End FORTRAN'


end

