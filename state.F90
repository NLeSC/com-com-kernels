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


! This program sets up everything to call the original implementation
! of the state kernel, which is why this is in Fortran 90.
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

program main
use iso_c_binding
use state_mod

#include "timer.fh"

!first is the interface block to C functions
interface

subroutine my_cudamallochost(hostptr, size) bind (c)
  use iso_c_binding
  type(C_PTR), intent(out) :: hostptr
  integer (c_int), intent(in) :: size
end subroutine my_cudaMallocHost

subroutine cuda_init() bind (c)
  use iso_c_binding
end subroutine cuda_init

subroutine state_mwjf(SALT, TEMP, DRHODT, DRHODS, RHOFULL, pressz) bind ( c )
        use iso_c_binding
	real (c_double), dimension(NX_BLOCK,NY_BLOCK,KM) :: &
	    RHOFULL, &! full density of water
	    DRHODT,  &! derivative of density with respect to temperature
	    DRHODS,  &! derivative of density with respect to salinity
	    TEMP,    &! temperature at level k
	    SALT      ! salinity    at level k
	real (c_double), dimension(KM) :: &
	      pressz              ! ref pressure (bars) at each level
end subroutine state_mwjf

end interface


!data declarations

real (c_double), pointer :: & 
    RHOFULL(:,:,:), &! full density of water
    DRHODT(:,:,:),  &! derivative of density with respect to temperature
    DRHODS(:,:,:)    ! derivative of density with respect to salinity

real (c_double), pointer :: &
    TEMP(:,:,:),             &! temperature
    SALT(:,:,:)               ! salinity   

integer (c_int) :: i,j,k
real (c_float) :: time
type(c_ptr) :: cptr

call cuda_init();

! The following allocations are performed in this precise manner to ensure
! that the memory is pinned and allocated in a way that enables asynchronous
! memory copies within the CUDA runtime, resulting in better performance and
! enabling overlap between communication and computation. 
!
! my_cudaMallocHost is simply a C function that calls cudaMallocHost()
! c_f_pointer is a language extension that allows a C pointer to be converted
! into a Fortran pointer.

!inputs
call my_cudaMallocHost(cptr, NX_BLOCK*NY_BLOCK*KM);
call c_f_pointer(cptr, TEMP, (/ NX_BLOCK, NY_BLOCK, KM /) );

call my_cudaMallocHost(cptr, NX_BLOCK*NY_BLOCK*KM);
call c_f_pointer(cptr, SALT, (/ NX_BLOCK, NY_BLOCK, KM /) )

!outputs
call my_cudaMallocHost(cptr, NX_BLOCK*NY_BLOCK*KM);
call c_f_pointer(cptr, RHOFULL, (/ NX_BLOCK, NY_BLOCK, KM /) )

call my_cudaMallocHost(cptr, NX_BLOCK*NY_BLOCK*KM);
call c_f_pointer(cptr, DRHODT, (/ NX_BLOCK, NY_BLOCK, KM /) )

call my_cudaMallocHost(cptr, NX_BLOCK*NY_BLOCK*KM);
call c_f_pointer(cptr, DRHODS, (/ NX_BLOCK, NY_BLOCK, KM /) )

! fill the input arrays with random numbers
! zero the output arrays
do i=1,NX_BLOCK
  do j=1,NY_BLOCK
    do k=1,KM
      SALT(i,j,k)  = rand(0)
      TEMP(i,j,k)  = (rand(0)*1000000.0)/1000.0
      DRHODT(i,j,k)  = 0.0
      DRHODS(i,j,k)  = 0.0
      RHOFULL(i,j,k) = 0.0
    enddo
  enddo
enddo

!timing the Fortran 90 version
time = 0.0
call start_timer()

do k=1, KM

  ! state with three outputs:
  ! call state(k, k, TEMP(:,:,k), SALT(:,:,k), RHOFULL=RHOFULL(:,:,k), DRHODT=DRHODT(:,:,k), DRHODS=DRHODS(:,:,k))

  ! state with one output:
  call state(k, k, TEMP(:,:,k), SALT(:,:,k), RHOFULL=RHOFULL(:,:,k))

enddo

call stop_timer(time)

write(*,*) 'Fortran took: ', time

! the next function is the entry point into the C part of the program
! execution continues in state.cu where the different implementations
! are benchmarked
call state_mwjf(SALT, TEMP, DRHODT, DRHODS, RHOFULL, pressz)

end




