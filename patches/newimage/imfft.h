/*  imfft.h

    Mark Jenkinson, FMRIB Image Analysis Group

    Copyright (C) 2001 University of Oxford  */

/*  CCOPYRIGHT  */


// FFT routines for 3D images
#if !defined(__imfft_h)
#define __imfft_h

#include <string>
#include <iostream>
#include <fstream>
#include <unistd.h>

#include "armawrap/newmatap.h"
#include "armawrap/newmatio.h"
#include "newimage.h"
#include "complexvolume.h"

#define _GNU_SOURCE 1
#define POSIX_SOURCE 1

namespace NEWIMAGE {

////////////////////////////////////////////////////////////////////////////

int ifft2(complexvolume& vol);
int ifft2(const complexvolume& vin, complexvolume& vout);

int fft2(complexvolume& vol);
int fft2(const complexvolume& vin, complexvolume& vout);

int ifft3(complexvolume& vol);
int ifft3(const complexvolume& vin, complexvolume& vout);

int fft3(complexvolume& vol);
int fft3(const complexvolume& vin, complexvolume& vout);

void fftshift(complexvolume& vol);

//------------------------------------------------------------------------//

int ifft2(volume<float>& realvol, volume<float>& imagvol);
int ifft2(const volume<float>& realvin, const volume<float>& imagvin,
	  volume<float>& realvout, volume<float>& imagvout);

int fft2(volume<float>& realvol, volume<float>& imagvol);
int fft2(const volume<float>& realvin, const volume<float>& imagvin,
	 volume<float>& realvout, volume<float>& imagvout);

int ifft3(volume<float>& realvol, volume<float>& imagvol);
int ifft3(const volume<float>& realvin, const volume<float>& imagvin,
	  volume<float>& realvout, volume<float>& imagvout);

int fft3(volume<float>& realvol, volume<float>& imagvol);
int fft3(const volume<float>& realvin, const volume<float>& imagvin,
	 volume<float>& realvout, volume<float>& imagvout);

void fftshift(volume<float>& vol);

//------------------------------------------------------------------------//

int ifft2(volume<double>& realvol, volume<double>& imagvol);
int ifft2(const volume<double>& realvin, const volume<double>& imagvin,
	  volume<double>& realvout, volume<double>& imagvout);

int fft2(volume<double>& realvol, volume<double>& imagvol);
int fft2(const volume<double>& realvin, const volume<double>& imagvin,
	 volume<double>& realvout, volume<double>& imagvout);

int ifft3(volume<double>& realvol, volume<double>& imagvol);
int ifft3(const volume<double>& realvin, const volume<double>& imagvin,
	  volume<double>& realvout, volume<double>& imagvout);

int fft3(volume<double>& realvol, volume<double>& imagvol);
int fft3(const volume<double>& realvin, const volume<double>& imagvin,
	 volume<double>& realvout, volume<double>& imagvout);

void fftshift(volume<double>& vol);

////////////////////////////////////////////////////////////////////////////

}

#endif
