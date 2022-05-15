/*  imfft.cc

    Mark Jenkinson, FMRIB Image Analysis Group

    Copyright (C) 2001 University of Oxford  */

/*  CCOPYRIGHT  */


// FFT routines for 3D images


#include "imfft.h"

using namespace std;
using namespace MISCMATHS;
using namespace NEWMAT;


namespace NEWIMAGE {

//////////////////////////////////////////////////////////////////////////////

template <class T>
int ifft(volume<T>& revol, volume<T>& imvol, bool transformz=true)
{
  ColumnVector fvr, fvi, vecr, veci;
  // do the transform in x
  int xoff = revol.minx()-1;
  vecr.ReSize(revol.maxx()-xoff);
  veci.ReSize(revol.maxx()-xoff);
  for (int z=revol.minz(); z<=revol.maxz(); z++) {
    for (int y=revol.miny(); y<=revol.maxy(); y++) {
      for (int x=revol.minx(); x<=revol.maxx(); x++) {
	vecr(x-xoff) = revol(x,y,z);
	veci(x-xoff) = imvol(x,y,z);
      }
      FFTI(vecr,veci,fvr,fvi);
      for (int x=revol.minx(); x<=revol.maxx(); x++) {
	revol(x,y,z) = fvr(x-xoff);
	imvol(x,y,z) = fvi(x-xoff);
      }
    }
  }
  // do the transform in y
  int yoff = revol.miny()-1;
  vecr.ReSize(revol.maxy()-yoff);
  veci.ReSize(revol.maxy()-yoff);
  for (int z=revol.minz(); z<=revol.maxz(); z++) {
    for (int x=revol.minx(); x<=revol.maxx(); x++) {
      for (int y=revol.miny(); y<=revol.maxy(); y++) {
	vecr(y-yoff) = revol(x,y,z);
	veci(y-yoff) = imvol(x,y,z);
      }
      FFTI(vecr,veci,fvr,fvi);
      for (int y=revol.miny(); y<=revol.maxy(); y++) {
	revol(x,y,z) = fvr(y-yoff);
	imvol(x,y,z) = fvi(y-yoff);
      }
    }
  }

  if (transformz && ((revol.maxz()-revol.minz())>0)) {
    // do the transform in z
    int zoff = revol.minz()-1;
    vecr.ReSize(revol.maxz()-zoff);
    veci.ReSize(revol.maxz()-zoff);
    for (int x=revol.minx(); x<=revol.maxx(); x++) {
      for (int y=revol.miny(); y<=revol.maxy(); y++) {
	for (int z=revol.minz(); z<=revol.maxz(); z++) {
	  vecr(z-zoff) = revol(x,y,z);
	  veci(z-zoff) = imvol(x,y,z);
	}
	FFTI(vecr,veci,fvr,fvi);
	for (int z=revol.minz(); z<=revol.maxz(); z++) {
	  revol(x,y,z) = fvr(z-zoff);
	  imvol(x,y,z) = fvi(z-zoff);
	}
      }
    }
  }
  return 0;
}

///////////////////////////////////////////////////////////////////////////////

int ifft3(complexvolume& vol)
{
  return ifft(vol.re(),vol.im(),true);
}

int ifft2(complexvolume& vol)
{
  return ifft(vol.re(),vol.im(),false);
}

int ifft3(const complexvolume& vin, complexvolume& vout)
{
  // set up dimensions
  vout = vin;
  return ifft(vout.re(),vout.im(),true);
}

int ifft2(const complexvolume& vin, complexvolume& vout)
{
  // set up dimensions
  vout = vin;
  return ifft(vout.re(),vout.im(),false);
}

int ifft2(volume<float>& realvol, volume<float>& imagvol)
{  return ifft(realvol,imagvol,false); }

int ifft2(const volume<float>& realvin, const volume<float>& imagvin,
	  volume<float>& realvout, volume<float>& imagvout)
{
  // set up dimensions
  realvout = realvin;
  imagvout = imagvin;
  return ifft(realvout,imagvout,false);
}

int ifft3(volume<float>& realvol, volume<float>& imagvol)
{  return ifft(realvol,imagvol,true); }

int ifft3(const volume<float>& realvin, const volume<float>& imagvin,
	  volume<float>& realvout, volume<float>& imagvout)
{
  // set up dimensions
  realvout = realvin;
  imagvout = imagvin;
  return ifft(realvout,imagvout,true);
}

int ifft2(volume<double>& realvol, volume<double>& imagvol)
{  return ifft(realvol,imagvol,false); }

int ifft2(const volume<double>& realvin, const volume<double>& imagvin,
	  volume<double>& realvout, volume<double>& imagvout)
{
  // set up dimensions
  realvout = realvin;
  imagvout = imagvin;
  return ifft(realvout,imagvout,false);
}

int ifft3(volume<double>& realvol, volume<double>& imagvol)
{  return ifft(realvol,imagvol,true); }

int ifft3(const volume<double>& realvin, const volume<double>& imagvin,
	  volume<double>& realvout, volume<double>& imagvout)
{
  // set up dimensions
  realvout = realvin;
  imagvout = imagvin;
  return ifft(realvout,imagvout,true);
}


///////////////////////////////////////////////////////////////////////////////

template <class T>
int fft(volume<T>& revol, volume<T>& imvol, bool transformz=true)
{
  ColumnVector fvr, fvi, vecr, veci;
  // do the transform in x
  int xoff = revol.minx()-1;
  vecr.ReSize(revol.maxx()-xoff);
  veci.ReSize(revol.maxx()-xoff);
  for (int z=revol.minz(); z<=revol.maxz(); z++) {
    for (int y=revol.miny(); y<=revol.maxy(); y++) {
      for (int x=revol.minx(); x<=revol.maxx(); x++) {
	vecr(x-xoff) = revol(x,y,z);
	veci(x-xoff) = imvol(x,y,z);
      }
      FFT(vecr,veci,fvr,fvi);
      for (int x=revol.minx(); x<=revol.maxx(); x++) {
	revol(x,y,z) = fvr(x-xoff);
	imvol(x,y,z) = fvi(x-xoff);
      }
    }
  }
  // do the transform in y
  int yoff = revol.miny()-1;
  vecr.ReSize(revol.maxy()-yoff);
  veci.ReSize(revol.maxy()-yoff);
  for (int z=revol.minz(); z<=revol.maxz(); z++) {
    for (int x=revol.minx(); x<=revol.maxx(); x++) {
      for (int y=revol.miny(); y<=revol.maxy(); y++) {
	vecr(y-yoff) = revol(x,y,z);
	veci(y-yoff) = imvol(x,y,z);
      }
      FFT(vecr,veci,fvr,fvi);
      for (int y=revol.miny(); y<=revol.maxy(); y++) {
	revol(x,y,z) = fvr(y-yoff);
	imvol(x,y,z) = fvi(y-yoff);
      }
    }
  }

  if (transformz && ((revol.maxz()-revol.minz())>0)) {
    // do the transform in z
    int zoff = revol.minz()-1;
    vecr.ReSize(revol.maxz()-zoff);
    veci.ReSize(revol.maxz()-zoff);
    for (int x=revol.minx(); x<=revol.maxx(); x++) {
      for (int y=revol.miny(); y<=revol.maxy(); y++) {
	for (int z=revol.minz(); z<=revol.maxz(); z++) {
	  vecr(z-zoff) = revol(x,y,z);
	  veci(z-zoff) = imvol(x,y,z);
	}
	FFT(vecr,veci,fvr,fvi);
	for (int z=revol.minz(); z<=revol.maxz(); z++) {
	  revol(x,y,z) = fvr(z-zoff);
	  imvol(x,y,z) = fvi(z-zoff);
	}
      }
    }
  }
  return 0;
}

//////////////////////////////////////////////////////////////////////////////

int fft3(complexvolume& vol)
{
  return fft(vol.re(),vol.im(),true);
}

int fft2(complexvolume& vol)
{
  return fft(vol.re(),vol.im(),false);
}

int fft3(const complexvolume& vin, complexvolume& vout)
{
  // set up dimensions
  vout = vin;
  return fft(vout.re(),vout.im(),true);
}

int fft2(const complexvolume& vin, complexvolume& vout)
{
  // set up dimensions
  vout = vin;
  return fft(vout.re(),vout.im(),false);
}

int fft2(volume<float>& realvol, volume<float>& imagvol)
{  return fft(realvol,imagvol,false); }

int fft2(const volume<float>& realvin, const volume<float>& imagvin,
	 volume<float>& realvout, volume<float>& imagvout)
{
  // set up dimensions
  realvout = realvin;
  imagvout = imagvin;
  return fft(realvout,imagvout,false);
}

int fft3(volume<float>& realvol, volume<float>& imagvol)
{  return fft(realvol,imagvol,true); }

int fft3(const volume<float>& realvin, const volume<float>& imagvin,
	 volume<float>& realvout, volume<float>& imagvout)
{
  // set up dimensions
  realvout = realvin;
  imagvout = imagvin;
  return fft(realvout,imagvout,true);
}

int fft2(volume<double>& realvol, volume<double>& imagvol)
{  return fft(realvol,imagvol,false); }

int fft2(const volume<double>& realvin, const volume<double>& imagvin,
	 volume<double>& realvout, volume<double>& imagvout)
{
  // set up dimensions
  realvout = realvin;
  imagvout = imagvin;
  return fft(realvout,imagvout,false);
}

int fft3(volume<double>& realvol, volume<double>& imagvol)
{  return fft(realvol,imagvol,true); }

int fft3(const volume<double>& realvin, const volume<double>& imagvin,
	 volume<double>& realvout, volume<double>& imagvout)
{
  // set up dimensions
  realvout = realvin;
  imagvout = imagvin;
  return fft(realvout,imagvout,true);
}

//////////////////////////////////////////////////////////////////////////////

template <class T>
void fftshift(volume<T>& vol, bool transformz) {
  if (transformz) {
    cerr << "WARNING:: fftshift not implemented in 3D - doing 2D instead"<<endl;
  }
  // does the fftshift for each 2D (z) plane separately
  volume<T> volb;
  volb = vol;
  int Na, Nb, mida, midb;
  Na = vol.xsize();
  Nb = vol.ysize();
  mida = (Na+1)/2;  // effectively a ceil()
  midb = (Nb+1)/2;  // effectively a ceil()

  for (int z=vol.minz(); z<=vol.maxz(); z++) {

    for (int a=0; a<Na; a++) {
      for (int b=midb; b<=Nb-1; b++) {
	vol(a,b-midb,z) = volb(a,b,z);
      }
      for (int b=0; b<=midb-1; b++) {
	vol(a,b+Nb-midb,z) = volb(a,b,z);
      }
    }

    volb = vol;

    for (int b=0; b<Nb; b++) {
      for (int a=mida; a<=Na-1; a++) {
	vol(a-mida,b,z) = volb(a,b,z);
      }
      for (int a=0; a<=mida-1; a++) {
	vol(a+Na-mida,b,z) = volb(a,b,z);
      }
    }

  }
}


void fftshift(complexvolume& vol) {
  fftshift(vol.re(),false);
  fftshift(vol.im(),false);
}

void fftshift(volume<double>& vol) {
  return fftshift(vol,false);
}

void fftshift(volume<float>& vol) {
  return fftshift(vol,false);
}

}
