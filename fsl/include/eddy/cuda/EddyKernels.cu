#include <stdio.h>
#include <cuda.h>
#include <string>
#include <iostream>
#include <math_constants.h>
#include <math_functions.h>
#pragma push
#pragma diag_suppress = code_is_unreachable // Supress warnings from armawrap
#include "EddyHelperClasses.h"
#pragma pop
#include "EddyKernels.h"

namespace EddyKernels {

__global__ void linear_ec_field(float *ec_field, 
				int xsz, int ysz, int zsz, 
				float xvxs, float yvxs, float zvxs,
				const float *ep, int npar,
				int max_id)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= max_id) { return; }
  int k = id / (xsz*ysz);
  int j = (id / xsz) % ysz;
  int i = id % xsz;

  ec_field[id] = ep[0]*xvxs*(i-(xsz-1)/2.0) + ep[1]*yvxs*(j-(ysz-1)/2.0) + ep[2]*zvxs*(k-(zsz-1)/2.0) + ep[3];
}

__global__ void quadratic_ec_field(float *ec_field, 
				   int xsz, int ysz, int zsz, 
				   float xvxs, float yvxs, float zvxs,
				   const float *ep, int npar,
				   int max_id)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= max_id) { return; }
  int k = id / (xsz*ysz);
  int j = (id / xsz) % ysz;
  int i = id % xsz;

  float x = xvxs*(i-(xsz-1)/2.0); 
  float y = yvxs*(j-(ysz-1)/2.0);
  float z = zvxs*(k-(zsz-1)/2.0);
  ec_field[id] = ep[0]*x + ep[1]*y + ep[2]*z + ep[3]*x*x + ep[4]*y*y + ep[5]*z*z + ep[6]*x*y + ep[7]*x*z + ep[8]*y*z + ep[9];
}

__global__ void cubic_ec_field(float *ec_field, 
			       int xsz, int ysz, int zsz, 
			       float xvxs, float yvxs, float zvxs,
			       const float *ep, int npar,
			       int max_id)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= max_id) { return; }
  int k = id / (xsz*ysz);
  int j = (id / xsz) % ysz;
  int i = id % xsz;

  float x = xvxs*(i-(xsz-1)/2.0); 
  float y = yvxs*(j-(ysz-1)/2.0);
  float z = zvxs*(k-(zsz-1)/2.0);
  float xx = x*x; float yy = y*y; float zz = z*z;
  ec_field[id] = ep[19] + ep[0]*x + ep[1]*y + ep[2]*z + ep[3]*xx + ep[4]*yy + ep[5]*zz + ep[6]*x*y + ep[7]*x*z + ep[8]*y*z;
  ec_field[id] += ep[9]*xx*x + ep[10]*yy*y + ep[11]*zz*z + ep[12]*xx*y + ep[13]*xx*z + ep[14]*x*y*z + ep[15]*x*yy + ep[16]*yy*z + ep[17]*x*zz + ep[18]*y*zz;
}

__global__ void make_coordinates(int xsz, int ysz, int zsz, 
				 float *xcoord, float *ycoord, float *zcoord, 
				 int max_id)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= max_id) { return; }
  zcoord[id] = id / (xsz*ysz);
  ycoord[id] = (id / xsz) % ysz;
  xcoord[id] = id % xsz;
}

__global__ void affine_transform_coordinates(int   xsz, int   ysz, int   zsz,
					     float A11, float A12, float A13, float A14,
					     float A21, float A22, float A23, float A24,
					     float A31, float A32, float A33, float A34,
					     float *xcoord, float *ycoord, float *zcoord, 
					     bool tec, int   max_id)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= max_id) { return; }

  float z = id / (xsz*ysz);
  float y = (id / xsz) % ysz;
  float x = id % xsz;
  if (tec) { x=xcoord[id]; y=ycoord[id]; z=zcoord[id]; }

  xcoord[id] = A11*x + A12*y + A13*z + A14;
  ycoord[id] = A21*x + A22*y + A23*z + A24;
  zcoord[id] = A31*x + A32*y + A33*z + A34;
}

__global__ void slice_wise_affine_transform_coordinates(int   xsz, int   ysz, int   zsz, const float *A,
							float *xcoord, float *ycoord, float *zcoord, 
							bool tec, int   max_id)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= max_id) { return; }

  int sl = id / (xsz*ysz);
  float z = sl;
  float y = (id / xsz) % ysz;
  float x = id % xsz;
  if (tec) { x=xcoord[id]; y=ycoord[id]; z=zcoord[id]; }

  int offs = 12*sl;
  xcoord[id] = A[offs]*x + A[offs+1]*y + A[offs+2]*z + A[offs+3];
  ycoord[id] = A[offs+4]*x + A[offs+5]*y + A[offs+6]*z + A[offs+7];
  zcoord[id] = A[offs+8]*x + A[offs+9]*y + A[offs+10]*z + A[offs+11];
}

__global__ void general_transform_coordinates(int   xsz, int   ysz, int   zsz,
					      const float *xfield, const float *yfield, const float *zfield, 
					      float A11, float A12, float A13, float A14,
					      float A21, float A22, float A23, float A24,
					      float A31, float A32, float A33, float A34,
					      float M11, float M12, float M13, float M14, 
					      float M21, float M22, float M23, float M24, 
					      float M31, float M32, float M33, float M34, 
					      float *xcoord, float *ycoord, float *zcoord, 
					      bool tec, int   max_id)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= max_id) { return; }

  float z = id / (xsz*ysz);
  float y = (id / xsz) % ysz;
  float x = id % xsz;
  if (tec) { x=xcoord[id]; y=ycoord[id]; z=zcoord[id]; }

  float xx = A11*x + A12*y + A13*z + A14;
  float yy = A21*x + A22*y + A23*z + A24;
  float zz = A31*x + A32*y + A33*z + A34;

  x = xx + xfield[id];  
  y = yy + yfield[id];  
  z = zz + zfield[id];  

  xcoord[id] = M11*x + M12*y + M13*z + M14;
  ycoord[id] = M21*x + M22*y + M23*z + M24;
  zcoord[id] = M31*x + M32*y + M33*z + M34;		  
}

__global__ void slice_wise_general_transform_coordinates(int   xsz, int   ysz, int   zsz,
							 const float *xfield, const float *yfield, 
							 const float *zfield, const float *A,
							 const float *M, float *xcoord, 
							 float *ycoord, float *zcoord, 
							 bool tec, int   max_id)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= max_id) { return; }

  int sl = id / (xsz*ysz);
  float z = sl;
  float y = (id / xsz) % ysz;
  float x = id % xsz;
  if (tec) { x=xcoord[id]; y=ycoord[id]; z=zcoord[id]; }

  int offs = 12*sl;

  float xx = A[offs]*x + A[offs+1]*y + A[offs+2]*z + A[offs+3];
  float yy = A[offs+4]*x + A[offs+5]*y + A[offs+6]*z + A[offs+7];
  float zz = A[offs+8]*x + A[offs+9]*y + A[offs+10]*z + A[offs+11];

  x = xx + xfield[id];  
  y = yy + yfield[id];  
  z = zz + zfield[id];  

  xcoord[id] = M[offs]*x + M[offs+1]*y + M[offs+2]*z + M[offs+3];
  ycoord[id] = M[offs+4]*x + M[offs+5]*y + M[offs+6]*z + M[offs+7];
  zcoord[id] = M[offs+8]*x + M[offs+9]*y + M[offs+10]*z + M[offs+11];
}

__global__ void slice_to_vol_xyz_coordinates(int   xsz, int   ysz, int   zsz,
					     const float *xfield, const float *yfield, 
					     const float *zfield, const float *M1,
					     const float *R, const float *M2, 
					     float *xcoord, float *ycoord, float *zcoord, 
					     float *zvolume, bool tec, int max_id)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= max_id) { return; }

  int sl = id / (xsz*ysz);
  float z = sl;
  float y = (id / xsz) % ysz;
  float x = id % xsz;
  if (tec) { x=xcoord[id]; y=ycoord[id]; z=zcoord[id]; }
  float xf, yf, zf;
  if (xfield) { xf = xfield[id]; yf = yfield[id]; zf = zfield[id]; }
  else { xf = 0.0; yf = 0.0; zf = 0.0; }

  float xx = M1[0]*x + M1[3];
  float yy = M1[5]*y + M1[7];
  float zz = M1[10]*z + M1[11];

  int offs = 12*sl;
  float zv = ( - R[offs+8]*xx - R[offs+9]*yy + zz - R[offs+11] - zf ) / R[offs+10];
  zvolume[id] = zv;

  float xxx = R[offs+0]*xx + R[offs+1]*yy + R[offs+2]*zv + R[offs+3] + xf;
  float yyy = R[offs+4]*xx + R[offs+5]*yy + R[offs+6]*zv + R[offs+7] + yf;
  
  xcoord[id] = M2[0]*xxx + M2[3];
  ycoord[id] = M2[5]*yyy + M2[7];
  zcoord[id] = sl;
  zvolume[id] = M2[10]*zv + M2[11];
}

__global__ void slice_to_vol_z_coordinates(int   xsz, int   ysz, int   zsz,
					   const float *xfield, const float *yfield, 
					   const float *zfield, const float *M1,
					   const float *R, const float *M2, 
					   float *xcoord, float *ycoord, float *zcoord, 
					   bool tec, int max_id)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= max_id) { return; }

  int sl = id / (xsz*ysz);
  float z = sl;
  float y = (id / xsz) % ysz;
  float x = id % xsz;
  if (tec) { x=xcoord[id]; y=ycoord[id]; z=zcoord[id]; }
  float zf;
  if (xfield) zf = zfield[id];
  else zf = 0.0;

  float xx = M1[0]*x + M1[3];
  float yy = M1[5]*y + M1[7];
  float zz = M1[10]*z + M1[11];

  int offs = 12*sl;
  float zv = ( - R[offs+8]*xx - R[offs+9]*yy + zz - R[offs+11] - zf ) / R[offs+10];

  xcoord[id] = x;
  ycoord[id] = y;
  zcoord[id] = M2[10]*zv + M2[11];
}

__global__ void get_mask(int xsz, int ysz, int zsz,
			 int epvx, int epvy, int epvz, 
			 const float *xcoord, const float *ycoord, const float *zcoord, 
			 float *mask, 
			 int max_id)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= max_id) { return; }
  if ((epvx || xcoord[id]>0 && xcoord[id]<xsz) && 
      (epvy || ycoord[id]>0 && ycoord[id]<ysz) &&
      (epvz || zcoord[id]>0 && zcoord[id]<zsz)) mask[id] = 1.0;
  else mask[id] = 0.0;
}

__global__ void make_deriv(int xsz, int ysz, int zsz,
			   const float *xcoord, const float *ycoord, const float *zcoord, 
			   const float *xgrad, const float *ygrad, const float *zgrad,
			   const float *base, const float *jac, const float *basejac,
			   float dstep,
			   float *deriv, 
			   int max_id)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= max_id) { return; }
  deriv[id] = (xcoord[id]*xgrad[id] + ycoord[id]*ygrad[id] + zcoord[id]*zgrad[id] + base[id]*(jac[id] - basejac[id])) / dstep;
  return;
}

// Index-to-index translation for constant boundary condition
__device__ int i2i_c(int i, int n)
{
  if (i<0) return(0);
  else if (i>=n) return(n-1);
  return(i);
}

// Index-to-index translation for periodic boundary condition
__device__ int i2i_p(int i, int n)
{
  if (i<0) return(n-1-((-i-1)%n));
  else return(i%n);
}

// Index-to-index translation for mirror boundary condition
__device__ int i2i_m(int i, int n)
{
  if (i<0) return((-i)%n);
  else if (i>=n) return(n - i%n - 2);
  return(i);
}

#define INDX(i,j) (i)*4+(j)

__global__ void spline_interpolate(int         xsz,
				   int         ysz,
				   int         zsz,
				   const float *spcoef,
				   const float *xcoord, 
				   const float *ycoord, 
				   const float *zcoord, 
				   int         sizeT,
				   int         bc,
				   float       *ima)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= sizeT) { return; }

  float coord[3];
  coord[0]=xcoord[id]; coord[1]=ycoord[id]; coord[2]=zcoord[id];

  if (bc==CONSTANT) {
    if (coord[0]<0) coord[0] = 0; else if (coord[0]>xsz-1) coord[0]=xsz-1;
    if (coord[1]<0) coord[1] = 0; else if (coord[1]>ysz-1) coord[1]=ysz-1;
    if (coord[2]<0) coord[2] = 0; else if (coord[2]>zsz-1) coord[2]=zsz-1;    
  }

  // Get start indicies of spline kernel into coeffcient field
  int sind[3];
  for (unsigned int i=0; i<3; i++) {
    sind[i] = (int)(coord[i]+0.5);
    sind[i] -= (sind[i] < coord[i]) ? 1 : 2;
  }
  
  // Calculate weights for separable kernels
  float wgts[12]; // wgts[0:3] x-dir, wgts[4:7] y-dir, wgts[8:11] z-dir
  for (int i=0; i<3; i++) {
    for (int j=0; j<4; j++) {
      float x = coord[i] - (sind[i] + j);
      float ax = abs(x); // Kernels symmetric
      if (ax < 1) wgts[INDX(i,j)] = 2.0/3.0 + 0.5*ax*ax*(ax-2);                // -1<x<1
      else if (ax < 2) { ax = 2-ax; wgts[INDX(i,j)] = (1.0/6.0)*(ax*ax*ax); }  // -2<x<-1 || 1<x<2
      else wgts[INDX(i,j)] = 0.0;                                              // x<-2 || x>2
    }
  }

  // Calculate weighted sum of coefficient values
  float ws = 0.0;
  for (int k=0; k<4; k++) {
    float wgt1 = wgts[INDX(2,k)];
    int linear1 = (xsz*ysz);
    if (bc==PERIODIC) linear1 *= i2i_p(sind[2]+k,zsz); 
    else if (bc==MIRROR) linear1 *= i2i_m(sind[2]+k,zsz);
    else linear1 *= i2i_c(sind[2]+k,zsz); // defaults to constant
    for (int j=0; j<4; j++) {
      float wgt2 = wgt1 * wgts[INDX(1,j)];
      int linear2 = linear1;
      if (bc==PERIODIC) linear2 += i2i_p(sind[1]+j,ysz) * xsz; 
      else if (bc==MIRROR) linear2 += i2i_m(sind[1]+j,ysz) * xsz;
      else linear2 += i2i_c(sind[1]+j,ysz) * xsz;
      for (int i=0; i<4; i++) {
	if (bc==PERIODIC) ws += spcoef[linear2+i2i_p(sind[0]+i,xsz)]*wgt2*wgts[i];
	else if (bc==MIRROR) ws += spcoef[linear2+i2i_m(sind[0]+i,xsz)]*wgt2*wgts[i];
	else ws += spcoef[linear2+i2i_c(sind[0]+i,xsz)]*wgt2*wgts[i];
      }
    }
  }
  ima[id] = ws;
}

__global__ void spline_interpolate(int         xsz,
				   int         ysz,
				   int         zsz,
				   const float *spcoef,
				   const float *xcoord, 
				   const float *ycoord, 
				   const float *zcoord, 
				   int         sizeT,
				   int         bc,
				   float       *ima,
				   float       *xd,
				   float       *yd,
				   float       *zd)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= sizeT) { return; }

  float coord[3];
  coord[0]=xcoord[id]; coord[1]=ycoord[id]; coord[2]=zcoord[id];

  if (bc==CONSTANT) {
    if (coord[0]<0) coord[0] = 0; else if (coord[0]>xsz-1) coord[0]=xsz-1;
    if (coord[1]<0) coord[1] = 0; else if (coord[1]>ysz-1) coord[1]=ysz-1;
    if (coord[2]<0) coord[2] = 0; else if (coord[2]>zsz-1) coord[2]=zsz-1;    
  }

  // Get start indicies of spline kernel into coeffcient field
  int sind[3];
  for (unsigned int i=0; i<3; i++) {
    sind[i] = (int)(coord[i]+0.5);
    sind[i] -= (sind[i] < coord[i]) ? 1 : 2;
  }

  // Calculate weights for separable kernels
  float wgts[12];  // wgts[0:3] x-dir, wgts[4:7] y-dir, wgts[8:11] z-dir
  float dwgts[12]; // Same arrangement as wgts
  for (int i=0; i<3; i++) {
    for (int j=0; j<4; j++) {
      float x = coord[i] - (sind[i] + j);
      float ax = abs(x);               // Kernels symmetric or anti-symmetrix
      int sign = (ax) ? int(x/ax) : 1; // Arbitrary choice for x=0
      if (ax < 1) {      // -1<x<1
	wgts[INDX(i,j)] = 2.0/3.0 + 0.5*ax*ax*(ax-2);
	dwgts[INDX(i,j)] = sign * (1.5*ax*ax - 2.0*ax);
      }
      else if (ax < 2) { // -2<x<-1 || 1<x<2
	ax = 2-ax; 
	wgts[INDX(i,j)] = (1.0/6.0)*(ax*ax*ax); 
	dwgts[INDX(i,j)] = sign * -0.5*ax*ax; 
      }  
      else wgts[INDX(i,j)] = dwgts[INDX(i,j)] = 0.0; // x<-2 || x>2
    }
  }

  // Calculate weighted sum of coefficient values
  float ws = 0.0; 
  float xws = 0.0; float yws = 0.0; float zws = 0.0;
  for (int k=0; k<4; k++) {
    float wgt1 = wgts[INDX(2,k)];
    float dzwgt1 = dwgts[INDX(2,k)];
    int linear1 = (xsz*ysz);
    if (bc==PERIODIC) linear1 *= i2i_p(sind[2]+k,zsz); 
    else if (bc==MIRROR) linear1 *= i2i_m(sind[2]+k,zsz);
    else linear1 *= i2i_c(sind[2]+k,zsz); // Defaults to constant
    for (int j=0; j<4; j++) {
      float wgt2 = wgt1 * wgts[INDX(1,j)];
      float dzwgt2 = dzwgt1 * wgts[INDX(1,j)];
      float dywgt2 = wgt1 * dwgts[INDX(1,j)];
      int linear2 = linear1;
      if (bc==PERIODIC) linear2 += i2i_p(sind[1]+j,ysz) * xsz; 
      else if (bc==MIRROR) linear2 += i2i_m(sind[1]+j,ysz) * xsz;
      else linear2 += i2i_c(sind[1]+j,ysz) * xsz;
      for (int i=0; i<4; i++) {
        float c;
	if (bc==PERIODIC) c = spcoef[linear2+i2i_p(sind[0]+i,xsz)]; 
	else if (bc==MIRROR) c = spcoef[linear2+i2i_m(sind[0]+i,xsz)];
	else c = spcoef[linear2+i2i_c(sind[0]+i,xsz)];
        ws += c*wgt2*wgts[i];
	xws += c*wgt2*dwgts[i];
        yws += c*dywgt2*wgts[i];
        zws += c*dzwgt2*wgts[i];
      }
    }
  }
  ima[id] = ws;
  xd[id] = xws;
  yd[id] = yws;
  zd[id] = zws;
}

__device__ int coord2index(int i, int j, int k, int xsz, int ysz)
{
  return(k*xsz*ysz + j*xsz + i);
}

__global__ void linear_interpolate(int   xsz,
				   int   ysz,
				   int   zsz,
				   const float *inima,
				   const float *xcoord, 
				   const float *ycoord, 
				   const float *zcoord, 
				   int         sizeT,
				   int         bc,
				   float       *oima)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= sizeT) { return; }
  float dx=xcoord[id]; float dy=ycoord[id]; float dz=zcoord[id];
  if (bc==CONSTANT) {
    if (dx<0) dx = 0; else if (dx>xsz-1) dx=xsz-1;
    if (dy<0) dy = 0; else if (dy>ysz-1) dy=ysz-1;
    if (dz<0) dz = 0; else if (dz>zsz-1) dz=zsz-1;    
  }
  int ix = ((int) floor(dx)); int iy = ((int) floor(dy)); int iz = ((int) floor(dz));
  dx -= ((float) ix); dy -= ((float) iy); dz -= ((float) iz);
  int ixp, iyp, izp;
  if (bc==PERIODIC) {
    ixp=i2i_p(ix+1,xsz); ix=i2i_p(ix,xsz);
    iyp=i2i_p(iy+1,ysz); iy=i2i_p(iy,ysz);
    izp=i2i_p(iz+1,zsz); iz=i2i_p(iz,zsz);
  }
  else if (bc==MIRROR) {
    ixp=i2i_m(ix+1,xsz); ix=i2i_m(ix,xsz);
    iyp=i2i_m(iy+1,ysz); iy=i2i_m(iy,ysz);
    izp=i2i_m(iz+1,zsz); iz=i2i_m(iz,zsz);
  }
  else {
    ixp=i2i_c(ix+1,xsz); ix=i2i_c(ix,xsz);
    iyp=i2i_c(iy+1,ysz); iy=i2i_c(iy,ysz);
    izp=i2i_c(iz+1,zsz); iz=i2i_c(iz,zsz);
  }
  float v000 = inima[coord2index(ix,iy,iz,xsz,ysz)];
  float v100 = inima[coord2index(ixp,iy,iz,xsz,ysz)];
  float v110 = inima[coord2index(ixp,iyp,iz,xsz,ysz)];
  float v101 = inima[coord2index(ixp,iy,izp,xsz,ysz)];
  float v111 = inima[coord2index(ixp,iyp,izp,xsz,ysz)];
  float v010 = inima[coord2index(ix,iyp,iz,xsz,ysz)];
  float v011 = inima[coord2index(ix,iyp,izp,xsz,ysz)];
  float v001 = inima[coord2index(ix,iy,izp,xsz,ysz)];
  float v00 = v000 + dz*(v001-v000);
  float v10 = v100 + dz*(v101-v100);
  float v01 = v010 + dz*(v011-v010);
  float v11 = v110 + dz*(v111-v110);
  float v0 = v00 + dy*(v01-v00);
  float v1 = v10 + dy*(v11-v10);
  oima[id] = v0 + dx*(v1-v0);
}

__global__ void linear_interpolate(int   xsz,
				   int   ysz,
				   int   zsz,
				   const float *inima,
				   const float *xcoord, 
				   const float *ycoord, 
				   const float *zcoord, 
				   int         sizeT,
				   int         bc,
				   float       *oima,
				   float       *xd,
				   float       *yd,
				   float       *zd)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= sizeT) { return; }
  float dx=xcoord[id]; float dy=ycoord[id]; float dz=zcoord[id];
  if (bc==CONSTANT) {
    if (dx<0) dx = 0; else if (dx>xsz-1) dx=xsz-1;
    if (dy<0) dy = 0; else if (dy>ysz-1) dy=ysz-1;
    if (dz<0) dz = 0; else if (dz>zsz-1) dz=zsz-1;    
  }
  int ix = ((int) floor(dx)); int iy = ((int) floor(dy)); int iz = ((int) floor(dz));
  dx -= ((float) ix); dy -= ((float) iy); dz -= ((float) iz);
  int ixp, iyp, izp;
  if (bc==PERIODIC) {
    ixp=i2i_p(ix+1,xsz); ix=i2i_p(ix,xsz);
    iyp=i2i_p(iy+1,ysz); iy=i2i_p(iy,ysz);
    izp=i2i_p(iz+1,zsz); iz=i2i_p(iz,zsz);
  }
  else if (bc==MIRROR) {
    ixp=i2i_m(ix+1,xsz); ix=i2i_m(ix,xsz);
    iyp=i2i_m(iy+1,ysz); iy=i2i_m(iy,ysz);
    izp=i2i_m(iz+1,zsz); iz=i2i_m(iz,zsz);
  }
  else {
    ixp=i2i_c(ix+1,xsz); ix=i2i_c(ix,xsz);
    iyp=i2i_c(iy+1,ysz); iy=i2i_c(iy,ysz);
    izp=i2i_c(iz+1,zsz); iz=i2i_c(iz,zsz);
  }
  float v000 = inima[coord2index(ix,iy,iz,xsz,ysz)];
  float v100 = inima[coord2index(ixp,iy,iz,xsz,ysz)];
  float v110 = inima[coord2index(ixp,iyp,iz,xsz,ysz)];
  float v101 = inima[coord2index(ixp,iy,izp,xsz,ysz)];
  float v111 = inima[coord2index(ixp,iyp,izp,xsz,ysz)];
  float v010 = inima[coord2index(ix,iyp,iz,xsz,ysz)];
  float v011 = inima[coord2index(ix,iyp,izp,xsz,ysz)];
  float v001 = inima[coord2index(ix,iy,izp,xsz,ysz)];

  float onemdz = 1.0-dz;
  float onemdy = 1.0-dy;    
  float tmp11 = onemdz*v000 + dz*v001;
  float tmp12 = onemdz*v010 + dz*v011;
  float tmp13 = onemdz*v100 + dz*v101;
  float tmp14 = onemdz*v110 + dz*v111;
  xd[id] = onemdy*(tmp13-tmp11) + dy*(tmp14-tmp12);
  yd[id] = (1.0-dx)*(tmp12-tmp11) + dx*(tmp14-tmp13);
  tmp11 = onemdy*v000 + dy*v010;
  tmp12 = onemdy*v001 + dy*v011;
  tmp13 = onemdy*v100 + dy*v110;
  tmp14 = onemdy*v101 + dy*v111;
  float tmp21 = (1.0-dx)*tmp11 + dx*tmp13;
  float tmp22 = (1.0-dx)*tmp12 + dx*tmp14;
  zd[id] = tmp22 - tmp21;
  oima[id] = onemdz*tmp21 + dz*tmp22;
  
  return;
}

#undef INDX

__device__ float init_fwd_sweep_periodic(float         *col,
					 int           coln,
					 float         z,
					 int           n)
{
  float iv = col[0];
  float *ptr = &col[coln-1];
  float z2i = z;
  for (int i=1; i<n; i++, ptr--, z2i*=z) iv += z2i * *ptr;
  return(iv);
}

__device__ float init_fwd_sweep_mirror(float         *col,
				       int           coln,
				       float         z,
				       int           n)
{
  float iv = col[0];
  float *ptr = &col[1];
  float z2i = z;
  for (int i=1; i<n; i++, ptr++, z2i*=z) iv += z2i * *ptr;
  return(iv);
}

__device__ float init_fwd_sweep_constant(float         *col,
					 int           coln,
					 float         z,
					 int           n)
{
  float iv = col[0];
  float z2i = z;
  for (int i=1; i<n; i++, z2i*=z) iv += z2i * col[0];
  return(iv);
}

__device__ float init_bwd_sweep_periodic(float        *col,
					 int          coln,
					 float        z,
					 int          n)
{
  float iv = z*col[coln-1];
  float z2i = z*z;
  for (int i=1; i<n; i++, z2i*=z) iv += z2i*col[i-1];
  return(iv / (z2i-1.0));
}

__device__ float init_bwd_sweep_mirror(float        *col,
				       int          coln,
				       float        z,
				       float        lv)
{
  float iv = -z/(1.0-z*z) * (2.0*col[coln-1] - lv);
  return(iv);
}

__device__ float init_bwd_sweep_constant(float        *col,
					 int          coln,
					 float        z,
					 int          n)
{
  float iv = z*col[coln-1];
  float z2i = z*z;
  for (int i=1; i<n; i++, z2i*=z) iv += z2i*col[coln-1];
  return(iv);
}

__global__ void cubic_spline_deconvolution(float        *data, 
					   unsigned int xsize, 
					   unsigned int ysize,
					   unsigned int zsize,
					   unsigned int dir,
					   unsigned int initn,
					   int          bc,
					   int          max_id)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= max_id) { return; }

  float col[MAX_IMA_SIZE];
  int startpos, stepsize, nsteps;
  float        z=-0.267949192431123f; // For cubic spline
  if (dir==0) { nsteps = xsize; startpos = id*xsize; stepsize = 1; }
  else if (dir==1) { nsteps = ysize; startpos = (id / xsize) * (xsize*ysize) + id%xsize; stepsize = xsize; }
  else if (dir==2) { nsteps = zsize; startpos = id; stepsize = xsize*ysize; }
	
  // Fill col from image
  for (int i=0; i<nsteps; i++) col[i] = data[startpos + i*stepsize];
  // Initialise forward sweep
  if (bc==PERIODIC) col[0] = init_fwd_sweep_periodic(col,nsteps,z,int(initn));
  else if (bc==MIRROR) col[0] = init_fwd_sweep_mirror(col,nsteps,z,int(initn));
  else col[0] = init_fwd_sweep_constant(col,nsteps,z,int(initn));
  float lv = col[nsteps-1];
  // Forward sweep
  for (int i=1; i<nsteps; i++) col[i] += z * col[i-1];
  // Initialise backward sweep
  if (bc==PERIODIC) col[nsteps-1] = init_bwd_sweep_periodic(col,nsteps,z,int(initn));
  else if (bc==MIRROR) col[nsteps-1] = init_bwd_sweep_mirror(col,nsteps,z,lv);
  else col[nsteps-1] = init_bwd_sweep_constant(col,nsteps,z,int(initn));
  // Backward sweep
  for (int i=nsteps-2; i>=0; i--) col[i] = z * (col[i+1] - col[i]);
  // Put coefficients back into place	
  for (int i=0; i<nsteps; i++) data[startpos + i*stepsize] = 6.0*col[i]; // 6 is scaling factor for cubic splines

  return;
}

__global__ void convolve_1D(// Input
			    unsigned int xsz,
			    unsigned int ysz,
			    unsigned int zsz,
			    const float  *ima,
			    const float  *krnl,
			    unsigned int krnsz,
			    unsigned int dir,
			    int          max_id,
			    // Output
			    float        *cima)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= max_id) { return; }
  int k = id / (xsz*ysz);
  int j = (id / xsz) % ysz;
  int i = id % xsz;
  float cv = 0.0;
  float wv = 0.0;
  if (dir==0) {
    for (int ii=0; ii<krnsz; ii++) {
      int indx = i+ii-krnsz/2;
      if (indx >= 0 && indx < xsz) {
	cv += krnl[ii] * ima[k*xsz*ysz+j*xsz+indx];
	wv += krnl[ii];
      }
    }
  }
  else if (dir==1) {
    for (int jj=0; jj<krnsz; jj++) {
      int indx = j+jj-krnsz/2;
      if (indx >= 0 && indx < ysz) {
	cv += krnl[jj] * ima[k*xsz*ysz+indx*xsz+i];
	wv += krnl[jj];
      }
    }
  }
  else {
    for (int kk=0; kk<krnsz; kk++) {
      int indx = k+kk-krnsz/2;
      if (indx >= 0 && indx < zsz) {
	cv += krnl[kk] * ima[indx*xsz*ysz+j*xsz+i];
	wv += krnl[kk];
      }
    }
  }
  cima[k*xsz*ysz+j*xsz+i] = cv/wv;
}

/****************************************************************//**
*  								  
*  Inverts a displacement field (in voxel units) without reference
*  to any other inverse fields.
*  
*  \param[in] dfield Displacement field in units of voxels. It is 
*  assumed that the displacements are in the fastest changing direction (x)
*  \param[in] inmask Mask indicating where the dfield is valid
*  \param[in] xsz Size of fields in x-direction
*  \param[in] ysz Size of fields in y-direction
*  \param[out] idfield The inverted field in units of voxels
*  \param[out] omask Mask indicating where idfield is valid
*  \param[in] max_id xsz*ysz*zsz
*
********************************************************************/ 
__global__ void invert_displacement_field_x(const float  *dfield,
					    const float  *inmask,
					    unsigned int xsz, 
					    unsigned int ysz,
					    float        *idfield,
					    float        *omask,
					    int          max_id)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= max_id) { return; }
  int i = threadIdx.x;
  int kpos = blockIdx.x*xsz*ysz;
  
  for (int j=0; j<ysz; j++) {
    int spos = kpos + j*xsz;
    int ub;
    for (ub=0; ub<xsz && dfield[spos+ub]+ub<i; ub++) ; // ub is now upper bound
    if (ub>0 && ub<xsz && inmask[spos+ub] && inmask[spos+ub-1]) { // If in valid range 
      idfield[spos+i] = (ub-1) - i + ((float) i - (ub-1) - dfield[spos+(ub-1)]) / ((float) dfield[spos+ub] - dfield[spos+(ub-1)] + 1.0);
      omask[spos+i] = 1.0;
    }
    else {
      idfield[spos+i] = 0.0;
      omask[spos+i] = 0.0;
    }
  }  
}

/****************************************************************//**
*  								  
*  Inverts a displacement field (in voxel units) _with_ reference
*  to another inverse field. The idea is that in "non-invertible" voxels
*  where there is some arbitrariness to what the "inverse" should be
*  we ensure consistency by setting the output inverse field to equal
*  the input inverse field.
*  
*  \param[in] dfield Displacement field in units of voxels. It is 
*  assumed that the displacements are in the fastest changing direction (x)
*  \param[in] inmask Mask indicating where the inidfield is valid
*  \param[in] inidfield Template inverse field. It will be used to 
*  bracket the inverse of dfield, and in the cases where the solution
*  doesn't fall in that brakcet idfield will be set to inidfield
*  \param[in] xsz Size of fields in x-direction
*  \param[in] ysz Size of fields in y-direction
*  \param[out] idfield The inverted field in units of voxels
*  \param[in] max_id xsz*ysz*zsz
*
********************************************************************/ 
__global__ void invert_displacement_field_x(const float  *dfield,
					    const float  *inmask,
					    const float  *inidfield,
					    unsigned int xsz, 
					    unsigned int ysz,
					    float        *idfield,
					    int          max_id)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= max_id) { return; }
  int i = threadIdx.x;
  int kpos = blockIdx.x*xsz*ysz;
  
  for (int j=0; j<ysz; j++) {
    int spos = kpos + j*xsz;
    if (inmask[spos+i]) {
      int ub = static_cast<int>(inidfield[spos+i] + i + 1); // Upper bound 
      if (ub>0 && ub<xsz && dfield[spos+ub]+ub>i && dfield[spos+ub-1]+ub-1<i) { // If in valid range 
	idfield[spos+i] = (ub-1) - i + ((float) i - (ub-1) - dfield[spos+(ub-1)]) / ((float) dfield[spos+ub] - dfield[spos+(ub-1)] + 1.0);
      }
      else idfield[spos+i] = inidfield[spos+i];
    }
    else idfield[spos+i] = 0.0;
  }  
}

/****************************************************************//**
*  								  
*  Inverts a displacement field (in voxel units) without reference
*  to any other inverse fields.
*  
*  \param[in] dfield Displacement field in units of voxels. It is 
*  assumed that the displacements are in the second fastest changing 
*  direction (y)
*  \param[in] inmask Mask indicating where the dfield is valid
*  \param[in] xsz Size of fields in x-direction
*  \param[in] ysz Size of fields in y-direction
*  \param[out] idfield The inverted field in units of voxels
*  \param[out] omask Mask indicating where idfield is valid
*  \param[in] max_id xsz*ysz*zsz
*
********************************************************************/ 
__global__ void invert_displacement_field_y(const float  *dfield,
					    const float  *inmask,
					    unsigned int xsz, 
					    unsigned int ysz,
					    float        *idfield,
					    float        *omask,
					    int          max_id)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= max_id) { return; }
  int i = threadIdx.x;
  int kpos = blockIdx.x*xsz*ysz;
  
  for (int j=0; j<ysz; j++) {
    int spos = kpos + j*xsz;
    int ub;
    for (ub=0; ub<ysz && dfield[kpos+ub*xsz+i]+ub<j; ub++) ; // ub is now upper bound
    if (ub>0 && ub<ysz && inmask[kpos+(ub-1)*xsz+i] && inmask[kpos+ub*xsz+i]) { // If in valid range
      idfield[spos+i] = (ub-1) - j + ((float) j - (ub-1) - dfield[kpos+(ub-1)*xsz+i]) / ((float) dfield[kpos+ub*xsz+i] - dfield[kpos+(ub-1)*xsz+i] + 1.0);
      omask[spos+i] = 1.0;
    }
    else {
      idfield[spos+i] = 0.0;
      omask[spos+i] = 0.0;
    }
  }  
}

/****************************************************************//**
*  								  
*  Inverts a displacement field (in voxel units) _with_ reference
*  to another inverse field. The idea is that in "non-invertible" voxels
*  where there is some arbitrariness to what the "inverse" should be
*  we ensure consistency by setting the output inverse field to equal
*  the input inverse field.
*  
*  \param[in] dfield Displacement field in units of voxels. It is 
*  assumed that the displacements are in the second fastest changing 
*  direction (y)
*  \param[in] inmask Mask indicating where the inidfield is valid
*  \param[in] inidfield Template inverse field. It will be used to 
*  bracket the inverse of dfield, and in the cases where the solution
*  doesn't fall in that brakcet idfield will be set to inidfield
*  \param[in] xsz Size of fields in x-direction
*  \param[in] ysz Size of fields in y-direction
*  \param[out] idfield The inverted field in units of voxels
*  \param[in] max_id xsz*ysz*zsz
*
********************************************************************/ 
__global__ void invert_displacement_field_y(const float  *dfield,
					    const float  *inmask,
					    const float  *inidfield,
					    unsigned int xsz, 
					    unsigned int ysz,
					    float        *idfield,
					    int          max_id)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= max_id) { return; }
  int i = threadIdx.x;
  int kpos = blockIdx.x*xsz*ysz;
  
  for (int j=0; j<ysz; j++) {
    int spos = kpos + j*xsz;
    if (inmask[spos+i]) {
      int ub = static_cast<int>(inidfield[spos+i] + j + 1); // Upper bound
      if (ub>0 && ub<ysz && dfield[kpos+ub*xsz+i]+ub>j && dfield[kpos+(ub-1)*xsz+i]+ub-1<j) { // If in valid range
	idfield[spos+i] = (ub-1) - j + ((float) j - (ub-1) - dfield[kpos+(ub-1)*xsz+i]) / ((float) dfield[kpos+ub*xsz+i] - dfield[kpos+(ub-1)*xsz+i] + 1.0);
      }
      else idfield[spos+i] = inidfield[spos+i];
    }
    else idfield[spos+i] = 0.0;
  }  
}

__global__ void invert_displacement_field(const float  *dfield,
					  const float  *inmask,
					  unsigned int xsize, 
					  unsigned int ysize,
					  unsigned int zsize,
					  unsigned int dir,
					  float        *idfield,
					  float        *omask,
					  int          max_id)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= max_id) { return; }
  unsigned int startpos;
  unsigned int stepsize;
  unsigned int nsteps;
  if (dir==0) {
    nsteps = xsize;
    startpos = id*xsize;
    stepsize = 1;
  }
  else if (dir==1) {
    nsteps = ysize;
    startpos = (id / xsize) * (xsize*ysize) + id%xsize;
    stepsize = xsize;
  }
  else if (dir==2) {
    nsteps = zsize;
    startpos = id;
    stepsize = xsize*ysize;
  }

  int oi=0;
  for (int i=0; i<nsteps; i++) {
    int ii=oi;
    for (; ii<nsteps && dfield[startpos+ii*stepsize]+ii<i; ii++) ; // On purpose
    if (ii>0 && ii<nsteps) {                                       // If we are in valid range
      idfield[startpos+i*stepsize] = ii - i - 1 + ((float) i+1-ii-dfield[startpos+(ii-1)*stepsize]) / ((float) dfield[startpos+ii*stepsize]+1.0-dfield[startpos+(ii-1)*stepsize]);
      if (inmask[startpos+(ii-1)*stepsize]) omask[startpos+i*stepsize] = 1.0;
      else omask[startpos+i*stepsize] = 0.0;
    }
    else {
      idfield[startpos+i*stepsize] = CUDART_MAX_NORMAL_F;          // NaN tag for further processing
      omask[startpos+i*stepsize] = 0.0;
    }
    oi = max(0,ii-1);
  }
  // Process NaNs at beginning of column
  int i=0;
  for (; i<nsteps-1 && idfield[startpos+i*stepsize]==CUDART_MAX_NORMAL_F; i++) ;    // On purpose
  for (; i>0; i--) idfield[startpos+(i-1)*stepsize] = idfield[startpos+i*stepsize];
  // Process NaNs at end of column
  for (i=nsteps-1; i>0 && idfield[startpos+i*stepsize]==CUDART_MAX_NORMAL_F; i--) ; // On purpose
  for (; i<nsteps-1; i++) idfield[startpos+(i+1)*stepsize] = idfield[startpos+i*stepsize];
  
  return;
}

__global__ void slice_modulate_deriv(const float  *inima,
				     const float  *mask,
				     unsigned int xsz,
				     unsigned int ysz,
				     unsigned int zsz,
				     const float  *mod,
				     float        *oima,
				     int          max_id)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= max_id || blockIdx.x >= zsz) { return; }
  int i = threadIdx.x;
  int kpos = blockIdx.x*xsz*ysz;
  float modval = mod[blockIdx.x];

  for (int j=0; j<ysz; j++) {
    int indx = kpos+j*xsz+i;
    if (mask[indx]) oima[indx] = modval*inima[indx];
    else oima[indx] = 0.0;
  }
  return;
}

__global__ void valid_voxels(unsigned int xs,
			     unsigned int ys,
			     unsigned int zs,
			     bool         xv,
			     bool         yv,
			     bool         zv,
			     const float  *x,
			     const float  *y,
			     const float  *z,
			     int          max_id,
			     float        *mask)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= max_id) { return; }

  if ((xv || (x[id]>0 && x[id]<xs-1)) &&
      (yv || (y[id]>0 && y[id]<ys-1)) &&
      (zv || (z[id]>0 && z[id]<zs-1))) mask[id] = 1.0;
  else mask[id] = 0.0;
}

__global__ void implicit_coord_sub(unsigned int xs,
				   unsigned int ys,
				   unsigned int zs,
				   float        *x,
				   float        *y,
				   float        *z,
				   int          max_id)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= max_id) { return; }

  float i = id % xs;
  float j = (id / xs) % ys;
  float k = id / (xs*ys);

  x[id] -= i;
  y[id] -= j;
  z[id] -= k;
}

__global__ void subtract_multiply_and_add_to_me(const float  *pv,
						const float  *nv,
						float        a,
						int          max_id,
						float        *out)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= max_id) { return; }

  out[id] += a * (pv[id] - nv[id]);
}

__device__ float square(float a) { return(a*a); }

__global__ void subtract_square_and_add_to_me(const float  *pv,
					      const float  *nv,
					      int          max_id,
					      float        *out)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= max_id) { return; }

  out[id] += square(pv[id] - nv[id]);
}

__global__ void make_deriv_first_part(int xsz, int ysz, int zsz,
				      const float *xcoord, const float *ycoord, const float *zcoord, 
				      const float *xgrad, const float *ygrad, const float *zgrad,
				      const float *base, const float *jac, const float *basejac,
				      float dstep,
				      float *deriv, 
				      int max_id)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= max_id) { return; }
  deriv[id] = (xcoord[id]*xgrad[id] + ycoord[id]*ygrad[id] + zcoord[id]*zgrad[id]) / dstep;
  return;
}

__global__ void make_deriv_second_part(int xsz, int ysz, int zsz,
				       const float *xcoord, const float *ycoord, const float *zcoord, 
				       const float *xgrad, const float *ygrad, const float *zgrad,
				       const float *base, const float *jac, const float *basejac,
				       float dstep,
				       float *deriv, 
				       int max_id)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= max_id) { return; }
  deriv[id] = base[id]*(jac[id] - basejac[id]) / dstep;
  return;
}

__global__ void sample_derivs_along_x(int xsz, int ysz, int zsz, 
				      const float  *ima, 
				      bool         add_one,
				      ExtrapType   ep,
				      float        *deriv, 
				      int          max_id)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= max_id) { return; }
  int i = threadIdx.x;
  int k = blockIdx.x;
  float add = (add_one) ? 1.0 : 0.0;
  for (int j=0; j<ysz; j++) {
    float pval; 
    float nval;
    if (i==0) {
      nval = ima[k*xsz*ysz+j*xsz+i+1];      
      if (ep == EddyKernels::PERIODIC) pval = ima[k*xsz*ysz+j*xsz+xsz-1];
      else pval = ima[k*xsz*ysz+j*xsz+i]; // MIRROR or CONSTANT
    }
    else if (i == (xsz-1)) {
      pval = ima[k*xsz*ysz+j*xsz+i-1];
      if (ep == EddyKernels::PERIODIC) nval = ima[k*xsz*ysz+j*xsz];
      else nval = ima[k*xsz*ysz+j*xsz+i]; // MIRROR or CONSTANT
    }
    else {
      pval = ima[k*xsz*ysz+j*xsz+i-1];
      nval = ima[k*xsz*ysz+j*xsz+i+1];
    }
    deriv[k*xsz*ysz+j*xsz+i] = add + (nval - pval) / 2.0;
  }
  return;
}

__global__ void masked_sample_derivs_along_x(int xsz, int ysz, int zsz, 
					     const float  *ima,
					     const float  *mask,
					     bool         add_one,
					     ExtrapType   ep,
					     float        *deriv, 
					     int          max_id)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= max_id) { return; }
  int i = threadIdx.x;
  int k = blockIdx.x;
  float add = (add_one) ? 1.0 : 0.0;
  for (int j=0; j<ysz; j++) {
    if (mask[k*xsz*ysz+j*xsz+i] == 0) {
      deriv[k*xsz*ysz+j*xsz+i] = add;
    }
    else {
      float pval; 
      float nval;
      float div = 2.0;
      if (i==0) {
	if (mask[k*xsz*ysz+j*xsz+i+1]) { nval = ima[k*xsz*ysz+j*xsz+i+1]; }
	else { nval = ima[k*xsz*ysz+j*xsz+i]; div = 1.0; }
	if (ep == EddyKernels::PERIODIC) {
	  if (mask[k*xsz*ysz+j*xsz+xsz-1]) { pval = ima[k*xsz*ysz+j*xsz+xsz-1]; }
	  else { pval = ima[k*xsz*ysz+j*xsz+i]; div = 1.0; }
	}
	else { pval = ima[k*xsz*ysz+j*xsz+i]; } // CONSTANT or MIRROR
      }
      else if (i == (xsz-1)) {
	if (mask[k*xsz*ysz+j*xsz+i-1]) { pval = ima[k*xsz*ysz+j*xsz+i-1]; }
	else { pval = ima[k*xsz*ysz+j*xsz+i]; div = 1.0; }
	if (ep == EddyKernels::PERIODIC) {
	  if (mask[k*xsz*ysz+j*xsz]) { nval = ima[k*xsz*ysz+j*xsz]; }
	  else { nval = ima[k*xsz*ysz+j*xsz+i]; div = 1.0; }
	}
	else { pval = ima[k*xsz*ysz+j*xsz+i]; } // CONSTANT or MIRROR
      }
      else {
	if (mask[k*xsz*ysz+j*xsz+i-1]) { pval = ima[k*xsz*ysz+j*xsz+i-1]; }
	else { pval = ima[k*xsz*ysz+j*xsz+i]; div = 1.0; }
	if (mask[k*xsz*ysz+j*xsz+i+1]) { nval = ima[k*xsz*ysz+j*xsz+i+1]; }
	else { nval = ima[k*xsz*ysz+j*xsz+i]; div = 1.0; }
      }
      deriv[k*xsz*ysz+j*xsz+i] = add + (nval - pval) / div;
    }
  }
  return;
}

__global__ void sample_derivs_along_y(int xsz, int ysz, int zsz, 
				      const float  *ima, 
				      bool         add_one,
				      ExtrapType   ep,
				      float        *deriv, 
				      int          max_id)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= max_id) { return; }
  int i = threadIdx.x;
  int k = blockIdx.x;
  float add = (add_one) ? 1.0 : 0.0;
  // First one on the "near edge"
  if (ep == EddyKernels::PERIODIC) deriv[k*xsz*ysz+i] = add + (ima[k*xsz*ysz+1*xsz+i] - ima[k*xsz*ysz+(ysz-1)*xsz+i]) / 2.0;
  else deriv[k*xsz*ysz+i] = add + (ima[k*xsz*ysz+1*xsz+i] - ima[k*xsz*ysz+i]) / 2.0; // MIRROR or CONSTANT
  // All the ones that are not on the edge
  for (int j=1; j<(ysz-1); j++) {
    deriv[k*xsz*ysz+j*xsz+i] = add + (ima[k*xsz*ysz+(j+1)*xsz+i] - ima[k*xsz*ysz+(j-1)*xsz+i]) / 2.0; 
  }
  // Last one of the "far edge" 
  if (ep == EddyKernels::PERIODIC) deriv[k*xsz*ysz+(ysz-1)*xsz+i] = add + (ima[k*xsz*ysz+i] - ima[k*xsz*ysz+(ysz-2)*xsz+i]) / 2.0;
  else deriv[k*xsz*ysz+(ysz-1)*xsz+i] = add + (ima[k*xsz*ysz+(ysz-1)*xsz+i] - ima[k*xsz*ysz+(ysz-2)*xsz+i]) / 2.0; // MIRROR or CONSTANT

  return;
}

__global__ void masked_sample_derivs_along_y(int xsz, int ysz, int zsz, 
					     const float  *ima,
					     const float  *mask,
					     bool         add_one,
					     ExtrapType   ep,
					     float        *deriv, 
					     int          max_id)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= max_id) { return; }
  int i = threadIdx.x;
  int k = blockIdx.x;
  float add = (add_one) ? 1.0 : 0.0;
  // First one on the "near edge"
  if (mask[k*xsz*ysz+i] == 0) deriv[k*xsz*ysz+i] = add;
  else {
    float div = 2.0;
    float pval;
    float nval;
    if (mask[k*xsz*ysz+1*xsz+i]) { nval = ima[k*xsz*ysz+1*xsz+i]; }
    else { nval = ima[k*xsz*ysz+i]; div = 1.0; }
    if (ep == EddyKernels::PERIODIC) {
      if (mask[k*xsz*ysz+(ysz-1)*xsz+i]) { pval = ima[k*xsz*ysz+(ysz-1)*xsz+i]; } 
      else { pval = ima[k*xsz*ysz+i]; div = 1.0; }
    }
    else { pval = ima[k*xsz*ysz+i]; } // CONSTANT or MIRROR
    deriv[k*xsz*ysz+i] = add + (nval - pval) / div;
  }
  // All the ones that are not on the edge
  for (int j=1; j<(ysz-1); j++) {
    if (mask[k*xsz*ysz+j*xsz+i] == 0) deriv[k*xsz*ysz+j*xsz+i] = add;
    else {
      float div = 2.0;
      float pval; 
      if (mask[k*xsz*ysz+(j-1)*xsz+i]) { pval = ima[k*xsz*ysz+(j-1)*xsz+i]; }
      else { pval = ima[k*xsz*ysz+j*xsz+i]; div = 1.0; }
      float nval; 
      if (mask[k*xsz*ysz+(j+1)*xsz+i]) { nval = ima[k*xsz*ysz+(j+1)*xsz+i]; }
      else { nval = ima[k*xsz*ysz+j*xsz+i]; div = 1.0; }
      deriv[k*xsz*ysz+j*xsz+i] = add + (nval - pval) / div; 
    }
  }
  // Last one on the "far edge"
  if (mask[k*xsz*ysz+(ysz-1)*xsz+i] == 0) deriv[k*xsz*ysz+(ysz-1)*xsz+i] = add;
  else {
    float div = 2.0;
    float pval;
    if (mask[k*xsz*ysz+(ysz-2)*xsz+i]) { pval = ima[k*xsz*ysz+(ysz-2)*xsz+i]; }
    else { pval = ima[k*xsz*ysz+(ysz-1)*xsz+i]; div = 1.0; }
    float nval;
    if (ep == EddyKernels::PERIODIC) { 
      if (mask[k*xsz*ysz+i]) { nval = ima[k*xsz*ysz+i]; }
      else { nval = ima[k*xsz*ysz+(ysz-1)*xsz+i]; div = 1.0; }
    }
    else { nval = ima[k*xsz*ysz+(ysz-1)*xsz+i]; }
    deriv[k*xsz*ysz+(ysz-1)*xsz+i] = add + (nval - pval) / div;
  }

  return;
}

__global__ void make_mask_from_stack(const float   *inmask,
				     const float   *zcoord,
				     unsigned int  xsz,
				     unsigned int  ysz,
				     unsigned int  zsz,
				     float         *omask)
{
  if (blockIdx.x < ysz && threadIdx.x < xsz) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int zstep = xsz*ysz;
    for (unsigned int k=0; k<zsz; k++) {
      int z = static_cast<int>(zcoord[id+k*zstep]+0.5);
      if (z>-1 && z<zsz-1 && inmask[id+z*zstep]>0.0) {
	omask[id+k*zstep] = 1.0;
      }
    }    
  }
}

__global__ void transfer_y_hat_to_volume(const float   *yhat,
					 unsigned int  xsz,
					 unsigned int  ysz,
					 unsigned int  zsz,
					 unsigned int  y,
					 float         *vol)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < xsz) {
    float *lvol = &vol[y*xsz+id];
    const float *ly = &yhat[id*zsz];
    int zstep = xsz*ysz;
    for (unsigned k=0; k<zsz; k++) {
      lvol[k*zstep] = ly[k];
    }
  }
}

__global__ void TransferAndCheckSorting(const float  *origz,
					unsigned int xsz,
					unsigned int ysz,
					unsigned int zsz,
					float        *sortz,
					unsigned int *flags)
{
  int i = threadIdx.x;
  int j = blockIdx.x;
  if (i<xsz && j<ysz) {
    unsigned int offs = j*xsz + i;
    const float *op = origz + offs;
    float *sp = sortz + zsz*offs;
    for (unsigned int k=0; k<zsz; k++) {
      sp[k] = *op;
      if (k) {
	if (sp[k-1] > sp[k]) flags[offs] = 1;
      }
      op+=(xsz*ysz);
    }
  }
}

__global__ void TransferVolumeToVectors(const float  *orig,
					unsigned int xsz,
					unsigned int ysz,
					unsigned int zsz,
					float        *trgt)
{
  int i = threadIdx.x;
  int j = blockIdx.x;
  if (i<xsz && j<ysz) {
    unsigned int offs = j*xsz + i;
    const float *op = orig + offs;
    float *tp = trgt + zsz*offs;
    for (unsigned int k=0; k<zsz; k++) { tp[k] = *op; op+=(xsz*ysz); }
  }
}

// Performs crappy bubble sort, but vectors will almost
// always be very close to sorted. So, hopefully ok.  
__global__ void SortVectors(const unsigned int  *indx,
			    unsigned int        nindx,
			    unsigned int        zsz,
			    float               *key,
			    float               *vec2)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < nindx) {
    unsigned int offs = indx[id]*zsz;
    float *kp = key + offs;
    float *vp = NULL;
    if (vec2) vp = vec2 + offs;
    unsigned int ns = zsz;
    for (unsigned int pass=0; pass<zsz; pass++) { 
      bool noswap=true;
      for (unsigned int i=1; i<ns; i++) {
	if (kp[i-1]>kp[i]) { 
	  float tmp = kp[i-1]; kp[i-1] = kp[i]; kp[i] = tmp; noswap=false;
	  if (vp) { tmp = vp[i-1]; vp[i-1] = vp[i]; vp[i] = tmp; }
	}
      }
      ns--;
      if (noswap) break;
    }
  }  
}

__global__ void LinearInterpolate(const float    *zcoord,
				  const float    *val,
				  unsigned int   zsz,
				  float          *ival)
{
  unsigned int offs = zsz * (blockIdx.x * blockDim.x + threadIdx.x);
  const float *zp = zcoord + offs;
  const float *vp = val + offs;
  float *ivp = ival + offs;
  for (unsigned int z=0; z<zsz; z++) {
    unsigned int k=0;
    for (k=0; k<zsz; k++) if (zp[k] > z) break;
    if (k==0) ivp[z] = 0.0;
    else if (k==zsz) ivp[z] = 0.0;
    else ivp[z] = vp[k-1] + (z-zp[k-1])*(vp[k]-vp[k-1])/(zp[k]-zp[k-1]);
  }
}

__global__ void TransferColumnsToVolume(const float    *zcols,
					unsigned int   xsz,
					unsigned int   ysz,
					unsigned int   zsz,
					float          *vol)
{
  if (blockIdx.x < ysz && threadIdx.x < xsz) {
    unsigned int offs = blockIdx.x * blockDim.x + threadIdx.x;
    const float *zp = zcols + zsz*offs;
    float *vp = vol + offs;
    unsigned int vstep = xsz*ysz;
    for (unsigned int i=0; i<zsz; i++) {
      *vp = zp[i];
      vp += vstep;
    }
  }
}

  /*
__global__ void MakeWeights(const float  *zcoord,
			    unsigned int xsz,
			    unsigned int zsz,
			    unsigned int j,
			    float        *weight)
{
  unsigned int i=blockIdx.x;
  unsigned int k=threadIdx.x;
  if (i<xsz && k<zsz) {
    const float *zp = zcoord + zsz*(j*xsz+i);
    float *wp = weight + zsz*i + k;
    unsigned int ii=0;
    for (ii=0; ii<zsz; ii++) if (zp[ii] > k) break;
    if (ii == 0) {
      *wp = 1.0;
    }
    else if (ii == zsz) {
      *wp = 1.0;
    }
    else { // The "normal" case
      float dp = (zp[ii]-zp[ii-1]);
      if (dp < 1.0) *wp = 0.0;                                            
      else *wp = 1; 
    }
  }
}

  */

__global__ void MakeWeights(const float  *zcoord,
			    unsigned int xsz,
			    unsigned int zsz,
			    unsigned int j,
			    float        *weight)
{
  unsigned int i=blockIdx.x;
  unsigned int k=threadIdx.x;
  if (i<xsz && k<zsz) {
    const float *zp = zcoord + zsz*(j*xsz+i);
    float *wp = weight + zsz*i + k;
    unsigned int ii=0;
    for (ii=0; ii<zsz; ii++) if (zp[ii] > k) break;
    if (ii == 0) {
      *wp = min(zp[ii]-k,1.0);
    }
    else if (ii == zsz) {
      *wp = min(k-zp[ii-1],1.0);
    }
    else { // The "normal" case
      float dp = (zp[ii]-zp[ii-1]);
      if (dp < 1.0) *wp = 0.0;                                            // Gap is less than one voxel
      else if (dp < 2.0 & max(k-zp[ii-1],zp[ii]-k) < 1.0) *wp = dp - 1.0; // Gap<2vox and only one pred in bracket
      else *wp = min(1.0,min(k-zp[ii-1],zp[ii]-k));                       // More than one pred in bracket
    }
    if (*wp > 1e-12) *wp = sqrt(*wp); // Lets not attempt sqrt from very small value
  }
}

__global__ void InsertWeights(const float  *wvec,
			      unsigned int j,
			      unsigned int xsz,
			      unsigned int ysz,
			      unsigned int zsz,
			      float        *wvol)
{
  unsigned int i=blockIdx.x;
  unsigned int k=threadIdx.x;
  if (i<xsz && k<zsz && j<ysz) {
    wvol[k*xsz*ysz+j*xsz+i] = wvec[i*zsz+k];
  }
}

__global__ void MakeDiagwpVecs(const float *pred,
			       const float *wgts,
			       unsigned int xsz,
			       unsigned int ysz,
			       unsigned int zsz,
			       unsigned int j,
			       float        *diagwp)
{
  unsigned int i=blockIdx.x;
  unsigned int k=threadIdx.x;
  if (i<xsz && k<zsz) {
    diagwp[i*zsz+k] = pred[k*xsz*ysz + j*xsz + i] * wgts[i*zsz+k];
  }
}

} // End namespace EddyKernels

// Some dead code


/*

__global__ void x_modulate_deriv(const float  *inima,
				 const float  *mask,
				 unsigned int xsz,
				 unsigned int ysz,
				 unsigned int zsz,
				 float        vxs,
				 float        *oima,
				 int          max_id)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= max_id || threadIdx.x >= xsz || blockIdx.x >= zsz) { return; }
  float modval = vxs*(threadIdx.x-(xsz-1)/2.0);
  int bpos = blockIdx.x*xsz*ysz+threadIdx.x;
  
  for (int j=0;j<ysz;j++) {
    int indx = bpos + j*xsz;
    if (mask[indx]) oima[indx] = modval*inima[indx];
    else oima[indx] = 0.0;
  }
  return;
}

__global__ void y_modulate_deriv(const float  *inima,
				 const float  *mask,
				 unsigned int xsz,
				 unsigned int ysz,
				 unsigned int zsz,
				 float        vxs,
				 float        *oima,
				 int          max_id)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= max_id || threadIdx.x >= xsz || blockIdx.x >= zsz) { return; }
  int bpos = blockIdx.x*xsz*ysz+threadIdx.x;
  
  for (int j=0;j<ysz;j++) {
    int indx = bpos + j*xsz;
    if (mask[indx]) oima[indx] = vxs*(j-(ysz-1)/2.0)*inima[indx];
    else oima[indx] = 0.0;
  }
  return;
}

__global__ void z_modulate_deriv(const float  *inima,
				 const float  *mask,
				 unsigned int xsz,
				 unsigned int ysz,
				 unsigned int zsz,
				 float        vxs,
				 float        *oima,
				 int          max_id)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= max_id || threadIdx.x >= xsz || blockIdx.x >= zsz) { return; }
  float modval = vxs*(blockIdx.x-(zsz-1)/2.0);
  int bpos = blockIdx.x*xsz*ysz+threadIdx.x;
  
  for (int j=0;j<ysz;j++) {
    int indx = bpos + j*xsz;
    if (mask[indx]) oima[indx] = modval*inima[indx];
    else oima[indx] = 0.0;
  }
  return;
}

__global__ void get_lower_bound_of_inverse_x(const float  *dfield,
					     const float  *inmask,
					     unsigned int xsz, 
					     unsigned int ysz,
					     unsigned int zsz,
					     int          *lb,
					     float        *omask,
					     int          max_id)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= max_id) { return; }
  int i = threadIdx.x;
  int kpos = blockIdx.x*xsz*ysz;

  for (int j=0; j<ysz; j++) {
    int spos = kpos + j*xsz;
    int ii;
    for (ii=0; ii<xsz && dfield[spos+ii]+ii<i; ii++) ; // ii is now upper bound
    if (ii>0 && ii<xsz) { // If in valid range 
      lb[spos+i] = ii-1;
      if (inmask[spos+ii-1]) omask[spos+i] = 1.0;
      else omask[spos+i] = 0.0;
    }
    else {
      lb[spos+i] = -1; // -1 is marker for outside valid range
      omask[spos+i] = 0.0;
    }
  }
}

__global__ void get_lower_bound_of_inverse_y(const float  *dfield,
					     const float  *inmask,
					     unsigned int xsz, 
					     unsigned int ysz,
					     unsigned int zsz,
					     int          *lb,
					     float        *omask,
					     int          max_id)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= max_id) { return; }
  int i = threadIdx.x;
  int kpos = blockIdx.x*xsz*ysz;

  for (int j=0; j<ysz; j++) {
    int jj;
    for (jj=0; jj<ysz && dfield[kpos+jj*xsz+i]+jj<j; jj++) ; // jj is now upper bound
    if (jj>0 && jj<ysz) { // If in valid range
      lb[kpos+j*xsz+i] = jj-1;
      if (inmask[kpos+(jj-1)*xsz+i]) omask[kpos+j*xsz+i] = 1.0;
      else omask[kpos+j*xsz+i] = 0.0;
    }
    else  {
      lb[kpos+j*xsz+i] = -1; // -1 is marker for outside valid range
      omask[kpos+j*xsz+i] = 0.0;
    }
  }
}

__global__ void invert_displacement_field_x(const float  *dfield,
					    const float  *inmask,
					    const int    *lb,
					    unsigned int xsz, 
					    unsigned int ysz,
					    unsigned int zsz,
					    float        *idfield,
					    int          max_id)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= max_id) { return; }
  int i = threadIdx.x;
  int kpos = blockIdx.x*xsz*ysz;
  
  for (int j=0; j<ysz; j++) {
    int spos = kpos + j*xsz;
    if (lb[spos+i] != -1 && inmask[spos+i]) { // If in valid voxels
      idfield[spos+i] = lb[spos+i] - i + ((float) i - lb[spos+i] - dfield[spos+lb[spos+i]]) / ((float) dfield[spos+lb[spos+i]+1] - dfield[spos+lb[spos+i]] + 1.0);  
    }
  }  
}

__global__ void invert_displacement_field_y(const float  *dfield,
					    const float  *inmask,
					    const int    *lb,
					    unsigned int xsz, 
					    unsigned int ysz,
					    unsigned int zsz,
					    float        *idfield,
					    int          max_id)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= max_id) { return; }
  int i = threadIdx.x;
  int kpos = blockIdx.x*xsz*ysz;
  
  for (int j=0; j<ysz; j++) {
    int spos = kpos + j*xsz;
    if (lb[spos+i] != -1 && inmask[spos+i]) { // If in valid voxels
      idfield[spos+i] = lb[spos+1] - j + ((float) j - lb[spos+i] - dfield[kpos+lb[spos+i]*xsz+i]) / ((float) dfield[kpos+(lb[spos+i]+1)*xsz+i] - dfield[kpos+lb[spos+i]*xsz+i] + 1.0);  
    }
  }  
}

// The functions make_deriv_first_part and make_deriv_second_part
// are for debugging purposes

__global__ void make_deriv_first_part(int xsz, int ysz, int zsz,
				      const float *xcoord, const float *ycoord, const float *zcoord, 
				      const float *xgrad, const float *ygrad, const float *zgrad,
				      const float *base, const float *jac, const float *basejac,
				      float dstep,
				      float *deriv, 
				      int max_id)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= max_id) { return; }
  deriv[id] = (xcoord[id]*xgrad[id] + ycoord[id]*ygrad[id] + zcoord[id]*zgrad[id]) / dstep;
  return;
}

__global__ void make_deriv_second_part(int xsz, int ysz, int zsz,
				       const float *xcoord, const float *ycoord, const float *zcoord, 
				       const float *xgrad, const float *ygrad, const float *zgrad,
				       const float *base, const float *jac, const float *basejac,
				       float dstep,
				       float *deriv, 
				       int max_id)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= max_id) { return; }
  deriv[id] = base[id]*(jac[id] - basejac[id]) / dstep;
  return;
}

*/

/*

__global__ void XtX(unsigned int        nima,
		    unsigned int        nvox,
		    const float* const  *iptrv,
		    const float         *mask,
		    float               *XtX,
		    int                 max_id)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= max_id || threadIdx.x > blockIdx.x || blockIdx.x >= nima) { return; }
  const float *cptr=iptrv[blockIdx.x];  // Column-pointer
  const float *rptr=iptrv[threadIdx.x]; // Row-pointer
  double ip = 0.0; // Inner product
  for (unsigned int i=0; i<nvox; i++) if (mask[i] != 0.0) ip += cptr[i]*rptr[i];
  XtX[blockIdx.x*nima+threadIdx.x] = ip;

  return;
}

*/
