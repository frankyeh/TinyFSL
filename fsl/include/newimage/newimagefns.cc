/*  newimagefns.cc

    Mark Jenkinson, FMRIB Image Analysis Group

    Copyright (C) 2000 University of Oxford  */

/*  CCOPYRIGHT  */

// General image processing functions


#include "armawrap/newmat.h"
#include "NewNifti/NewNifti.h"
#include "miscmaths/miscmaths.h"
#include "newimagefns.h"


using namespace std;
using namespace NEWMAT;
using namespace NiftiIO;
using namespace NiftiIO::legacy;
using namespace MISCMATHS;

namespace NEWIMAGE {

  ///////////////////////////////////////////////////////////////////////////

  // BASIC IMAGE SUPPORT FUNCTIONS
  template <class T>
  void pad(const volume<T>& vol, volume<T>& paddedvol,
	     int offsetx, int offsety, int offsetz)
    {
      // Note: the voxel at (offsetx,offsety,offsetz) in PADDEDVOL
      //       will be the same as (0,0,0) in VOL
      paddedvol.copyproperties(vol);

      extrapolation oldex = vol.getextrapolationmethod();
      if ((oldex==boundsassert) || (oldex==boundsexception))
	{ vol.setextrapolationmethod(constpad); }
      for (int z=0; z<paddedvol.zsize(); z++) {
	for (int y=0; y<paddedvol.ysize(); y++) {
	  for (int x=0; x<paddedvol.xsize(); x++) {
	    paddedvol(x,y,z) = vol(x-offsetx,y-offsety,z-offsetz);
	  }
	}
      }
      // set the sform and qform appropriately (currently equal to vol's)
      Matrix pad2vol(4,4);
      pad2vol = IdentityMatrix(4);
      pad2vol(1,4) = -offsetx;
      pad2vol(2,4) = -offsety;
      pad2vol(3,4) = -offsetz;
      if (paddedvol.sform_code()!=NIFTI_XFORM_UNKNOWN) {
	paddedvol.set_sform(paddedvol.sform_code(),
			    paddedvol.sform_mat() * pad2vol);
      }
      if (paddedvol.qform_code()!=NIFTI_XFORM_UNKNOWN) {
	paddedvol.set_qform(paddedvol.qform_code(),
			    paddedvol.qform_mat() * pad2vol);
      }
      vol.setextrapolationmethod(oldex);
    }

template void pad(const volume<char>& vol, volume<char>& paddedvol, int offsetx, int offsety, int offsetz);
template void pad(const volume<short>& vol, volume<short>& paddedvol, int offsetx, int offsety, int offsetz);
template void pad(const volume<int>& vol, volume<int>& paddedvol, int offsetx, int offsety, int offsetz);
template void pad(const volume<float>& vol, volume<float>& paddedvol, int offsetx, int offsety, int offsetz);
template void pad(const volume<double>& vol, volume<double>& paddedvol, int offsetx, int offsety, int offsetz);

 template <class T>
  void raw_affine_transform(const volume<T>& vin, volume<T>& vout,
			    const Matrix& aff)
    {
      // NB: the size of vout MUST BE SET BEFORE IT IS PASSED IN!
      // takes the volume (vin) and applies a spatial transform, given
      //  by the 4x4 matrix (aff) in WORLD COORDINATES
      // the result is a new volume (vout)


      // Do everything in practice via the inverse transformation
      // That is, for every point in vout, calculate the pre-image in
      //  vin to which it corresponds, and interpolate vin to get the
      //  value there.
      // Also, the sampling transformations must be accounted for:
      //     T_vox1->vox2 = (T_samp2)^-1 * T_world * T_samp1

      // The sform/qform is unchanged if it is set in the output
      // The qform and sform are made equal if one is unset in the output
      // If both are unset in the output, then the sform is set to a
      //   transformed input sform (or qform if sform is unset) and the
      //   output qform is made equal to the output sform

      if (vout.nvoxels() <= 0) {
	imthrow("Attempted to use affine transform with no voxels in vout",8);
      }

      extrapolation oldex = vin.getextrapolationmethod();
      if ((oldex==boundsassert) || (oldex==boundsexception))
	{ vin.setextrapolationmethod(constpad); }

      // iaffbig goes from output mm coords to input (reference) mm coords
      Matrix iaffbig = aff.i();
      // check the left-right data orientations of the images and modify
      //   the transformation matrix to flip to radiological coords if necessary
      if (vin.left_right_order()==FSL_NEUROLOGICAL) {
	iaffbig = vin.swapmat(-1,2,3) * iaffbig;
      }
      if (vout.left_right_order()==FSL_NEUROLOGICAL) {
	iaffbig = iaffbig * vout.swapmat(-1,2,3);
      }
      // convert iaffbig to go from output voxel coords to input (reference) voxel coords
      iaffbig = vin.sampling_mat().i() * iaffbig * vout.sampling_mat();
      Matrix iaff=iaffbig.SubMatrix(1,3,1,3);

      float a11=iaff(1,1), a12=iaff(1,2), a13=iaff(1,3), a14=iaffbig(1,4),
	a21=iaff(2,1), a22=iaff(2,2), a23=iaff(2,3), a24=iaffbig(2,4),
	a31=iaff(3,1), a32=iaff(3,2), a33=iaff(3,3), a34=iaffbig(3,4), o1,o2,o3;

      // The matrix algebra below has been hand-optimized from
      //  [o1 o2 o3] = a * [x y z]  at each iteration
      for (int z=0; z<vout.zsize(); z++) {
	for (int x=0; x<vout.xsize(); x++) {
	  o1=x*a11 + z*a13 + a14;  // y=0
	  o2=x*a21 + z*a23 + a24;  // y=0
	  o3=x*a31 + z*a33 + a34;  // y=0
	  for (int y=0; y<vout.ysize(); y++) {
	    vout(x,y,z) = (T) vin.interpolate(o1,o2,o3);
	    o1 += a12;
	    o2 += a22;
	    o3 += a32;
	  }
	}
      }

      // Set the sform and qform appropriately (if set)
      // That is, copy the sform from vout if it is set, otherwise use
      //  the transformed one from vin
      // Always copy the transformed qform (if set)
      Matrix nmat;
      if ( (vout.sform_code()==NIFTI_XFORM_UNKNOWN) &&
	   (vout.qform_code()!=NIFTI_XFORM_UNKNOWN) ) {
	vout.set_sform(vout.qform_code(), vout.qform_mat());
      }
      if ( (vout.qform_code()==NIFTI_XFORM_UNKNOWN) &&
	   (vout.sform_code()!=NIFTI_XFORM_UNKNOWN) ) {
	vout.set_qform(vout.sform_code(), vout.sform_mat());
      }
      if ( (vout.qform_code()==NIFTI_XFORM_UNKNOWN) &&
	   (vout.sform_code()==NIFTI_XFORM_UNKNOWN) ) {
	if (vin.sform_code()!=NIFTI_XFORM_UNKNOWN) {
	  nmat = vin.sform_mat() * iaffbig;
	  vout.set_sform(vin.sform_code(), nmat);
	  vout.set_qform(vin.sform_code(), nmat);
	} else if (vin.qform_code()!=NIFTI_XFORM_UNKNOWN) {
	  nmat = vin.qform_mat() * iaffbig;
	  vout.set_sform(vin.qform_code(), nmat);
	  vout.set_qform(vin.qform_code(), nmat);
	}
      }

      // restore settings and return
      vin.setextrapolationmethod(oldex);
    }

template void raw_affine_transform(const volume<char>& vin, volume<char>& vout, const Matrix& aff);
template void raw_affine_transform(const volume<short>& vin, volume<short>& vout, const Matrix& aff);
template void raw_affine_transform(const volume<int>& vin, volume<int>& vout, const Matrix& aff);
template void raw_affine_transform(const volume<float>& vin, volume<float>& vout, const Matrix& aff);
template void raw_affine_transform(const volume<double>& vin, volume<double>& vout, const Matrix& aff);


 template <class T>
  void affine_transform_mask(const volume<T>& vin, volume<T>& vout,
			     const Matrix& aff, float padding, const T padval)
    {
      // padding is in voxels, not mm

      if (vout.nvoxels() <= 0) {
	imthrow("Attempted to use affine transform with no voxels in vout",8);
      }
      Matrix iaffbig = vin.sampling_mat().i() * aff.i() *
	                     vout.sampling_mat();
      Matrix iaff=iaffbig.SubMatrix(1,3,1,3);

      float a11=iaff(1,1), a12=iaff(1,2), a13=iaff(1,3), a14=iaffbig(1,4),
	a21=iaff(2,1), a22=iaff(2,2), a23=iaff(2,3), a24=iaffbig(2,4),
	a31=iaff(3,1), a32=iaff(3,2), a33=iaff(3,3), a34=iaffbig(3,4), o1,o2,o3;

      int xb=vin.xsize()-1, yb=vin.ysize()-1, zb=vin.zsize()-1;
      float xb0=-padding, yb0=-padding, zb0=-padding;
      float xb1=xb+padding, yb1=yb+padding, zb1=zb+padding;

      // The matrix algebra below has been hand-optimized from
      //  [o1 o2 o3] = a * [x y z]  at each iteration

      for (int z=0; z<vout.zsize(); z++) {
	for (int x=0; x<vout.xsize(); x++) {
	  o1=x*a11 + z*a13 + a14;  // y=0
	  o2=x*a21 + z*a23 + a24;  // y=0
	  o3=x*a31 + z*a33 + a34;  // y=0
	  for (int y=0; y<vout.ysize(); y++) {
	    if ( (o1>=xb0) && (o2>=yb0) && (o3>=zb0) &&
		 (o1<=xb1) && (o2<=yb1) && (o3<=zb1) ) {
	      // do nothing
	    } else {
	      vout(x,y,z) = padval;
	    }
	    o1 += a12;
	    o2 += a22;
	    o3 += a32;
	  }
	}
      }
    }


 template void
 affine_transform_mask(const volume<char>& vin, volume<char>& vout,
		       const Matrix& aff, float padding, const char padval);
 template void
 affine_transform_mask(const volume<short int>& vin, volume<short int>& vout,
		       const Matrix& aff, float padding, const short int padval);
 template void
 affine_transform_mask(const volume<int>& vin, volume<int>& vout,
		       const Matrix& aff, float padding, const int padval);
 template void
 affine_transform_mask(const volume<float>& vin, volume<float>& vout,
		       const Matrix& aff, float padding, const float padval);
 template void
 affine_transform_mask(const volume<double>& vin, volume<double>& vout,
		       const Matrix& aff, float padding, const double padval);



  template <class T>
  volume<T> isotropic_resample(const volume<T>& aniso, float scale)
  {
    // takes the anisotropic volume, with given sampling and resamples
    // to an isotropic scale given by scale
    if (scale<0.0) {
      cerr << "WARNING:: Negative scale in isotropic_resample - using abs value"
	   << endl;
      scale = fabs(scale);
    }
    extrapolation oldex = aniso.getextrapolationmethod();
    if ((oldex==boundsassert) || (oldex==boundsexception))
      { aniso.setextrapolationmethod(constpad); }
    float stepx, stepy, stepz;
    stepx = scale / aniso.xdim();
    stepy = scale / aniso.ydim();
    stepz = scale / aniso.zdim();
    int sx, sy, sz;
    sz = (int) Max(1.0, ( ((float)aniso.zsize()) / stepz));
    sy = (int) Max(1.0, ( ((float)aniso.ysize()) / stepy));
    sx = (int) Max(1.0, ( ((float)aniso.xsize()) / stepx));
    volume<T> iso(sx,sy,sz);
    float fx, fy, fz;
    int x, y, z;
    for (fz=0.0, z=0; z<sz; z++, fz+=stepz) {
      for (fy=0.0, y=0; y<sy; y++, fy+=stepy) {
	for (fx=0.0, x=0; x<sx; x++, fx+=stepx) {
	  iso(x,y,z) = (T)aniso.interpolate(fx,fy,fz);
	}
      }
    }
    iso.copyproperties(aniso);
    iso.setdims(scale,scale,scale);
    // transform the sform and qform matrix appropriately (if set)
    Matrix iso2aniso(4,4);
    iso2aniso = 0.0;
    iso2aniso(1,1)=stepx;
    iso2aniso(2,2)=stepy;
    iso2aniso(3,3)=stepz;
    iso2aniso(4,4)=1.0;
    if (aniso.sform_code()!=NIFTI_XFORM_UNKNOWN) {
      iso.set_sform(aniso.sform_code(), aniso.sform_mat() * iso2aniso);
    }
    if (aniso.qform_code()!=NIFTI_XFORM_UNKNOWN) {
      iso.set_qform(aniso.qform_code(), aniso.qform_mat() * iso2aniso);
    }
    aniso.setextrapolationmethod(oldex);
    return iso;
  }

template volume<char>  isotropic_resample(const volume<char>& aniso, float scale);
template volume<short> isotropic_resample(const volume<short>& aniso, float scale);
template volume<int>   isotropic_resample(const volume<int>& aniso, float scale);
template volume<float> isotropic_resample(const volume<float>& aniso, float scale);
template volume<double>isotropic_resample(const volume<double>& aniso, float scale);

  template <class T>
  int upsample_by_2(volume<T>& highresvol, const volume<T>& lowresvol,
			  bool centred)
  {
      // upsamples a volume (lowresvol) to give a new volume (highresvol)
      // note that this uses the highresvol size if it is OK but will
      //   throw away any data previous contained therein
      int sx, sy, sz;
      sz=lowresvol.zsize();
      sy=lowresvol.ysize();
      sx=lowresvol.xsize();

      extrapolation oldex = lowresvol.getextrapolationmethod();
      if ((oldex==boundsassert) || (oldex==boundsexception))
	{ lowresvol.setextrapolationmethod(constpad); }

      if (highresvol.nvoxels()<1) { highresvol.reinitialize(sx*2+1,sy*2+1,sz*2+1); }
      highresvol.copyproperties(lowresvol);
      highresvol = lowresvol.backgroundval();
      highresvol.setdims(lowresvol.xdim() / 2.0,
			 lowresvol.ydim() / 2.0,
			 lowresvol.zdim() / 2.0);
      // set sform and qform appropriately (if set)
      // voxel 2 voxel mapping of subsampled vol -> original vol
      Matrix sub2mat(4,4);
      sub2mat = IdentityMatrix(4);
      sub2mat(1,1) = 2.0;
      sub2mat(2,2) = 2.0;
      sub2mat(3,3) = 2.0;
      if (!centred) {
	sub2mat(1,4) = 0.5;
	sub2mat(2,4) = 0.5;
	sub2mat(3,4) = 0.5;
      }
      if (lowresvol.sform_code()!=NIFTI_XFORM_UNKNOWN) {
	highresvol.set_sform(lowresvol.sform_code(),lowresvol.sform_mat() * sub2mat.i());
      }
      if (lowresvol.qform_code()!=NIFTI_XFORM_UNKNOWN) {
	highresvol.set_qform(lowresvol.qform_code(),lowresvol.qform_mat() * sub2mat.i());
      }


      Matrix high2lowresvox(4,4);
      high2lowresvox = sub2mat.i();
      for (int z=0; z<highresvol.zsize(); z++) {
	for (int y=0; y<highresvol.ysize(); y++) {
	  for (int x=0; x<highresvol.xsize(); x++) {
	    // Look up the (voxel) coordinate
	    ColumnVector highrescoord(4), lowrescoord(4);
	    highrescoord << x << y << z << 1.0;
	    lowrescoord = high2lowresvox * highrescoord;
	    highresvol(x,y,z) = (T) lowresvol.interpolate(lowrescoord(1),
							  lowrescoord(2),
							  lowrescoord(3));
	  }
	}
      }
      lowresvol.setextrapolationmethod(oldex);
      return 0;
  }

  template int upsample_by_2(volume<char>& highresvol,
				      const volume<char>& lowresvol, bool centred);
  template int upsample_by_2(volume<short>& highresvol,
				      const volume<short>& lowresvol, bool centred);
  template int upsample_by_2(volume<int>& highresvol,
				      const volume<int>& lowresvol, bool centred);
  template int upsample_by_2(volume<float>& highresvol,
				      const volume<float>& lowresvol, bool centred);
  template int upsample_by_2(volume<double>& highresvol,
				      const volume<double>& lowresvol, bool centred);

  template <class T>
  volume<T> subsample_by_2(const volume<T>& refvol, bool centred)
    {
      // subsamples a volume (refvol) by blurring and subsampling to give
      //  a new volume
      // note that this creates the new volume, throwing away any data
      //  previous contained therein
      int sx, sy, sz;
      sz=refvol.zsize();
      sy=refvol.ysize();
      sx=refvol.xsize();

      extrapolation oldex = refvol.getextrapolationmethod();
      if ((oldex==boundsassert) || (oldex==boundsexception))
	{ refvol.setextrapolationmethod(constpad); }

      volume<T> halfvol((sx+1)/2,(sy+1)/2,(sz+1)/2);
      halfvol.copyproperties(refvol);
      halfvol = refvol.backgroundval();
      halfvol.setdims(refvol.xdim() * 2.0,
		      refvol.ydim() * 2.0,
		      refvol.zdim() * 2.0);
      // set sform and qform appropriately (if set)
      // voxel 2 voxel mapping of subsampled vol -> original vol
      Matrix sub2mat(4,4);
      sub2mat = IdentityMatrix(4);
      sub2mat(1,1) = 2.0;
      sub2mat(2,2) = 2.0;
      sub2mat(3,3) = 2.0;
      if (!centred) {
	sub2mat(1,4) = 0.5;
	sub2mat(2,4) = 0.5;
	sub2mat(3,4) = 0.5;
      }
      if (refvol.sform_code()!=NIFTI_XFORM_UNKNOWN) {
	halfvol.set_sform(refvol.sform_code(),refvol.sform_mat() * sub2mat);
      }
      if (refvol.qform_code()!=NIFTI_XFORM_UNKNOWN) {
	halfvol.set_qform(refvol.qform_code(),refvol.qform_mat() * sub2mat);
      }


      for (int64_t z=0, bz=0; z<halfvol.zsize(); z++, bz+=2) {
	for (int64_t y=0, by=0; y<halfvol.ysize(); y++, by+=2) {
	  for (int64_t x=0, bx=0; x<halfvol.xsize(); x++, bx+=2) {
	    // The following includes a hand-coded smoothing kernel
	    if (centred) {
	      halfvol(x,y,z) = (T) (0.125 * (refvol(bx,by,bz))
		+ 0.0625 * (refvol(bx+1,by,bz) +
			    refvol(bx-1,by,bz) +
			    refvol(bx,by+1,bz) +
			    refvol(bx,by-1,bz) +
			    refvol(bx,by,bz+1) +
			    refvol(bx,by,bz-1))
		+ 0.0312 * (refvol(bx+1,by+1,bz) +
			    refvol(bx+1,by-1,bz) +
			    refvol(bx-1,by+1,bz) +
			    refvol(bx-1,by-1,bz) +
			    refvol(bx+1,by,bz+1) +
			    refvol(bx+1,by,bz-1) +
			    refvol(bx-1,by,bz+1) +
			    refvol(bx-1,by,bz-1) +
			    refvol(bx,by+1,bz+1) +
			    refvol(bx,by+1,bz-1) +
			    refvol(bx,by-1,bz+1) +
			    refvol(bx,by-1,bz-1))
		+ 0.0156 * (refvol(bx+1,by+1,bz+1) +
			    refvol(bx+1,by+1,bz-1) +
			    refvol(bx+1,by-1,bz+1) +
			    refvol(bx+1,by-1,bz-1) +
			    refvol(bx-1,by+1,bz+1) +
			    refvol(bx-1,by+1,bz-1) +
			    refvol(bx-1,by-1,bz+1) +
			    refvol(bx-1,by-1,bz-1)) );
	    } else {
	      if (refvol.in_bounds(bx+1,by+1,bz+1)) {
		T v000,v001,v010,v011,v100,v101,v110,v111;
		refvol.getneighbours(bx,by,bz,v000,v001,v010,v011,v100,v101,v110,v111);
		halfvol(x,y,z)=(T) ((v000+v001+v010+v011+v100+v101+v110+v111)/8.0);
	      } else {
		halfvol(x,y,z) = (T) ( ( refvol(bx,by,bz) +
		  refvol(bx+1,by,bz) + refvol(bx,by+1,bz) +
		  refvol(bx,by,bz+1) + refvol(bx+1,by+1,bz) +
		  refvol(bx+1,by,bz+1) + refvol(bx,by+1,bz+1) +
					 refvol(bx+1,by+1,bz+1) )/8.0);
	      }
	    }
	  }
	}
      }
      refvol.setextrapolationmethod(oldex);
      return halfvol;
    }

template volume<char>  subsample_by_2(const volume<char>& refvol, bool centred);
template volume<short> subsample_by_2(const volume<short>& refvol, bool centred);
template volume<int>   subsample_by_2(const volume<int>& refvol, bool centred);
template volume<float> subsample_by_2(const volume<float>& refvol, bool centred);
template volume<double>subsample_by_2(const volume<double>& refvol, bool centred);



  template <class T>
  void get_axis_orientations(const volume<T>& inp1,
			     int& icode, int& jcode, int& kcode)
  {
    MISCMATHS::get_axis_orientations(inp1.sform_mat(),inp1.sform_code(),
				     inp1.qform_mat(),inp1.qform_code(),
				     icode, jcode, kcode);
  }

  template void get_axis_orientations(const volume<char>& inp1,
				      int& icode, int& jcode, int& kcode);
  template void get_axis_orientations(const volume<short int>& inp1,
				      int& icode, int& jcode, int& kcode);
  template void get_axis_orientations(const volume<int>& inp1,
				      int& icode, int& jcode, int& kcode);
  template void get_axis_orientations(const volume<float>& inp1,
				      int& icode, int& jcode, int& kcode);
  template void get_axis_orientations(const volume<double>& inp1,
				      int& icode, int& jcode, int& kcode);


 volume<float> sqrt(const volume<char>& vol)
  {
    volume<float> retvol;
    retvol=sqrt_float(vol);
    return retvol;
  }

 volume<float> sqrt(const volume<short>& vol)
  {
    volume<float> retvol;
    retvol=sqrt_float(vol);
    return retvol;
  }


 volume<float> sqrt(const volume<int>& vol)
  {
    volume<float> retvol;
    retvol=sqrt_float(vol);
    return retvol;
  }

  volume<float> sqrt(const volume<float>& vol)
  {
    volume<float> retvol;
    retvol=sqrt_float(vol);
    return retvol;
  }

  volume<double> sqrt(const volume<double>& vol)
  {
    volume<double> retvol;
    copyconvert(vol,retvol);
    for (int z=0; z<=vol.zsize(); z++) {
      for (int y=0; y<=vol.ysize(); y++) {
	for (int x=0; x<=vol.xsize(); x++) {
	  if (vol(x,y,z)>0) {
	    retvol(x,y,z) = sqrt((double) vol(x,y,z));
	  } else {
	    retvol(x,y,z) = 0;
	  }
	}
      }
    }
    return retvol;
  }

  float length(float x, float y) { return sqrt(x*x + y*y); }

  volume<float> abs(const volume<float>& realvol, const volume<float>& imagvol)
  {
      volume<float> absmap;
      absmap = realvol;
      for (int z=0; z<realvol.zsize(); z++)
	for (int y=0; y<realvol.ysize(); y++)
	  for (int x=0; x<realvol.xsize(); x++)
	    absmap(x,y,z) = length(imagvol(x,y,z),realvol(x,y,z));
      return absmap;
  }

  volume<float> phase(const volume<float>& realvol,
		      const volume<float>& imagvol)
  {
      volume<float> phasemap;
      phasemap = realvol;
      for (int z=0; z<realvol.zsize(); z++)
	for (int y=0; y<realvol.ysize(); y++)
	  for (int x=0; x<realvol.xsize(); x++)
	    phasemap(x,y,z) = atan2(imagvol(x,y,z),realvol(x,y,z));
      return phasemap;
  }


  volume<float> real(const volume<float>& absvol,
		      const volume<float>& phasevol)
    {
      volume<float> realmap;
      realmap = absvol;
      for (int z=0; z<absvol.zsize(); z++)
	for (int y=0; y<absvol.ysize(); y++)
	  for (int x=0; x<=absvol.xsize(); x++)
	    realmap(x,y,z) = absvol(x,y,z) * cos(phasevol(x,y,z));

      return realmap;
    }


  volume<float> imag(const volume<float>& absvol,
		      const volume<float>& phasevol)
    {
      volume<float> imagmap;
      imagmap = absvol;
      for (int z=0; z<=absvol.zsize(); z++)
	for (int y=0; y<=absvol.ysize(); y++)
	  for (int x=0; x<=absvol.xsize(); x++)
	    imagmap(x,y,z) = absvol(x,y,z) * sin(phasevol(x,y,z));
      return imagmap;
    }

  ///////////////////////////////////////////////////////////////////////////

  // IMAGE PROCESSING ROUTINES

  void make_grad_masks(volume<float>& maskx, volume<float>& masky,
		       volume<float>& maskz)
    {
      maskx.reinitialize(3,3,3);
      masky.reinitialize(3,3,3);
      maskz.reinitialize(3,3,3);
      for (int z=0; z<3; z++) {
	for (int y=0; y<3; y++) {
	  for (int x=0; x<3; x++) {
	    maskx(x,y,z)=(x-1.0)*pow(3.0,1.0-fabs(y-1.0)-fabs(z-1.0));
	    masky(x,y,z)=(y-1.0)*pow(3.0,1.0-fabs(x-1.0)-fabs(z-1.0));
	    maskz(x,y,z)=(z-1.0)*pow(3.0,1.0-fabs(x-1.0)-fabs(y-1.0));
	  }
	}
      }
      return;
    }

  void make_blur_mask(ColumnVector& bmask, const float final_vox_dim,
		     const float init_vox_dim)
    {
      // construct the default output
      bmask.ReSize(1);
      bmask = 1.0;
      if (fabs(init_vox_dim)<1e-8) { return; }

      float sampling_ratio = final_vox_dim / init_vox_dim;
      if (sampling_ratio < 1.1) { return; }

      float sigma = 0.85*(sampling_ratio/2.0);
      if (sigma<0.5) { return; }

      int n=((int) (sigma-0.001))*2 + 3;
      int midn = n/2 + 1;
      bmask.ReSize(n);
      for (int x=1; x<=n; x++) {
	bmask(x) = exp(-((float) Sqr(x-midn))/( Sqr(sigma) * 4.0));
      }
      bmask = bmask / Sum(bmask);
      return;
    }


  ColumnVector gaussian_kernel1D(float sigma, int radius)
    {
      ColumnVector kern(2*radius+1);
      float sum=0.0, val=0.0;

      for(int j=-radius; j<=radius; j++) {
	if (sigma>1e-6) {
	  val = exp(-(j*j)/(2.0*sigma*sigma));
	} else {
	  if (j==0) { val=1; } else { val=0; }
	}
	kern(j+radius+1) = val;
	sum += val;
      }

      kern *= (1.0/sum);
      return kern;
    }


  volume<float> gaussian_kernel2D(float sigma, int radius)
    {
      volume<float> new_kernel((2*radius+1),(2*radius+1),1);
      float sum=0.0, val=0.0;

      for(int i=-radius; i<=radius; i++) {
	for(int j=-radius; j<=radius; j++) {
	  if (sigma>1e-6) {
	    val = exp(-(i*i+j*j)/(2.0*sigma*sigma));
	  } else {
	    if ((i*i + j*j)==0) { val=1; } else { val=0; }
	  }
	  new_kernel((j+radius),(i+radius),0) = val;
	  sum += val;
	}
      }

      new_kernel *= (1.0/sum);
      return new_kernel;
    }


  volume<float> gaussian_kernel3D(float sigma, int radius)
    {
      volume<float> new_kernel((2*radius+1),(2*radius+1),(2*radius+1));
      float sum=0.0, sum2=0.0, val=0.0;

      for(int i=-radius; i<=radius; i++) {
	for(int j=-radius; j<=radius; j++) {
	  for(int k=-radius; k<=radius; k++) {
	    if (sigma>1e-6) {
	      val = exp(-(i*i+j*j+k*k)/(2.0*sigma*sigma));
	    } else {
	      if ((i*i + j*j + k*k)==0) { val=1; } else { val=0; }
	    }
	    new_kernel((j+radius),(i+radius),(k+radius)) = val;
	    sum += val;
	  }
	}
	sum2 += sum; sum=0.0;
      }

      new_kernel *= (1.0/sum2);
      return new_kernel;
    }




  volume<float> gaussian_kernel3D(float sigma, float xdim, float ydim, float zdim,float cutoff) {
  int sx = ((int) ceil(sigma*cutoff/xdim))*2 + 1;
  int sy = ((int) ceil(sigma*cutoff/ydim))*2 + 1;
  int sz = ((int) ceil(sigma*cutoff/zdim))*2 + 1;
  volume<float> vker(sx,sy,sz);
  float dx2=Sqr(xdim);
  float dy2=Sqr(ydim);
  float dz2=Sqr(zdim);
  for (int z=-sz/2; z<=sz/2; z++) {
    for (int y=-sy/2; y<=sy/2; y++) {
      for (int x=-sx/2; x<=sx/2; x++) {
	vker(x+sx/2,y+sy/2,z+sz/2)=exp(-(x*x*dx2+y*y*dy2+z*z*dz2)/(2*sigma*sigma));
      }
    }
  }
  return vker;
  }


  volume<float> spherical_kernel(float radius, float xdim, float ydim, float zdim)
  {
  int sx = MISCMATHS::round(radius/xdim)*2 + 1;
  int sy = MISCMATHS::round(radius/ydim)*2 + 1;
  int sz = MISCMATHS::round(radius/zdim)*2 + 1;
  volume<float> vker(sx,sy,sz);
  vker = 0.0;
  float dx2=Sqr(xdim);
  float dy2=Sqr(ydim);
  float dz2=Sqr(zdim);
  for (int z=-sz/2; z<=sz/2; z++) {
    for (int y=-sy/2; y<=sy/2; y++) {
      for (int x=-sx/2; x<=sx/2; x++) {
	if ((x*x*dx2+y*y*dy2+z*z*dz2)<=Sqr(radius)) {
	  vker(x+sx/2,y+sy/2,z+sz/2)=1.0;
	}
      }
    }
  }
  return vker;
  }

  volume<float> box_kernel(float length, float xdim,float ydim,float zdim)  //mm dimensions
  {
      int x = ((int) floor(length/xdim/2))*2 + 1;
      int y = ((int) floor(length/ydim/2))*2 + 1;
      int z = ((int) floor(length/zdim/2))*2 + 1;
      volume<float> new_kernel(x,y,z);
      new_kernel=1.0;
      return new_kernel;
  }




  volume<float> box_kernel(int x,int y, int z)  //voxel dimensions
  {
      volume<float> new_kernel(x,y,z);
      new_kernel=1.0;
      return new_kernel;
  }








float fsllog2(float x)
{
  // a cygwin annoyance!
  return log(x)/log(2);
}




  ///////////////////////////////////////////////////////////////////////////

  // support functions for connected components

  int find_first_nonzero(const Matrix& mat)
    {
      Tracer tr("first");
      for (int idx=1; idx<=mat.Nrows(); idx++) {
	if (mat(idx,1)!=0.0) return idx;
      }
      return -1;  // Failed
    }

  void addpair2set(int x, int y, std::vector<int>& sx, std::vector<int>& sy)
    {
      sx.push_back(x);
      sy.push_back(y);
    }


  inline void get_parent_label(int& idx, const Matrix& idxmap)
  {
    while (idxmap(idx,1)>0.0) { idx = MISCMATHS::round(float(idxmap(idx,1))); }
  }


  void relabel_components_uniquely(volume<int>& labelvol,
				   const std::vector<int>& equivlista,
				   const std::vector<int>& equivlistb, ColumnVector& clustersizes)
  {
    int labelnum = labelvol.max();
    Matrix emap(labelnum,1);
    emap = -0.2;

    int n1, n2;
    for (unsigned int n=0; n<equivlista.size(); n++) {
      n1 = equivlista[n];
      get_parent_label(n1,emap);
      n2 = equivlistb[n];
      get_parent_label(n2,emap);
      if (n1!=n2) emap(Max(n1,n2),1) = Min(n1,n2);
    }

    // re-parse emap to assign sequential, unique numbers
    int newlabel=1;
    for (int n=1; n<=labelnum; n++) {
      int n1 = n;
      get_parent_label(n1,emap);
      if (n1<n) {  // it points to another label
	emap(n,1) = emap(n1,1);
      } else {  // it is a newly found label
	emap(n,1) = -newlabel;
	newlabel++;
      }
    }

    int numclusts=newlabel-1;
    clustersizes.ReSize(numclusts);
    clustersizes=0;

    // Change the old labels to new ones

    for (int z=0; z<labelvol.zsize(); z++)
      for (int y=0; y<labelvol.ysize(); y++)
	for (int x=0; x<labelvol.xsize(); x++)
	  if (labelvol(x,y,z)>0) {
	    int tmp = MISCMATHS::round(-float(emap(labelvol(x,y,z),1)));
	    labelvol(x,y,z)=tmp;
	    clustersizes(tmp)+=1;
	  }
  }

  void relabel_components_uniquely(volume<int>& labelvol,
				   const std::vector<int>& equivlista,
				   const std::vector<int>& equivlistb){
    ColumnVector clustersize;
    relabel_components_uniquely(labelvol, equivlista, equivlistb,clustersize);


  }




bool rowentry_lessthan(const rowentry& r1, const rowentry& r2)
{
  return r1.d < r2.d ;
}

vector<offset> backConnectivity(int nDirections) {
  vector<offset> neighbours;
  if ( nDirections == 6 ) {
    neighbours.push_back(offset(-1,0,0));
    neighbours.push_back(offset(0,-1,0));
    neighbours.push_back(offset(0,0,-1));
    return neighbours;
  }
  if ( nDirections == 18 ) {
    neighbours=backConnectivity(6);
    neighbours.push_back(offset(-1,-1,0));
    neighbours.push_back(offset(-1,0,-1));
    neighbours.push_back(offset(0,1,-1));
    neighbours.push_back(offset(0,-1,-1));
    neighbours.push_back(offset(1,-1,0));
    neighbours.push_back(offset(1,0,-1));
    return neighbours;
  }
  if ( nDirections == 26 ) {
    neighbours=backConnectivity(18);
    neighbours.push_back(offset(-1,+1,-1));
    neighbours.push_back(offset(-1,-1,-1));
    neighbours.push_back(offset(+1,+1,-1));
    neighbours.push_back(offset(+1,-1,-1));
  }
  return neighbours;
}


  ///////////////////////////////////////////////////////////////////////////


template <class T>
Matrix NewimageVox2NewimageVoxMatrix(const Matrix& flirt_in2ref,
				     const volume<T>& invol,
				     const volume<T>& refvol)
{
  Matrix v2vmat, in2mm, ref2mm;
  in2mm = invol.sampling_mat();
  ref2mm = refvol.sampling_mat();
  if (invol.left_right_order() == FSL_NEUROLOGICAL) {
    in2mm = invol.swapmat(-1,2,3);
  }
  if (refvol.left_right_order() == FSL_NEUROLOGICAL) {
    ref2mm = refvol.swapmat(-1,2,3);
  }
  v2vmat = ref2mm.i() * flirt_in2ref * in2mm;
  return v2vmat;
}


template Matrix NewimageVox2NewimageVoxMatrix(const Matrix& flirt_in2ref,
					      const volume<char>& invol,
					      const volume<char>& refvol);
template Matrix NewimageVox2NewimageVoxMatrix(const Matrix& flirt_in2ref,
					      const volume<short int>& invol,
					      const volume<short int>& refvol);
template Matrix NewimageVox2NewimageVoxMatrix(const Matrix& flirt_in2ref,
					      const volume<int>& invol,
					      const volume<int>& refvol);
template Matrix NewimageVox2NewimageVoxMatrix(const Matrix& flirt_in2ref,
					      const volume<float>& invol,
					      const volume<float>& refvol);
template Matrix NewimageVox2NewimageVoxMatrix(const Matrix& flirt_in2ref,
					      const volume<double>& invol,
					      const volume<double>& refvol);

  ///////////////////////////////////////////////////////////////////////////

}
