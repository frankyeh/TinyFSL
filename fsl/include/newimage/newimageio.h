/*  General IO functions (images and transformation files)

    Mark Jenkinson and Matthew Webster, FMRIB Image Analysis Group

    Copyright (C) 2000-2012 University of Oxford  */

/*  CCOPYRIGHT  */


#if !defined(__newimageio_h)
#define __newimageio_h

#include <string>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include "NewNifti/NewNifti.h"
#include "armawrap/newmatio.h"
#include "miscmaths/miscmaths.h"
#include "newimage.h"
#include "complexvolume.h"

namespace NEWIMAGE {

bool FslIsSingleFileType(int filetype);
int FslNiftiVersionFileType(int filetype);
int FslFiletypeFromHeader(const NiftiIO::NiftiHeader& header);
int fslFileType(std::string filename);
int FslGetEnvOutputType(void);
std::string outputExtension(const int filetype);

bool FslFileExists(const std::string& filename);
bool FslImageExists(const std::string& filename);
inline bool fsl_imageexists(const std::string& filename) { return FslImageExists(filename); }
bool FslIsCompressedFileType(int filetype);
std::string return_validimagefilename(const std::string& filename, const bool quiet=false, const bool strict=false);
std::string make_basename(std::string& filename);
std::string make_basename(const std::string& filename);
std::string appendFSLfilename(const std::string inputName, const std::string addendum);
int find_pathname(std::string& filename);
int fslFileType(std::string filename);
template <class T>
void ConvertAndScaleNewNiftiBuffer(char* buffer, T*& tbuffer, const NiftiIO::NiftiHeader& niihdr, const size_t & imagesize);
  // read
template <class T>
int read_volume(volume<T>& target, const std::string& filename, const bool& legacyRead=true);
template <class T>
int read_volumeROI(volume<T>& target, const std::string& filename,
		   int64_t x0, int64_t y0, int64_t z0, int64_t x1, int64_t y1, int64_t z1);
template <class T>
int read_volumeROI(volume<T>& target, const std::string& filename,
		   int64_t x0, int64_t y0, int64_t z0, int64_t x1, int64_t y1, int64_t z1,
		   int64_t xskip, int64_t yskip, int64_t zskip);
template <class T>
int read_volumeROI(volume<T>& target, const std::string& filename,
		     int64_t x0, int64_t y0, int64_t z0, int64_t t0,
		     int64_t x1, int64_t y1, int64_t z1, int64_t t1);

template <class T>
int read_volumeROI(volume<T>& target, const std::string& filename, short& dtype,
		   int64_t x0, int64_t y0, int64_t z0, int64_t t0,
		   int64_t x1, int64_t y1, int64_t z1, int64_t t1,
		   const bool swap2radiological, const bool readAs4D=false) {
  return readGeneralVolume(target,filename,dtype,swap2radiological,x0,y0,z0,t0,-1L,-1L,-1L,x1,y1,z1,t1,-1L,-1L,-1L,readAs4D);
 }

template <class T>
int read_volumeROI(volume<T>& target, const std::string& filename,
		     int64_t x0, int64_t y0, int64_t z0, int64_t t0,
		     int64_t x1, int64_t y1, int64_t z1, int64_t t1,
		     const int64_t xskip, int64_t yskip, int64_t zskip, int64_t tskip);

int read_complexvolume(volume<float>& realvols, volume<float>& imagvols,
		       const std::string& filename,  bool read_img_data=true);
int read_complexvolume(complexvolume& vol, const std::string& filename);

template <class T>
void set_volume_properties(const NiftiIO::NiftiHeader& niihdr, volume<T>& target);


template <class T>
int read_volume_hdr_only(volume<T>& target, const std::string& filename) {
  target.destroy();
  NiftiIO::NiftiHeader niihdr;
  try {
    niihdr = NiftiIO::loadHeader(return_validimagefilename(filename));
  } catch ( std::exception& e ) { imthrow("Failed to read volume "+filename+"\nError : "+e.what(),22); }
  for (int n=1; n<=7; n++) {
    if (niihdr.dim[n]<1) niihdr.dim[n]=1;  // make it robust to dim[n]=0
  }
  T* tbuffer=new T[1];
  target.initialize(niihdr.dim[1],niihdr.dim[2],niihdr.dim[3],niihdr.dim[4],niihdr.dim[5],niihdr.dim[6],niihdr.dim[7],tbuffer,true);
  set_volume_properties(niihdr,target);
  if (!target.RadiologicalFile) target.makeradiological(true);
  return 0;
 }

template <class T>
int read_volume4D_hdr_only(volume<T>& target, const std::string& filename) {
  return read_volume_hdr_only(target,filename);
}


  // save

template <class T>
int save_volume(const volume<T>& source, const std::string& filename, const int filetype=-1);

int save_complexvolume(const volume<float>& realvol,
		       const volume<float>& imagvol, const std::string& filename);
int save_complexvolume(const complexvolume& vol, const std::string& filename);





// Helper functions
short closestTemplatedType(const short inputType);

int read_volume_size(const std::string& filename,
		     int64_t& sx, int64_t& sy, int64_t& sz, int64_t& st, int64_t& s5, int64_t& s6, int64_t& s7);

short dtype(const char* T);
short dtype(const short* T);
short dtype(const int* T);
short dtype(const float* T);
short dtype(const double* T);

short dtype(const volume<char>& vol);
short dtype(const volume<short>& vol);
short dtype(const volume<int>& vol);
short dtype(const volume<float>& vol);
short dtype(const volume<double>& vol);

short dtype(const std::string& filename);

// Boring overloads to enable different names (load and write)


// load

template <class T>
int load_volume(volume<T>& target, const std::string& filename);

// write

template <class T>
int write_volume(const volume<T>& source, const std::string& filename);


////////////////////////////////////////////////////////////////////////
///////////////////////// TEMPLATE DEFINITIONS /////////////////////////
////////////////////////////////////////////////////////////////////////


// External functions

// READ FUNCTIONS


template <class T>
int read_volumeROI(volume<T>& target, const std::string& filename,
		   int64_t x0, int64_t y0, int64_t z0, int64_t x1, int64_t y1, int64_t z1,
		   int64_t xskip, int64_t yskip, int64_t zskip)
{
  int retval=read_volumeROI(target,filename,x0,y0,z0,x1,y1,z1);
  if (retval==0) {
    if (xskip<1) xskip=1;
    if (yskip<1) yskip=1;
    if (zskip<1) zskip=1;
    int64_t sx=(target.maxx()-target.minx())/xskip + 1;
    int64_t sy=(target.maxy()-target.miny())/yskip + 1;
    int64_t sz=(target.maxz()-target.minz())/zskip + 1;
    volume<T> tmpvol(sx,sy,sz);
    int64_t xx=0, yy=0, zz=0, x=0, y=0, z=0;
    for (z=target.minz(), zz=0; z<=target.maxz(); z+=zskip, zz++) {
      for (y=target.miny(), yy=0; y<=target.maxy(); y+=yskip, yy++) {
	for (x=target.minx(), xx=0; x<=target.maxx(); x+=xskip, xx++) {
	  tmpvol(xx,yy,zz) = target(x,y,z);
	}
      }
    }
    tmpvol.copyproperties(target);
    target = tmpvol;
  }
  return retval;
}






template <class T>
int read_volumeROI(volume<T>& target, const std::string& filename,
		     int64_t x0, int64_t y0, int64_t z0,
		     int64_t x1, int64_t y1, int64_t z1)
{
  short dtype;
  return read_volumeROI(target,filename,dtype,
			x0,y0,z0,0,x1,y1,z1,0);
}


template <class T>
int read_volumeROI(volume<T>& target, const std::string& filename,
		     int64_t x0, int64_t y0, int64_t z0, int64_t t0,
		     int64_t x1, int64_t y1, int64_t z1, int64_t t1,
		     int64_t xskip, int64_t yskip, int64_t zskip, int64_t tskip)
{
  read_volumeROI(target,filename,x0,y0,z0,t0,x1,y1,z1,t1);
  if (xskip<1) xskip=1;
  if (yskip<1) yskip=1;
  if (zskip<1) zskip=1;
  if (tskip<1) tskip=1;
  int64_t sx=(target.maxx()-target.minx())/xskip + 1;
  int64_t sy=(target.maxy()-target.miny())/yskip + 1;
  int64_t sz=(target.maxz()-target.minz())/zskip + 1;
  int64_t st=(target.maxt()-target.mint())/tskip + 1;
  volume<T> tmpvol(sx,sy,sz,st);
  int64_t xx=0, yy=0, zz=0, tt=0, x=0, y=0, z=0, t=0;
  for (t=target.mint(), tt=0; t<=target.maxt(); t+=tskip, tt++) {
    for (z=target.minz(), zz=0; z<=target.maxz(); z+=zskip, zz++) {
      for (y=target.miny(), yy=0; y<=target.maxy(); y+=yskip, yy++) {
	for (x=target.minx(), xx=0; x<=target.maxx(); x+=xskip, xx++) {
	  tmpvol(xx,yy,zz,tt) = target(x,y,z,t);
	}
      }
    }
  }
  tmpvol.copyproperties(target[0]);
  target = tmpvol;
  return 0;
}

template <class T>
int read_volumeROI(volume<T>& target, const std::string& filename,
		     int64_t x0, int64_t y0, int64_t z0, int64_t t0,
		     int64_t x1, int64_t y1, int64_t z1, int64_t t1)
{
  short dtype;
  return (read_volumeROI(target,filename,dtype,
			      x0,y0,z0,t0,x1,y1,z1,t1,true));
}


template <class T>
int read_volume(volume<T>& target, const std::string& filename, const bool& legacyRead)
{
  short dtype;
  read_volumeROI(target,filename,dtype,0,0,0,0,-1,-1,-1,-1,true);
  if ( legacyRead && target.tsize() > 1 ) {
    std::cerr << "Warning: An input intended to be a single 3D volume has " <<
    "multiple timepoints. Input will be truncated to first volume, but " <<
    "this functionality is deprecated and will be removed in a future release." << std::endl;
    target=volume<T>(target[0]);
  }
  return 0;
}


// SAVE FUNCTIONS


NiftiIO::mat44 newmat2mat44(const NEWMAT::Matrix& nmat);

template <class T>
int set_fsl_hdr(const volume<T>& source, NiftiIO::NiftiHeader& niihdr)
{
  niihdr.dim[0]=source.dimensionality();
  niihdr.dim[1]=source.size1();
  niihdr.dim[2]=source.size2();
  niihdr.dim[3]=source.size3();
  niihdr.dim[4]=source.size4();
  niihdr.dim[5]=source.size5();
  niihdr.dim[6]=source.size6();
  niihdr.dim[7]=source.size7();

  niihdr.datatype=dtype(source);

  niihdr.pixdim[1]=source.pixdim1();
  niihdr.pixdim[2]=source.pixdim2();
  niihdr.pixdim[3]=source.pixdim3();
  niihdr.pixdim[4]=source.pixdim4();
  niihdr.pixdim[5]=source.pixdim5();
  niihdr.pixdim[6]=source.pixdim6();
  niihdr.pixdim[7]=source.pixdim7();


  niihdr.sformCode = source.sform_code();
  niihdr.qformCode = source.qform_code();
  niihdr.setSForm(newmat2mat44(source.sform_mat()));
  niihdr.setQForm(newmat2mat44(source.qform_mat()));

  niihdr.intentCode = source.intent_code();
  niihdr.intent_p1 = source.intent_param(1);
  niihdr.intent_p2 = source.intent_param(2);
  niihdr.intent_p3 = source.intent_param(3);

  niihdr.sclSlope = 1.0;
  niihdr.sclInter = 0.0;

  niihdr.cal_min = source.getDisplayMinimum();
  niihdr.cal_max = source.getDisplayMaximum();
  niihdr.auxillaryFile = source.getAuxFile();

  niihdr.units= NiftiIO::NIFTI_UNITS_SEC | NiftiIO::NIFTI_UNITS_MM; //NIFTI unit setting is formed by bitwise addition of defined values, in this case 10
  niihdr.sliceOrdering=0;

  return 0;
}

template <class T>
int save_basic_volume(const volume<T>& source, const std::string& filename,
		      int filetype, bool save_orig=false);

template <class T>
int save_volume(const volume<T>& source, const std::string& filename, const int filetype) {
  return save_basic_volume(source,filename,filetype,false);
}

template <class T, class dType>
struct typedSave {
  int operator()(const volume<T> source, const std::string& filename,const int filetype) {
    volume<dType> output;
    copyconvert(source,output);
    return save_volume(output,filename,filetype);
  }
};

template <class T>
struct typedSave<T,T> {
  int operator()(const volume<T> source, const std::string& filename,const int filetype) {
    return save_volume(source,filename,filetype);
  }
};

template <class T>
int save_volume_dtype(const volume<T>& source, const std::string& filename,short datatype,const int filetype=-1)
{
  switch(closestTemplatedType(datatype)) {
    case NiftiIO::DT_UNSIGNED_CHAR:  return typedSave<T,char>()(source,filename,filetype);
    case NiftiIO::DT_SIGNED_SHORT:   return typedSave<T,short>()(source,filename,filetype);
    case NiftiIO::DT_SIGNED_INT:     return typedSave<T,int>()(source,filename,filetype);
    case NiftiIO::DT_FLOAT:          return typedSave<T,float>()(source,filename,filetype);
    case NiftiIO::DT_DOUBLE:         return typedSave<T,double>()(source,filename,filetype);
    default:
      std::ostringstream errmsg;
      errmsg << "NEWIMAGE::save_volume_dtype: DT " << datatype <<  " not supported";
      perror(errmsg.str().c_str());
  }
  return -1;  // should never get here
}

template <class T>
int save_volume_and_splines(const volume<T>& source, const std::string& filename)
{
  if (!source.hasSplines()) {
    std::cerr << "NEWIMAGE::save_volume_and_splines: volume has no valid splines" << std::endl;
    return(-1);
  }
  if (source.tsize() > 1) {
    std::cerr << "NEWIMAGE::save_volume_and_splines: writing of 4D files not supported" << std::endl;
    return(-1);
  }
  volume<T> ovol(source.xsize(),source.ysize(),source.zsize(),2);
  copybasicproperties(source,ovol);
  for (int k=0; k<source.zsize(); k++) {
    for (int j=0; j<source.ysize(); j++) {
      for (int i=0; i<source.xsize(); i++) {
	ovol(i,j,k,0) = source(i,j,k);
	ovol(i,j,k,1) = source.splineCoef(i,j,k);
      }
    }
  }
  return(save_volume(ovol,filename));
}

// functions to save without doing any swapping (i.e. just as passed in)

template <class T>
int save_orig_volume(const volume<T>& source, const std::string& filename, const int filetype = -1)
{
  return save_basic_volume(source,filename,filetype,true);
}


////////////////////////////////////////////////////////////////////////
///// Boring overloads to enable different names (load and write) //////
////////////////////////////////////////////////////////////////////////

// load
template <class T>
int load_volume(volume<T>& target, const std::string& filename)
{ return read_volume(target,filename); }

 // write
template <class T>
int write_volume(const volume<T>& source, const std::string& filename)
{ return save_volume(source,filename); }

// Basic I/O functions
// read original storage order - do not swap to radiological
template <class T>
int read_orig_volume(volume<T>& target, const std::string& filename)
{
  short dtype;
  read_volumeROI(target,filename,dtype,0,0,0,0,-1,-1,-1,-1,false);
  return 0;
}


//TODO REMOVE after 4D is dead
template <class T>
int write_volume4D(const volume<T>& source, const std::string& filename) {
  return save_volume(source,filename); }
template <class T>
int save_volume4D(const volume<T>& source, const std::string& filename) {
  return save_volume(source,filename); }
template <class T>
int load_volume4D(volume<T>& source, const std::string& filename) {
  return read_volume(source,filename,false); }
template <class T>
int read_volume4D(volume<T>& source, const std::string& filename) {
  return read_volume(source,filename,false); }
template <class T>
int read_orig_volume4D(volume<T>& target, const std::string& filename) {
  return read_orig_volume(target,filename);}

}

#endif
