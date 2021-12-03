/*  General IO functions (images and transformation files)

    Mark Jenkinson and Matthew Webster, FMRIB Image Analysis Group

    Copyright (C) 1999-2008 University of Oxford  */

/*  CCOPYRIGHT  */
#include <sys/stat.h>
#include "newimageio.h"
#include "miscmaths/miscmaths.h"
#include "armawrap/newmat.h"
#include "NewNifti/NewNifti.h"


using namespace std;
using namespace NEWMAT;
using namespace MISCMATHS;
using namespace NiftiIO;
using namespace NiftiIO::legacy;

namespace NEWIMAGE {

////////////////////////////////////////////////////////////////////////////

class imageExtensions {
public:
  static const int size=4;
  static const string extension[size];
};

const string imageExtensions::extension[imageExtensions::size]={".nii.gz",".nii",".hdr",".hdr.gz"};

bool FslIsSingleFileType(int filetype)
{
  if ( filetype % 100 >= 10 )
    return false;
  return true;
}


bool FslIsCompressedFileType(int filetype)
{
  if ( filetype >=100 ) return true;
  return false;
}

int FslNiftiVersionFileType(int filetype) //Note that this relies on the definitions in Newimage.h, if the numbering scheme
{                                         //changes this may become invalid
  return filetype %100 %10 %3;
}

int FslFiletypeFromHeader(const NiftiHeader& header)
{
  int filetype(0);
  filetype+=header.niftiVersion();
  if ( !header.singleFile() )
    filetype+=10;
  return filetype;
}

int fslFileType(string filename)
{
  filename=return_validimagefilename(filename);
  NiftiHeader header = loadHeader(filename);
  int filetype(FslFiletypeFromHeader(header));
  if ( filename.substr(filename.size()-3,3) == ".gz" )
    filetype+=100;
  return filetype;
}

bool FslFileExists(const string& filename) {
  try {
    return_validimagefilename(filename,true);
  } catch(...) {
    return false;
  }
  return true;
}

bool FslImageExists(const string& filename) {
  return FslFileExists(filename);
}

int FslGetEnvOutputType(void)
{
  return FSL_TYPE_NIFTI_GZ;
}

string outputExtension(const int filetype)
{
  if ( filetype == FSL_TYPE_NIFTI || filetype == FSL_TYPE_NIFTI2 )
    return ".nii";
  if ( filetype == FSL_TYPE_NIFTI_GZ || filetype == FSL_TYPE_NIFTI2_GZ )
    return ".nii.gz";
  if ( filetype == FSL_TYPE_ANALYZE || filetype == FSL_TYPE_NIFTI_PAIR || filetype == FSL_TYPE_NIFTI2_PAIR )
    return ".hdr";
  if ( filetype == FSL_TYPE_NIFTI_PAIR_GZ || filetype == FSL_TYPE_NIFTI2_PAIR_GZ || filetype == FSL_TYPE_ANALYZE_GZ )
    return ".hdr.gz";
  return "";
}


int NiftiGetLeftRightOrder(const NiftiHeader& niihdr)
{
  int order=FSL_RADIOLOGICAL, sform_code, qform_code;
  mat44 sform44, qform44;
  sform_code = niihdr.sformCode;
  qform_code = niihdr.qformCode;
  sform44 = niihdr.getSForm();
  qform44 = niihdr.getQForm();
  // Determines if the image is stored in neurological or radiological convention
  float dets=-1.0, detq=-1.0, det=-1.0;
  mat33 sform33, qform33;
  if (qform_code!=NIFTI_XFORM_UNKNOWN) {
    qform33 = mat44_to_mat33(qform44);
    detq = nifti_mat33_determ(qform33);
    det = detq;
  }
  if (sform_code!=NIFTI_XFORM_UNKNOWN) {
    sform33 = mat44_to_mat33(sform44);
    dets = nifti_mat33_determ(sform33);
    det = dets;
  }

  if (det<0.0) order=FSL_RADIOLOGICAL;
  else order=FSL_NEUROLOGICAL;
  // check for inconsistency if both are set
  if ( (sform_code!=NIFTI_XFORM_UNKNOWN) &&
       (qform_code!=NIFTI_XFORM_UNKNOWN) ) {
    if (dets * detq < 0.0) order=FSL_INCONSISTENT;
    if (fabs(dets * detq)<1e-12)  order=FSL_ZERODET;
  }
  if (fabs(det)<1e-12) order=FSL_ZERODET;
  return order;
}


// VOLUME I/O
template <class T>
void set_volume_properties(const NiftiHeader& niihdr, volume<T>& target)
{
  target.setdims(niihdr.pixdim[1],niihdr.pixdim[2],niihdr.pixdim[3],niihdr.pixdim[4],
		 niihdr.pixdim[5],niihdr.pixdim[6],niihdr.pixdim[7]);
  int sform_code, qform_code;
  mat44 smat, qmat;
  qmat = niihdr.getQForm();
  qform_code = niihdr.qformCode;
  smat = niihdr.getSForm();
  sform_code = niihdr.sformCode;
  Matrix snewmat(4,4), qnewmat(4,4);
  for (int i=1; i<=4; i++) {
    for (int j=1; j<=4; j++) {
      snewmat(i,j) = smat.m[i-1][j-1];
      qnewmat(i,j) = qmat.m[i-1][j-1];
    }
  }
  target.set_sform(sform_code,snewmat);
  target.set_qform(qform_code,qnewmat);
  target.RadiologicalFile = (NiftiGetLeftRightOrder(niihdr)==FSL_RADIOLOGICAL);

  target.set_intent(niihdr.intentCode,niihdr.intent_p1,niihdr.intent_p2,niihdr.intent_p3);
  target.setDisplayMinimum(niihdr.cal_min);
  target.setDisplayMaximum(niihdr.cal_max);
  target.setAuxFile(niihdr.auxillaryFile);
}

template void set_volume_properties(const NiftiHeader& niihdr, volume<char>& target);
template void set_volume_properties(const NiftiHeader& niihdr, volume<short>& target);
template void set_volume_properties(const NiftiHeader& niihdr, volume<int>& target);
template void set_volume_properties(const NiftiHeader& niihdr, volume<float>& target);
template void set_volume_properties(const NiftiHeader& niihdr, volume<double>& target);


int read_volume_size(const string& filename,
		     int64_t& sx, int64_t& sy, int64_t& sz, int64_t& st, int64_t& s5, int64_t& s6, int64_t& s7)
{
  // read in sizes only
  NiftiHeader niihdr = loadHeader(filename);
  sx=niihdr.dim[1];
  sy=niihdr.dim[2];
  sz=niihdr.dim[3];
  st=niihdr.dim[4];
  s5=niihdr.dim[5];
  s6=niihdr.dim[6];
  s7=niihdr.dim[7];

  return 0;
}

template <class T>
int readGeneralVolume(volume<T>& target, const string& filename,
		  short& dtype, const bool swap2radiological,
		  int64_t x0, int64_t y0, int64_t z0, int64_t t0, int64_t d50, int64_t d60, int64_t d70,
		  int64_t x1, int64_t y1, int64_t z1, int64_t t1, int64_t d51, int64_t d61, int64_t d71,
		  const bool readAs4D)
{
  // to get the whole volume use x0=y0=z0=t0=0 and x1=y1=z1=t1=-1
  // NB: coordinates are in "radiological" convention when swapping (i.e.
  ///    *not* the same as nifti/fslview), or untouched otherwise
  target.destroy();

  NiftiHeader header;
  char *buffer;
  try {
    header = loadImageROI(return_validimagefilename(filename),buffer,target.extensions,x0,x1,y0,y1,z0,z1,t0,t1,d50,d51,d60,d61,d70,d71);
  } catch ( exception& e ) { imthrow("Failed to read volume "+filename+"\nError : "+e.what(),22); }
  // sanity check stuff (well, forcing sanity really)
  if ( header.isAnalyze() ) {
    header.sX[0]=header.pixdim[1];
    header.sY[1]=header.pixdim[2];
    header.sZ[2]=header.pixdim[3];
    header.sX[3]=-(header.legacyFields.origin()[0]-1)*header.pixdim[1];
    header.sY[3]=-(header.legacyFields.origin()[1]-1)*header.pixdim[2];
    header.sZ[3]=-(header.legacyFields.origin()[2]-1)*header.pixdim[3];
    header.setQForm(header.getSForm());
    header.qformCode=header.sformCode=NIFTI_XFORM_ALIGNED_ANAT;
  }
  for ( int i = 1; i <= header.dim[0]; i++ ) //pixheader.dim 1..dim[0] must be +ve for NIFTI
    header.pixdim[i] = header.pixdim[i] == 0 ? 1 : fabs(header.pixdim[i]);
  // allocate and fill buffer with required data
  T* tbuffer;
  ConvertAndScaleNewNiftiBuffer(buffer,tbuffer,header,header.nElements());  // buffer will get deleted inside (unless T=char)
  if (tbuffer==NULL)
    cout << "help" << endl;
  target.initialize(header.dim[1],header.dim[2],header.dim[3],header.dim[4],header.dim[5],header.dim[6],header.dim[7],tbuffer,true);
  // copy info from file
  set_volume_properties(header,target);
  // return value gives info about file datatype
  dtype = header.datatype;
  // swap to radiological if necessary
  if (swap2radiological && !target.RadiologicalFile) target.makeradiological();
  //TODO if readAs4D use the 5Dto4D method that we _will_ write
  return 0;
}

template int read_volumeROI(volume<char>& target, const string& filename,
			      short& dtype,
			      int64_t x0, int64_t y0, int64_t z0, int64_t t0,
			      int64_t x1, int64_t y1, int64_t z1, int64_t t1,
			      const bool swap2radiological, const bool readAs4D);
template int read_volumeROI(volume<short>& target, const string& filename,
			      short& dtype,
			      int64_t x0, int64_t y0, int64_t z0, int64_t t0,
			      int64_t x1, int64_t y1, int64_t z1, int64_t t1,
			      const bool swap2radiological, const bool readAs4D);
template int read_volumeROI(volume<int>& target, const string& filename,
			      short& dtype,
			      int64_t x0, int64_t y0, int64_t z0, int64_t t0,
			      int64_t x1, int64_t y1, int64_t z1, int64_t t1,
			      const bool swap2radiological, const bool readAs4D);
template int read_volumeROI(volume<float>& target, const string& filename,
			      short& dtype,
			      int64_t x0, int64_t y0, int64_t z0, int64_t t0,
			      int64_t x1, int64_t y1, int64_t z1, int64_t t1,
			      const bool swap2radiological, const bool readAs4D);
template int read_volumeROI(volume<double>& target, const string& filename,
			      short& dtype,
			      int64_t x0, int64_t y0, int64_t z0, int64_t t0,
			      int64_t x1, int64_t y1, int64_t z1, int64_t t1,
			      const bool swap2radiological, const bool readAs4D);

template int readGeneralVolume(volume<char>& target, const string& filename,
                               short& dtype, const bool swap2radiological,
                               int64_t x0, int64_t y0, int64_t z0, int64_t t0, int64_t d50, int64_t d60, int64_t d70,
                               int64_t x1, int64_t y1, int64_t z1, int64_t t1, int64_t d51, int64_t d61, int64_t d71,
                               const bool readAs4D);
template int readGeneralVolume(volume<short>& target, const string& filename,
                               short& dtype, const bool swap2radiological,
                               int64_t x0, int64_t y0, int64_t z0, int64_t t0, int64_t d50, int64_t d60, int64_t d70,
                               int64_t x1, int64_t y1, int64_t z1, int64_t t1, int64_t d51, int64_t d61, int64_t d71,
                               const bool readAs4D);
template int readGeneralVolume(volume<int>& target, const string& filename,
                               short& dtype, const bool swap2radiological,
                               int64_t x0, int64_t y0, int64_t z0, int64_t t0, int64_t d50, int64_t d60, int64_t d70,
                               int64_t x1, int64_t y1, int64_t z1, int64_t t1, int64_t d51, int64_t d61, int64_t d71,
                               const bool readAs4D);
template int readGeneralVolume(volume<float>& target, const string& filename,
                               short& dtype, const bool swap2radiological,
                               int64_t x0, int64_t y0, int64_t z0, int64_t t0, int64_t d50, int64_t d60, int64_t d70,
                               int64_t x1, int64_t y1, int64_t z1, int64_t t1, int64_t d51, int64_t d61, int64_t d71,
                               const bool readAs4D);
template int readGeneralVolume(volume<double>& target, const string& filename,
                               short& dtype, const bool swap2radiological,
                               int64_t x0, int64_t y0, int64_t z0, int64_t t0, int64_t d50, int64_t d60, int64_t d70,
                               int64_t x1, int64_t y1, int64_t z1, int64_t t1, int64_t d51, int64_t d61, int64_t d71,
                               const bool readAs4D);

template <class V>
int save_unswapped_vol(const V& source, const string& filename, int filetype,int bitsPerVoxel)
{
  NiftiHeader header;
  set_fsl_hdr(source,header);
  if ( filetype<0 )
    filetype=FslGetEnvOutputType();
  header.setNiftiVersion(FslNiftiVersionFileType(filetype),FslIsSingleFileType(filetype));
  header.bitsPerVoxel=bitsPerVoxel;
  if ( header.isAnalyze() ) {
    if (header.sformCode != NIFTI_XFORM_UNKNOWN) {
      mat44 inverse=nifti_mat44_inverse(header.getSForm());
      for ( unsigned int i = 0; i < 3; i++)
        header.legacyFields.origin()[i]=(short) inverse.m[i][3] + 1;
    } else if (header.qformCode != NIFTI_XFORM_UNKNOWN) {
      mat44 inverse=nifti_mat44_inverse(header.getQForm());
      for ( unsigned int i = 0; i < 3; i++)
        header.legacyFields.origin()[i]=(short) inverse.m[i][3] + 1;
    }
    header.pixdim[1]*=-1;
  }
  NiftiIO::saveImage(make_basename(filename)+outputExtension(filetype), (const char *)source.fbegin(), source.extensions, header, FslIsCompressedFileType(filetype));
  return 0;
}

template <class T>
int save_basic_volume(const volume<T>& source, const string& filename,
			int filetype, bool noSwapping)
{
  if (source.tsize()<1) return -1;
  bool currently_rad = source.left_right_order()==FSL_RADIOLOGICAL;
  if (!noSwapping && !source.RadiologicalFile && currently_rad)  const_cast< volume <T>& > (source).makeneurological();
  save_unswapped_vol(source,filename,filetype,sizeof(T)*8);
  if (!noSwapping && !source.RadiologicalFile && currently_rad)  const_cast< volume <T>& > (source).makeradiological();
  return 0;
}

template int save_basic_volume(const volume<char>& source, const string& filename,
				 int filetype, bool save_orig);
template int save_basic_volume(const volume<short>& source, const string& filename,
				 int filetype, bool save_orig);
template int save_basic_volume(const volume<int>& source, const string& filename,
				 int filetype, bool save_orig);
template int save_basic_volume(const volume<float>& source, const string& filename,
				 int filetype, bool save_orig);
template int save_basic_volume(const volume<double>& source, const string& filename,
				 int filetype, bool save_orig);

mat44 newmat2mat44(const Matrix& nmat)
{
  mat44 ret;
  for (int i=1; i<=4; i++) {
    for (int j=1; j<=4; j++) {
      ret.m[i-1][j-1] = nmat(i,j);
    }
  }
  return ret;
}


std::string make_basename(std::string &filename) //note passed as reference for compat with old API - use as a return type is preferred
{
    for (int i = 0; i < imageExtensions::size; ++i)
    {
        if (filename.length() >= imageExtensions::extension[i].length() && filename.compare(filename.length() - imageExtensions::extension[i].length(), imageExtensions::extension[i].length(), imageExtensions::extension[i]) == 0) {
            filename.erase(filename.length() - imageExtensions::extension[i].length(), imageExtensions::extension[i].length());
            break;
        }
    }
    return (filename);
}

string make_basename(const string& inputName)
{
  string filename(inputName);
  return(make_basename(filename));
}

string appendFSLfilename(const string inputName, const string addendum) {
  return make_basename(inputName)+addendum;
}


bool valid_imagefilename(const string& filename)
{
  //return boost::filesystem::exists(filename);
  struct stat buf;
  return (stat(filename.c_str(), &buf) == 0);
}

string return_validimagefilename(const string& filename, const bool quiet, const bool strict) {
  string bname(make_basename(filename)), validname="";
  int validcount=0;
  if (bname != filename) {
    if (valid_imagefilename(filename))
      return filename; //If the original filename had a full image extension and exists, return it.
    else if ( strict )
      imthrow(filename+" is not an image.",62,quiet); //Strict doesn't search for next-closest match
  }
  //Look at possible extensions ( if filename is full, this will effectively repeat the test above )
  for ( int i=0; i < imageExtensions::size && validcount < 2; ++i )
    if (valid_imagefilename(bname+imageExtensions::extension[i])) { validname=bname+imageExtensions::extension[i]; validcount++; }
  if (validcount>1)
    imthrow("Multiple possible filenames detected for basename: "+bname,61,quiet);
  if (validcount==0)
    imthrow("No image files match: "+bname,63,quiet);
  return validname;
}

int find_pathname(string& filename)
{
  if (filename.size() < 1) return -1;
  string pathname = filename;
  int fsize = pathname.length(), indx;

  // working backwards, find '/' and remove everything after it

  indx = fsize-1;
  while ((pathname[indx] != '/') && (indx != 0))
    indx--;

  if (indx<fsize-1)
    pathname.erase(indx+1);

  filename = pathname;
  return 0;
}

short closestTemplatedType(const short inputType)
{
  switch (inputType) {
  case DT_UNSIGNED_CHAR:
  case DT_INT8:
    return DT_UNSIGNED_CHAR;
  case DT_SIGNED_SHORT:
    return DT_SIGNED_SHORT;
  case DT_SIGNED_INT:
  case DT_UINT16:
    return DT_SIGNED_INT;
  case DT_FLOAT:
  case DT_UINT32:
  case DT_INT64:
  case DT_UINT64:
    return DT_FLOAT;
  case DT_DOUBLE:
  case DT_FLOAT128:
    return DT_DOUBLE;
  case DT_COMPLEX:
    cerr << "COMPLEX not supported as an independent type" << endl;
    return -1;
  default:
    cerr << "Datatype " << inputType << " is NOT supported - please check your image" << endl;
    return -1;
  }
}

short dtype(const char* ptr)   { return DT_UNSIGNED_CHAR; }
short dtype(const short* ptr)  { return DT_SIGNED_SHORT; }
short dtype(const int* ptr)    { return DT_SIGNED_INT; }
short dtype(const float* ptr)  { return DT_FLOAT; }
short dtype(const double* ptr) { return DT_DOUBLE; }

short dtype(const volume<char>& vol)   { return DT_UNSIGNED_CHAR; }
short dtype(const volume<short>& vol)  { return DT_SIGNED_SHORT; }
short dtype(const volume<int>& vol)    { return DT_SIGNED_INT; }
short dtype(const volume<float>& vol)  { return DT_FLOAT; }
short dtype(const volume<double>& vol) { return DT_DOUBLE; }

short dtype(const string& filename)
{
  if ( filename.empty() ) return -1;
  NiftiHeader niihdr = NiftiIO::loadHeader(return_validimagefilename(filename));
  if ( niihdr.sclSlope != 1.0 || niihdr.sclInter != 0.0 ) {
    if ( niihdr.sclSlope == 0.0 || niihdr.datatype == DT_DOUBLE)
      return niihdr.datatype;
    return DT_FLOAT;
  }
  return niihdr.datatype;
}


template <class T>
void ConvertAndScaleNewNiftiBuffer(char* buffer, T*& tbuffer, const NiftiHeader& niihdr, const size_t & nElements)
{
  short originalType = niihdr.datatype;
  float slope = niihdr.sclSlope, intercept = niihdr.sclInter;
  if (fabs(slope)<1e-30) {
    slope = 1.0;
    intercept = 0.0;
  }
  bool doscaling( (fabs(slope - 1.0)>1e-30) || (fabs(intercept)>1e-30) );
  // create buffer pointer of the desired type and allocate if necessary (scaling or not)
  tbuffer = dtype(tbuffer) == originalType ?  (T*) buffer : new T[nElements] ;

  if ( (dtype(tbuffer) != originalType) || doscaling ) {
    switch(originalType) {
      case DT_SIGNED_SHORT:   convertbuffer((short *) buffer,tbuffer,nElements,slope,intercept);
	                            break;
      case DT_UNSIGNED_CHAR:  convertbuffer((unsigned char *) buffer,tbuffer,nElements,slope,intercept);
	                            break;
      case DT_SIGNED_INT:     convertbuffer((int *) buffer,tbuffer,nElements,slope,intercept);
	                            break;
      case DT_FLOAT:          convertbuffer((float *) buffer,tbuffer,nElements,slope,intercept);
	                            break;
      case DT_DOUBLE:         convertbuffer((double *) buffer,tbuffer,nElements,slope,intercept);
                            	break;
	/*------------------- new codes for NIFTI ---*/
      case DT_INT8:           convertbuffer((signed char *) buffer,tbuffer,nElements,slope,intercept);
                              break;
      case DT_UINT16:         convertbuffer((unsigned short *) buffer,tbuffer,nElements,slope,intercept);
	                            break;
      case DT_UINT32:         convertbuffer((unsigned int *) buffer,tbuffer,nElements,slope,intercept);
                              break;
      case DT_INT64:          convertbuffer((long signed int *) buffer,tbuffer,nElements,slope,intercept);
                              break;
      case DT_UINT64:         convertbuffer((long unsigned int *) buffer,tbuffer,nElements,slope,intercept);
                            	break;
      default:
	  /* includes: DT_BINARY, DT_RGB, DT_ALL, DT_FLOAT128, DT_COMPLEX's */
	                            delete [] tbuffer;
	                            delete [] buffer;
	                            imthrow("Fslread: DT " + num2str(originalType) + " not supported",8);
    }
  }
  // delete old buffer *ONLY* if a new buffer has been allocated
  if (dtype(tbuffer) != originalType)  delete[] buffer;
}

//////////////////////////////////////////////////////////////////////////

// COMPLEX IMAGE I/O
int read_complexvolume(volume<float>& realvols, volume<float>& imagvols,
			 const string& filename, bool read_img_data)
{
  if ( filename.size()<1 ) return -1;
  NiftiHeader niihdr;
  vector<NiftiExtension> extensions;

  try {
    niihdr = NiftiIO::loadHeader(return_validimagefilename(filename));
  } catch ( exception& e ) { imthrow("Failed to read volume "+filename+"\nError : "+e.what(),22); }
  bool isComplex( niihdr.datatype == DT_COMPLEX );

  int64_t sx(niihdr.dim[1]),sy(niihdr.dim[2]),sz(niihdr.dim[3]),st(std::max(niihdr.dim[4],(int64_t)1));
  size_t volsize=sx*sy*sz*st;

  float *rbuffer(NULL), *ibuffer(NULL);
  char *buffer;
  if ( read_img_data) {
    rbuffer=new float[volsize];
    if (rbuffer==0)
      imthrow("Out of memory",99);
    if ( isComplex ) {
      ibuffer=new float[volsize];
      if (ibuffer==0)
	      imthrow("Out of memory",99);
    }
    niihdr = NiftiIO::loadImageROI(return_validimagefilename(filename),buffer,extensions);
    for ( size_t voxel=0;voxel<volsize && !isComplex;voxel++) //Only copy real buffer for non-complex
      rbuffer[voxel]=((float*)buffer)[voxel];
    for ( size_t voxel=0;voxel<volsize && isComplex;voxel++) {
      rbuffer[voxel]=((float *)buffer)[2*voxel];
    ibuffer[voxel]=((float *)buffer)[2*voxel+1];
    }
    realvols.reinitialize(sx,sy,sz,st,rbuffer,true);
    imagvols.reinitialize(sx,sy,sz,st,ibuffer,true);
  }

  realvols.setdims(niihdr.pixdim[1],niihdr.pixdim[2],niihdr.pixdim[3],niihdr.pixdim[4]);
  imagvols.setdims(niihdr.pixdim[1],niihdr.pixdim[2],niihdr.pixdim[3],niihdr.pixdim[4]);
  // swap to Radiological when necessary
  if ( NiftiGetLeftRightOrder(niihdr) != FSL_RADIOLOGICAL ) {
    realvols.RadiologicalFile = false;
    realvols.makeradiological();
    imagvols.RadiologicalFile = false;
    imagvols.makeradiological();
  } else {
    realvols.RadiologicalFile = true;
    imagvols.RadiologicalFile = true;
  }
  delete [] buffer;
  return 0;
}

int read_complexvolume(complexvolume& vol, const string& filename)
{ return read_complexvolume(vol.re(),vol.im(),filename,true); }

int save_complexvolume(const volume<float>& realvols, const volume<float>& imagvols, const string& filename)
{
  NiftiHeader niihdr;
  if (realvols.tsize()<=0) return -1;
  // convert back to Neurological if necessary
  if (!realvols.RadiologicalFile) { const_cast< volume <float>& > (realvols).makeneurological(); }
  if (!imagvols.RadiologicalFile) { const_cast< volume <float>& > (imagvols).makeneurological(); }
  set_fsl_hdr(realvols,niihdr);
  niihdr.datatype=DT_COMPLEX;
  int filetype=FslGetEnvOutputType();
  niihdr.setNiftiVersion(FslNiftiVersionFileType(filetype),FslIsSingleFileType(filetype));
  niihdr.bitsPerVoxel=sizeof(double)*8;
  float *buffer=new float[ 2*realvols.totalElements() ];
  volume<float>::fast_const_iterator rit(realvols.fbegin()), iit(imagvols.fbegin());
  for ( size_t voxel=0;voxel<(size_t)realvols.totalElements();++voxel) {
    buffer[2*voxel]=*rit++;
    buffer[2*voxel+1]=*iit++;
  }
  NiftiIO::saveImage(make_basename(filename)+outputExtension(filetype), (const char *)buffer, realvols.extensions, niihdr, FslIsCompressedFileType(filetype));
  // restore to original ?
  if (!realvols.RadiologicalFile) { const_cast< volume <float>& > (realvols).makeradiological(); }
  if (!imagvols.RadiologicalFile) { const_cast< volume <float>& > (imagvols).makeradiological(); }
  return 0;
}

int save_complexvolume(const complexvolume& vol, const string& filename)
{ return save_complexvolume(vol.re(),vol.im(),filename); }

//////////////////////////////////////////////////////////////////////////
}
