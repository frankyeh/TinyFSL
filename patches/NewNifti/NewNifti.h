/*
  Matthew Webster (WIN@FMRIB)
  Copyright (C) 2018 University of Oxford  */
/*  CCOPYRIGHT  */
#if !defined(__newnifti_h)
#define __newnifti_h

#include <string>
#include <vector>

#include <array>
using std::array;

#include "nifti2.h"
#include "legacyFunctions.h"
#include "znzlib/znzlib.h"

namespace NiftiIO {

  enum HeaderType { analyze, nifti1, nifti2, other=-1};


  struct NiftiException : public std::exception
  {
    std::string errorMessage;
    NiftiException(const std::string& error) : errorMessage(error) {}
    ~NiftiException() throw() {}
    const char* what() const throw() { return errorMessage.c_str(); }
  };


  class NiftiExtension
  {
  public:
    int32_t    esize ;
    int32_t    ecode ;
    std::vector<char> edata ;
    size_t extensionSize() const { return( sizeof(esize) + sizeof(ecode) + edata.size() ); }
  };


  class NiftiHeader
  {
  private:
    void initialise(void);
  public:
  NiftiHeader();
  //Common Header parameters
  int sizeof_hdr;
  int64_t vox_offset;
  std::string magic;
  short datatype;
  int bitsPerVoxel;
  array<int64_t,8> dim;
  array<double,8> pixdim;
  int intentCode;
  double intent_p1;
  double intent_p2;
  double intent_p3;
  std::string intentName;
  double sclSlope;
  double sclInter;
  double cal_max;
  double cal_min;
  double sliceDuration;
  double toffset;
  int64_t sliceStart;
  int64_t sliceEnd;
  std::string description;
  std::string auxillaryFile;
  int qformCode;
  int sformCode;
  double qB;
  double qC;
  double qD;
  double qX;
  double qY;
  double qZ;
  array<double,4> sX;
  array<double,4> sY;
  array<double,4> sZ;
  int sliceCode;
  int units;
  char sliceOrdering;
  LegacyFields legacyFields;
  //Useful extras
  bool isStrictlyCompliant;
  bool wasWrongEndian;
  int bpvOfDatatype(void) const;
  const size_t datumByteWidth() const;
  std::string datatypeString() const;
  std::string fileType() const;
  int leftHanded() const { return( (pixdim[0] < 0.0) ? -1.0 : 1.0 ) ; }
  std::string niftiOrientationString( const int orientation ) const;
  std::string niftiSliceString() const;
  std::string niftiIntentString() const;
  std::string niftiTransformString(const int transform) const;
  std::string unitsString(const int units) const;
  std::string originalOrder() const;
  void sanitise();
  char freqDim() const { return sliceOrdering & 3; }
  char phaseDim() const { return (sliceOrdering >> 2) & 3; }
  char sliceDim() const { return (sliceOrdering >> 4) & 3; }
  char niftiVersion() const { return NIFTI2_VERSION(*this); }
  bool isAnalyze() const { return ( (NIFTI2_VERSION(*this) == 1) && (NIFTI_VERSION(*this) == 0) ); }
  bool singleFile() const { return NIFTI_ONEFILE(*this); }
  size_t nElements() const { size_t elements(dim[1]); for (int dims=2;dims<=dim[0];dims++) elements*=dim[dims]; return elements; }
  mat44 getQForm() const;
  void setQForm(const mat44& qForm);
  std::string qFormName() const { return niftiTransformString(qformCode);}
  mat44 getSForm() const;
  void setSForm(const mat44& sForm);
  std::string sFormName() const { return niftiTransformString(qformCode);}
  int64_t nominalVoxOffset() { return( ( singleFile() && vox_offset < 352 ) ? 352 : vox_offset ); }
  void setNiftiVersion( const char niftiVersion, const bool isSingleFile );
  void report() const;
  template<class T>
    void readAllFieldsFromRaw(const T& rawHeader);
  template<class T>
    void readCommonFieldsFromRaw(const T& rawHeader);
  template<class T>
    void readNiftiFieldsFromRaw(const T& rawNiftiHeader);
  template<class T>
    void writeAllFieldsToRaw(T& rawHeader) const;
  template<class T>
    void writeCommonFieldsToRaw(T& rawHeader) const;
  template<class T>
    void writeNiftiFieldsToRaw(T& rawNiftiHeader) const;
  };

    NiftiHeader loadExtensions(const std::string filename, std::vector<NiftiExtension>& extensions);
    NiftiHeader loadHeader(const std::string filename);
    NiftiHeader loadImage( std::string filename, char*& buffer, std::vector<NiftiExtension>& extensions, bool allocateBuffer=true);
    NiftiHeader loadImageROI( std::string filename, char*& buffer, std::vector<NiftiExtension>& extensions, int64_t xmin=-1, int64_t xmax=-1, int64_t ymin=-1, int64_t ymax=-1, int64_t zmin=-1, int64_t zmax=-1, int64_t tmin=-1, int64_t tmax=-1, int64_t d5min=-1, int64_t d5max=-1, int64_t d6min=-1, int64_t d6max=-1, int64_t d7min=-1, int64_t d7max=-1);
    void reportHeader(const analyzeHeader& header);
    void reportHeader(const nifti_1_header& header);
    void reportHeader(const nifti_2_header& header);
    void saveImage(const std::string filename, const char* buffer, const std::vector<NiftiExtension>& extensions, const NiftiHeader header, const bool useCompression=true);
    template<class T>
      void byteSwap(T& rawNiftiHeader);
    void byteSwap(const size_t elementLength, void* buffer,const unsigned long nElements=1);
    void byteSwapAnalyzeFields(analyzeHeader& rawHeader);
    template<class T>
      void byteSwapImageFields(T& rawHeader);
    template<class T>
      void byteSwapLegacyFields(T& rawHeader);
    template<class T>
      void byteSwapNiftiFields(T& rawNiftiHeader);
    template<class T>
    HeaderType headerType(const T& header);
    void reportLegacyHeader(const nifti_1_header& header);


  //fileIO
  //This class actually does the reading/writing to/from files
  //An instance _must_ be constructed with a file
  //The file will be closed on destruction of the instance
  //To open a new file, use the assignment operator, e.g.
  // instance = fileIO(newFile)
  class fileIO
  {
  public:
    bool debug;

    fileIO(const std::string& filename, const bool reading, const bool useCompression=true);
    ~fileIO() { znzclose(fileHandle); }
    fileIO& operator=(fileIO rhs);
    void reset() {znzclose(fileHandle);}
    void readData(const NiftiHeader& header,void* buffer);
    void readExtensions(const NiftiHeader& header, std::vector<NiftiExtension>& extensions );
    NiftiHeader readHeader();
    HeaderType readHeaderType();
    size_t readRawBytes( void *buffer, size_t length );
    template <class T>
      void readRawHeader(NiftiHeader& header);
    void seek(size_t nBytes, int mode) { znzseek(fileHandle, nBytes, mode); }
    void writeData(const NiftiHeader& header,const void* buffer);
    void writeExtensions(const NiftiHeader& header, const std::vector<NiftiExtension>& extensions );
    void writeHeader(const NiftiHeader& header);
    void writeRawBytes( const void *buffer, size_t length );
    template<class T>
      void writeRawHeader(const NiftiHeader& header);
  private:
    znzFile fileHandle;
  };


  //Explicit specialisations declared here
  template<> void byteSwap(nifti_1_header& rawHeader);
  template<> void byteSwap(nifti_2_header& rawHeader);
  template<> void byteSwap(analyzeHeader& rawHeader);
}

#endif
