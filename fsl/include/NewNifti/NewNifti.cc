/*
  Matthew Webster (WIN@FMRIB)
  Copyright (C) 2018 University of Oxford  */
/*  CCOPYRIGHT  */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctype.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>

#include "NewNifti.h"

using namespace std;
using namespace NiftiIO::legacy;

#define LSB_FIRST 1
#define MSB_FIRST 2

namespace NiftiIO {

  int systemByteOrder(void)   /* determine CPU's byte order */
  {
    union {
      unsigned char c[2] ;
      short         s    ;
    } testUnion ;

    testUnion.c[0] = 1 ;
    testUnion.c[1] = 0 ;

    return (testUnion.s == 1) ? LSB_FIRST : MSB_FIRST;
  }


  void NiftiHeader::initialise(void) {
    wasWrongEndian=false;
    fill(dim.begin(),dim.end(),0);
    fill(pixdim.begin(),pixdim.end(),0);
    sclSlope=1.0;
    cal_max=cal_min=intent_p1=intent_p2=intent_p3=0.0;
    datatype=sizeof_hdr=vox_offset=bitsPerVoxel=intentCode=sliceOrdering=0;
    sclInter=sliceStart=sliceEnd=sliceDuration=sliceCode=units=toffset=0;
    qformCode=sformCode=qB=qC=qD=qX=qY=qZ=0;
    fill(sX.begin(),sX.end(),0);
    fill(sY.begin(),sY.end(),0);
    fill(sZ.begin(),sZ.end(),0);
    description=string(80,'\0');
    auxillaryFile=string(24,'\0');
    intentName=string(16,'\0');
    magic=string(4,'\0');
  }


  void NiftiHeader::sanitise(void) {
    for(char axis(1); axis<=dim[0]; axis++)
      pixdim[axis] = pixdim[axis] > 0 ? pixdim[axis] : 1;
  }


  NiftiHeader::NiftiHeader() :
    isStrictlyCompliant(false) {
    initialise();
  }


  int NiftiHeader::bpvOfDatatype(void) const {
    switch( datatype ) {
      case DT_UNSIGNED_CHAR:
      case DT_INT8:
        return 8;
      case DT_SIGNED_SHORT:
      case DT_UINT16:
        return 16;
      case DT_SIGNED_INT:
      case DT_FLOAT:
      case DT_UINT32:
        return 32;
      case DT_INT64:
      case DT_UINT64:
      case DT_DOUBLE:
      case DT_COMPLEX:
        return 64;
      case DT_FLOAT128:
        return 128;
    }
    return 0;
  }


  //According to the NIFTI standard, the bitpix header field is used to
  //determine voxel data width but it _must_ the be same as the width
  //implied by datatype, implying datatype is the canonical field.
  //Some APIs have been observed which correctly set datatype but not
  //bitpix, so this method returns the datatype bit-width, but will throw
  //for inconsistent headers when the reader has strict compliance set.
  const size_t NiftiHeader::datumByteWidth() const{
    if ( isStrictlyCompliant && bpvOfDatatype() != bitsPerVoxel )
      throw NiftiException("Error: Header fields datatype and bitpix are inconsistent.");
    return bpvOfDatatype()/8;
  }


  string NiftiHeader::niftiOrientationString( const int orientation ) const
  {
    switch( orientation ) {
      case NIFTI_L2R: return "Left-to-Right" ;
      case NIFTI_R2L: return "Right-to-Left" ;
      case NIFTI_P2A: return "Posterior-to-Anterior" ;
      case NIFTI_A2P: return "Anterior-to-Posterior" ;
      case NIFTI_I2S: return "Inferior-to-Superior" ;
      case NIFTI_S2I: return "Superior-to-Inferior" ;
    }
    return "Unknown" ;
  }


  string NiftiHeader::niftiSliceString() const {
    switch( sliceCode ){
      case NIFTI_SLICE_SEQ_INC:  return "sequential_increasing";
      case NIFTI_SLICE_SEQ_DEC:  return "sequential_decreasing";
      case NIFTI_SLICE_ALT_INC:  return "alternating_increasing";
      case NIFTI_SLICE_ALT_DEC:  return "alternating_decreasing";
      case NIFTI_SLICE_ALT_INC2: return "alternating_increasing_2";
      case NIFTI_SLICE_ALT_DEC2: return "alternating_decreasing_2";
    }
    return "Unknown";
  }


  string NiftiHeader::niftiIntentString() const {
    switch( intentCode ){
      case NIFTI_INTENT_CORREL:     return "Correlation statistic";
      case NIFTI_INTENT_TTEST:      return "T-statistic";
      case NIFTI_INTENT_FTEST:      return "F-statistic";
      case NIFTI_INTENT_ZSCORE:     return "Z-score";
      case NIFTI_INTENT_CHISQ:      return "Chi-squared distribution";
      case NIFTI_INTENT_BETA:       return "Beta distribution";
      case NIFTI_INTENT_BINOM:      return "Binomial distribution";
      case NIFTI_INTENT_GAMMA:      return "Gamma distribution";
      case NIFTI_INTENT_POISSON:    return "Poisson distribution";
      case NIFTI_INTENT_NORMAL:     return "Normal distribution";
      case NIFTI_INTENT_FTEST_NONC: return "F-statistic noncentral";
      case NIFTI_INTENT_CHISQ_NONC: return "Chi-squared noncentral";
      case NIFTI_INTENT_LOGISTIC:   return "Logistic distribution";
      case NIFTI_INTENT_LAPLACE:    return "Laplace distribution";
      case NIFTI_INTENT_UNIFORM:    return "Uniform distribition";
      case NIFTI_INTENT_TTEST_NONC: return "T-statistic noncentral";
      case NIFTI_INTENT_WEIBULL:    return "Weibull distribution";
      case NIFTI_INTENT_CHI:        return "Chi distribution";
      case NIFTI_INTENT_INVGAUSS:   return "Inverse Gaussian distribution";
      case NIFTI_INTENT_EXTVAL:     return "Extreme Value distribution";
      case NIFTI_INTENT_PVAL:       return "P-value";
      case NIFTI_INTENT_LOGPVAL:    return "Log P-value";
      case NIFTI_INTENT_LOG10PVAL:  return "Log10 P-value";
      case NIFTI_INTENT_ESTIMATE:   return "Estimate";
      case NIFTI_INTENT_LABEL:      return "Label index";
      case NIFTI_INTENT_NEURONAME:  return "NeuroNames index";
      case NIFTI_INTENT_GENMATRIX:  return "General matrix";
      case NIFTI_INTENT_SYMMATRIX:  return "Symmetric matrix";
      case NIFTI_INTENT_DISPVECT:   return "Displacement vector";
      case NIFTI_INTENT_VECTOR:     return "Vector";
      case NIFTI_INTENT_POINTSET:   return "Pointset";
      case NIFTI_INTENT_TRIANGLE:   return "Triangle";
      case NIFTI_INTENT_QUATERNION: return "Quaternion";
      case NIFTI_INTENT_DIMLESS:    return "Dimensionless number";
    }
    return "Unknown" ;
  }


  string NiftiHeader::niftiTransformString(const int transform) const {
    switch( transform ) {
      case NIFTI_XFORM_SCANNER_ANAT:  return "Scanner Anat";
      case NIFTI_XFORM_ALIGNED_ANAT:  return "Aligned Anat";
      case NIFTI_XFORM_TALAIRACH:     return "Talairach";
      case NIFTI_XFORM_MNI_152:       return "MNI_152";
    }
    return "Unknown" ;
  }


  string NiftiHeader::unitsString(const int units) const {
    switch( units ){
      case NIFTI_UNITS_METER:  return "m" ;
      case NIFTI_UNITS_MM:     return "mm" ;
      case NIFTI_UNITS_MICRON: return "um" ;
      case NIFTI_UNITS_SEC:    return "s" ;
      case NIFTI_UNITS_MSEC:   return "ms" ;
      case NIFTI_UNITS_USEC:   return "us" ;
      case NIFTI_UNITS_HZ:     return "Hz" ;
      case NIFTI_UNITS_PPM:    return "ppm" ;
      case NIFTI_UNITS_RADS:   return "rad/s" ;
    }
    return "Unknown";
  }


  string NiftiHeader::datatypeString() const {
    switch( datatype ){
      case DT_UNKNOWN:    return "UNKNOWN"    ;
      case DT_BINARY:     return "BINARY"     ;
      case DT_INT8:       return "INT8"       ;
      case DT_UINT8:      return "UINT8"      ;
      case DT_INT16:      return "INT16"      ;
      case DT_UINT16:     return "UINT16"     ;
      case DT_INT32:      return "INT32"      ;
      case DT_UINT32:     return "UINT32"     ;
      case DT_INT64:      return "INT64"      ;
      case DT_UINT64:     return "UINT64"     ;
      case DT_FLOAT32:    return "FLOAT32"    ;
      case DT_FLOAT64:    return "FLOAT64"    ;
      case DT_FLOAT128:   return "FLOAT128"   ;
      case DT_COMPLEX64:  return "COMPLEX64"  ;
      case DT_COMPLEX128: return "COMPLEX128" ;
      case DT_COMPLEX256: return "COMPLEX256" ;
      case DT_RGB24:      return "RGB24"      ;
    }
    return "**ILLEGAL**" ;
  }


  string NiftiHeader::originalOrder() const {
    if ( systemByteOrder() == LSB_FIRST && wasWrongEndian )
      return("MSB_FIRST");
    return("LSB_FIRST");
  }


  string NiftiHeader::fileType() const {
    if ( isAnalyze() )
      return "ANALYZE-7.5";
    string type("NIFTI-"+string(1,char(niftiVersion()+'0')) );
    if (singleFile())
      type+="+";
    return type;
  }


  void NiftiHeader::setNiftiVersion( const char niftiVersion, const bool isSingleFile ) {
    sizeof_hdr=348;
    if ( niftiVersion==0 )
      return;
    magic=string("ni1\0",4);
    if ( isSingleFile )
      magic[1]='+';
    magic[2]=niftiVersion+(int)'0';
    if ( niftiVersion == 2 ) {
      magic.append("\r\n\032\n",4);
      sizeof_hdr=540;
    }
  }


  mat44 NiftiHeader::getQForm() const {
    return ( nifti_quatern_to_mat44( qB, qC, qD, qX, qY, qZ, pixdim[1], pixdim[2], pixdim[3], leftHanded() ) );
  }

  void NiftiHeader::setQForm(const mat44& qForm) {
    double dx,dy,dz;
    nifti_mat44_to_quatern( qForm , qB, qC, qD, qX, qY, qZ, dx, dy, dz, pixdim[0] );
  }


  mat44 NiftiHeader::getSForm() const {
    mat44 R ;
    copy(sX.begin(),sX.end(),R.m[0]);
    copy(sY.begin(),sY.end(),R.m[1]);
    copy(sZ.begin(),sZ.end(),R.m[2]);
    R.m[3][0]=R.m[3][1]=R.m[3][2] = 0.0 ;
    R.m[3][3]= 1.0 ;
    return R;
  }


  void NiftiHeader::setSForm(const mat44& sForm) {
    copy(sForm.m[0],sForm.m[0]+4,sX.begin());
    copy(sForm.m[1],sForm.m[1]+4,sY.begin());
    copy(sForm.m[2],sForm.m[2]+4,sZ.begin());
  }


  //Read common NIFTI/ANALYZE fields to a raw header
  template<class T>
  void NiftiHeader::readCommonFieldsFromRaw(const T& rawHeader) {
    sizeof_hdr = rawHeader.sizeof_hdr;
    vox_offset = rawHeader.vox_offset;
    datatype = rawHeader.datatype;
    bitsPerVoxel = rawHeader.bitpix;
    copy(rawHeader.dim,rawHeader.dim+8,dim.begin());
    copy(rawHeader.pixdim,rawHeader.pixdim+8,pixdim.begin());
    auxillaryFile = string(rawHeader.aux_file);
    description = string(rawHeader.descrip);
    cal_max = rawHeader.cal_max;
    cal_min = rawHeader.cal_min;
  }


  template<class T>
  void NiftiHeader::readNiftiFieldsFromRaw(const T& rawHeader) {
    intent_p1 = rawHeader.intent_p1;
    intent_p2 = rawHeader.intent_p2;
    intent_p3 = rawHeader.intent_p3;
    intentName = string(rawHeader.intent_name);
    intentCode = rawHeader.intent_code;
    sliceOrdering = rawHeader.dim_info;
    sliceStart = rawHeader.slice_start;
    sliceEnd = rawHeader.slice_end;
    sliceCode = rawHeader.slice_code;
    sliceDuration = rawHeader.slice_duration;
    sclSlope = rawHeader.scl_slope;
    sclInter = rawHeader.scl_inter;
    copy(rawHeader.srow_x,rawHeader.srow_x+4,sX.begin());
    copy(rawHeader.srow_y,rawHeader.srow_y+4,sY.begin());
    copy(rawHeader.srow_z,rawHeader.srow_z+4,sZ.begin());
    qB = rawHeader.quatern_b;
    qC = rawHeader.quatern_c;
    qD = rawHeader.quatern_d;
    qX = rawHeader.qoffset_x;
    qY = rawHeader.qoffset_y;
    qZ = rawHeader.qoffset_z;
    sformCode = rawHeader.sform_code;
    qformCode = rawHeader.qform_code;
    units = rawHeader.xyzt_units;
    toffset = rawHeader.toffset;
    magic = string(rawHeader.magic, sizeof(rawHeader.magic) );
  }


  //Write common NIFTI/ANALYZE fields to a raw header
  template<class T>
  void NiftiHeader::writeCommonFieldsToRaw(T& rawNiftiHeader) const {
    rawNiftiHeader.sizeof_hdr = sizeof_hdr;
    copy(dim.begin(),dim.end(),rawNiftiHeader.dim);
    rawNiftiHeader.datatype = datatype;
    rawNiftiHeader.bitpix = bitsPerVoxel;
    copy(pixdim.begin(),pixdim.end(),rawNiftiHeader.pixdim);
    rawNiftiHeader.vox_offset = vox_offset;
    rawNiftiHeader.cal_max = cal_max;
    rawNiftiHeader.cal_min = cal_min;
    strncpy ( rawNiftiHeader.descrip, description.c_str(), 80 );
    strncpy ( rawNiftiHeader.aux_file, auxillaryFile.c_str(), 24 );
  }


  template<class T>
  void NiftiHeader::writeNiftiFieldsToRaw(T& rawNiftiHeader) const {
    memcpy ( rawNiftiHeader.magic, magic.data(), magic.size() );
    rawNiftiHeader.intent_p1 = intent_p1;
    rawNiftiHeader.intent_p2 = intent_p2;
    rawNiftiHeader.intent_p3 = intent_p3;
    strncpy ( rawNiftiHeader.intent_name, intentName.c_str(), 16 );
    rawNiftiHeader.intent_code = intentCode;
    rawNiftiHeader.dim_info = sliceOrdering;
    rawNiftiHeader.slice_start = sliceStart;
    rawNiftiHeader.slice_end = sliceEnd;
    rawNiftiHeader.slice_code = sliceCode;
    rawNiftiHeader.slice_duration = sliceDuration;
    rawNiftiHeader.scl_slope = sclSlope;
    rawNiftiHeader.scl_inter = sclInter;
    copy(sX.begin(),sX.end(),rawNiftiHeader.srow_x);
    copy(sY.begin(),sY.end(),rawNiftiHeader.srow_y);
    copy(sZ.begin(),sZ.end(),rawNiftiHeader.srow_z);
    rawNiftiHeader.quatern_b = qB;
    rawNiftiHeader.quatern_c = qC;
    rawNiftiHeader.quatern_d = qD;
    rawNiftiHeader.qoffset_x = qX;
    rawNiftiHeader.qoffset_y = qY;
    rawNiftiHeader.qoffset_z = qZ;
    rawNiftiHeader.sform_code = sformCode;
    rawNiftiHeader.qform_code = qformCode;
    rawNiftiHeader.xyzt_units = units;
    rawNiftiHeader.toffset = toffset;
  }


  template<>
  void NiftiHeader::readAllFieldsFromRaw(const nifti_1_header& rawHeader) {
    this->readCommonFieldsFromRaw(rawHeader);
    this->readNiftiFieldsFromRaw(rawHeader);
    this->legacyFields.readFieldsFromRaw(rawHeader);
  }


  template<>
  void NiftiHeader::readAllFieldsFromRaw(const nifti_2_header& rawHeader) {
    this->readCommonFieldsFromRaw(rawHeader);
    this->readNiftiFieldsFromRaw(rawHeader);
  }

  template<>
  void NiftiHeader::readAllFieldsFromRaw(const analyzeHeader& rawHeader) {
    this->readCommonFieldsFromRaw(rawHeader);
    this->legacyFields.readFieldsFromRaw(rawHeader);
  }

  template<>
  void NiftiHeader::writeAllFieldsToRaw(nifti_1_header& rawHeader) const {
    this->writeCommonFieldsToRaw(rawHeader);
    this->writeNiftiFieldsToRaw(rawHeader);
    this->legacyFields.writeFieldsToRaw(rawHeader);
  }

  template<>
  void NiftiHeader::writeAllFieldsToRaw(nifti_2_header& rawHeader) const {
    this->writeCommonFieldsToRaw(rawHeader);
    this->writeNiftiFieldsToRaw(rawHeader);
  }

  template<>
  void NiftiHeader::writeAllFieldsToRaw(analyzeHeader& rawHeader) const {
    this->writeCommonFieldsToRaw(rawHeader);
    this->legacyFields.writeFieldsToRaw(rawHeader);
  }

  void NiftiHeader::report() const
  {
    if ( this->isAnalyze() ) {
      analyzeHeader rawHeader;
      writeCommonFieldsToRaw(rawHeader);
      legacyFields.writeFieldsToRaw(rawHeader);
      reportHeader(rawHeader);
      cout << "file_type\t" << this->fileType() << endl;
      cout << "file_code\t" << 0 << endl;
    } else {
      cout << "sizeof_hdr\t" << sizeof_hdr << endl;
      cout << "data_type\t" << datatypeString() << endl;
      for ( int i=0; i<=7; i++ )
        cout << "dim" << i << "\t\t" << dim[i] << endl;
      cout << "vox_units\t" << unitsString(XYZT_TO_SPACE(units)) << endl;
      cout << "time_units\t" << unitsString(XYZT_TO_TIME(units)) << endl;
      cout << "datatype\t" << datatype << endl;
      cout << "nbyper\t\t" << datumByteWidth() << endl;
      cout << "bitpix\t\t" << bitsPerVoxel << endl;
      cout.setf(ios::fixed);
      for ( int i=0; i<=7; i++ )
        cout << "pixdim" << setprecision(6) << i << "\t\t" << pixdim[i] << endl;
      cout << "vox_offset\t" << vox_offset << endl;
      cout << "cal_max\t\t" << cal_max << endl;
      cout << "cal_min\t\t" << cal_min << endl;
      cout << "scl_slope\t" << sclSlope << endl;
      cout << "scl_inter\t" << sclInter << endl;
      cout << "phase_dim\t" << (int)phaseDim() << endl;
      cout << "freq_dim\t" << (int)freqDim() << endl;
      cout << "slice_dim\t" << (int)sliceDim() << endl;
      cout << "slice_name\t" << niftiSliceString() << endl;
      cout << "slice_code\t" << sliceCode << endl;
      cout << "slice_start\t" << sliceStart << endl;
      cout << "slice_end\t" << sliceEnd << endl;
      cout << "slice_duration\t" << sliceDuration << endl;
      cout << "toffset\t\t" << toffset << endl;
      cout << "intent\t\t" << niftiIntentString() << endl;
      cout << "intent_code\t" << intentCode << endl;
      cout << "intent_name\t" << intentName << endl;
      cout << "intent_p1\t" << intent_p1 << endl;
      cout << "intent_p2\t" << intent_p2 << endl;
      cout << "intent_p3\t" << intent_p3 << endl;
      cout << "qform_name\t" << qFormName() << endl;
      cout << "qform_code\t" << qformCode << endl;
      mat44 output(getQForm());
      for ( int i=0;i<4;i++ ) {
        cout << "qto_xyz:" << char(i+'1') << "\t";
        for ( int j=0;j<4;j++ )
          cout << output.m[i][j] << " ";
        cout << endl;
      }
      int icode,jcode,kcode;
      nifti_mat44_to_orientation(output,&icode,&jcode,&kcode);
      cout << "qform_xorient\t" << niftiOrientationString(icode) << endl;
      cout << "qform_yorient\t" << niftiOrientationString(jcode) << endl;
      cout << "qform_zorient\t" << niftiOrientationString(kcode) << endl;
      cout << "sform_name\t" << sFormName() << endl;
      cout << "sform_code\t" << sformCode << endl;
      output=getSForm();
      for ( int i=0;i<4;i++ ) {
        cout << "sto_xyz:" << char(i+'1') << "\t";
        for ( int j=0;j<4;j++ )
          cout << output.m[i][j] << " ";
        cout << endl;
      }
      nifti_mat44_to_orientation(output,&icode,&jcode,&kcode);
      cout << "sform_xorient\t" << niftiOrientationString(icode) << endl;
      cout << "sform_yorient\t" << niftiOrientationString(jcode) << endl;
      cout << "sform_zorient\t" << niftiOrientationString(kcode) << endl;
      cout << "file_type\t" << fileType() << endl;
      cout << "file_code\t" << (int)niftiVersion() << endl;
      cout << "descrip\t\t" << description << endl;
      cout << "aux_file\t" << auxillaryFile << endl;
    }
  }


  //************************End Of NiftiHeader definitions************************


  fileIO::fileIO(const string& filename, const bool reading, const bool useCompression) : debug(false), fileHandle(NULL)
  {
    if ( debug )
      cout << "filename\t" << filename << endl;
    string options( reading ? "rb" : "wb" );
    if ( ! znz_isnull(fileHandle) ) //close currently open file
      znzclose(fileHandle);
    if ( useCompression )
      fileHandle=znzopen( filename.c_str(), options.c_str(), 1 );
    else
      fileHandle=znzopen( filename.c_str(), options.c_str(), 0 );
    if( znz_isnull(fileHandle) ) {
      throw NiftiException("Error: cant open file "+string(filename));
    }
  }


  fileIO& fileIO::operator=(fileIO rhs) {
    std::swap(fileHandle, rhs.fileHandle);
    std::swap(debug, rhs.debug);
    return *this;
  }


  void fileIO::readExtensions(const NiftiHeader& header, vector<NiftiExtension>& extensions)
  {
    znzseek(fileHandle, header.sizeof_hdr, SEEK_SET);
    //To determine if extensions are present, read next 4 bytes ( if present ) if byte[0] is non-zero we have extensions
    char buffer[4];
    try { readRawBytes( buffer, 4 ); }
    catch (const NiftiException &) { //Short read at this point implies NIFTI-pair with no extension
      return;
    }
    //Otherwise we now have 4 bytes - test the first
    if ( buffer[0] == 0 ) //No extensions so just return
      return;
    //We have extensions - begin looping over 8 bytes ( 2 ints ) + buffer
    int32_t iBuffer[2];
    size_t bytesRead(header.sizeof_hdr+4);
    while ( true ) {
      NiftiExtension newExtension;
      if ( header.singleFile() && bytesRead+8 > (size_t)header.vox_offset )
        return; //Return as extension data is malformed and ignored as per nifti-spec
      try {
        bytesRead+=readRawBytes( iBuffer, 8 );
      } catch (const NiftiException &) { //On a read error, return if reading pair or rethrow
        if ( header.singleFile() )
          throw; //Rethrow original exception
        return;
      }
      if ( header.wasWrongEndian )
        byteSwap(sizeof(iBuffer[0]), iBuffer, 2); // swap 2 elements each of size int32_t
      newExtension.esize=iBuffer[0];
      newExtension.ecode=iBuffer[1];
      if ( header.singleFile() && bytesRead+newExtension.esize-8 > (size_t)header.vox_offset )
        return; //Return as extension data is malformed and ignored as per nifti-spec
      if ( newExtension.esize == 0 || newExtension.esize % 16 != 0 )
        throw NiftiException("Error: esize must be positive multiple of 16");
      newExtension.edata.resize(newExtension.esize-8);
      try {
        bytesRead+=readRawBytes( newExtension.edata.data(), newExtension.esize-8 );
      } catch (const NiftiException &) { //On a read error, return if reading pair or rethrow
        if ( header.singleFile() )
          throw;
        return;
      }
      //If we're here then we've been able to read the correct number of bytes and not exceed file limit
      extensions.push_back( newExtension );
      if ( header.singleFile() && bytesRead == (size_t)header.vox_offset ) //We've read final extension for a single file, exit gracefully
        return;
    }
  }


  void fileIO::readData(const NiftiHeader& header, void* buffer)
  {
    znzseek(fileHandle, header.vox_offset, SEEK_SET);
    readRawBytes(buffer, header.nElements()*header.datumByteWidth() );
    if ( header.wasWrongEndian )
      byteSwap( header.datumByteWidth(), buffer, header.nElements() );
  }


  NiftiHeader fileIO::readHeader()
  {
    HeaderType headerType(readHeaderType());
    NiftiHeader header;
    switch ( headerType ) {
      case analyze: readRawHeader<analyzeHeader>(header);
                    break;
      case nifti1:  readRawHeader<nifti_1_header>(header);
                    break;
      case nifti2:  readRawHeader<nifti_2_header>(header);
                    break;
      case other:   throw NiftiException("Error: file does not appear to be a valid NIFTI or ANALYZE image");
    }
    return header;
  }


  HeaderType fileIO::readHeaderType() {
    nifti_1_header rawHeader;
    znzrewind(fileHandle);
    readRawBytes( &rawHeader, (size_t)sizeof(rawHeader) );
    return headerType(rawHeader);
  }


  size_t fileIO::readRawBytes(void* buffer, size_t length)
  {
    size_t bytes = (size_t)znzread( buffer, 1, (size_t)length, fileHandle );
    if ( bytes != length )
      throw NiftiException("Error: short read, file may be truncated");
    return bytes;
  }


  template<class T>
  void fileIO::readRawHeader(NiftiHeader& header) {
    T rawHeader;
    znzrewind(fileHandle);
    readRawBytes( &rawHeader, (size_t)sizeof(rawHeader) );
    if ( NIFTI2_NEEDS_SWAP(rawHeader) == 1 ) {
      header.wasWrongEndian=true;
      byteSwap(rawHeader);
    }
    header.readAllFieldsFromRaw(rawHeader);
    if ( debug )
      reportHeader(rawHeader);
    //Nifti2 check magic
    //if ( header.magic.substr(4,4) != "\r\n\032\n" )
    //   throw NiftiException("Error: bad NIFTI2 signature");
  }


  void fileIO::writeData(const NiftiHeader& header,const void* buffer)
  {
    writeRawBytes(buffer, header.nElements()*header.datumByteWidth() );
  }


  void fileIO::writeExtensions(const NiftiHeader& header, const vector<NiftiExtension>& extensions )
  {
    char temp[4] = {0,0,0,0};
    if ( extensions.size() > 0 ) //We have extensions
      temp[0]=1;
    writeRawBytes(temp, 4 );
    for ( unsigned int current=0; current < extensions.size(); current++ ) {
      if ( extensions[current].esize % 16 != 0)
        throw NiftiException("Error: extension size must be multiple of 16 bytes");
      writeRawBytes(&extensions[current].esize,4);
      writeRawBytes(&extensions[current].ecode,4);
      writeRawBytes(extensions[current].edata.data(),extensions[current].esize-8);
    }
  }


  void fileIO::writeHeader(const NiftiHeader& header)
  {
    if ( header.isAnalyze() )
      writeRawHeader<analyzeHeader>(header);
    else if ( header.niftiVersion() == 1 )
      writeRawHeader<nifti_1_header>(header);
    else if ( header.niftiVersion() == 2 )
      writeRawHeader<nifti_2_header>(header);
  }


  void fileIO::writeRawBytes(const void* buffer, size_t length )
  {
    size_t bytes = (size_t)znzwrite( buffer, 1, (size_t)length, fileHandle );
    if ( bytes != length )
      throw NiftiException("Error: short write, output file will be truncated");
  }


  template<class T>
  void fileIO::writeRawHeader(const NiftiHeader& header) {
    T rawHeader;
    memset(&rawHeader,0,sizeof(rawHeader));
    header.writeAllFieldsToRaw(rawHeader);
    writeRawBytes( &rawHeader, (size_t)sizeof(rawHeader) );
  }


  //This loads in the section of data stored between the limits input ( a value of -1 will default to either 0 ( for minimum limits ) or dim[<idx>]-1 ( for maximum limits ).
  //The returned header will have dim[<idx>] modified to represent ROI size
  NiftiHeader loadImageROI(string filename, char*& buffer, vector<NiftiExtension>& extensions, int64_t xmin, int64_t xmax, int64_t ymin, int64_t ymax, int64_t zmin, int64_t zmax, int64_t tmin, int64_t tmax, int64_t d5min, int64_t d5max, int64_t d6min, int64_t d6max, int64_t d7min, int64_t d7max)
  {
    fileIO reader(filename,true);
    NiftiHeader header=reader.readHeader();
    fill(header.dim.begin()+header.dim[0]+1,header.dim.end(),1);
    reader.readExtensions(header, extensions );
    xmin = xmin == -1 ? 0 : xmin;
    ymin = ymin == -1 ? 0 : ymin;
    zmin = zmin == -1 ? 0 : zmin;
    tmin = tmin == -1 ? 0 : tmin;
    d5min = d5min == -1 ? 0 : d5min;
    d6min = d6min == -1 ? 0 : d6min;
    d7min = d7min == -1 ? 0 : d7min;
    xmax = xmax == -1 ? header.dim[1]-1 : xmax;
    ymax = ymax == -1 ? header.dim[2]-1 : ymax;
    zmax = zmax == -1 ? header.dim[3]-1 : zmax;
    tmax = tmax == -1 ? header.dim[4]-1 : tmax;
    d5max = d5max == -1 ? header.dim[5]-1 : d5max;
    d6max = d6max == -1 ? header.dim[6]-1 : d6max;
    d7max = d7max == -1 ? header.dim[7]-1: d7max;

    //cerr << xmin << " " << xmax << " "  << ymin << " " << ymax << " " << zmin << " " << zmax << " " << tmin << " " << tmax << " " << d5min << " " << d5max << " " << d6min << " " << d6max << " " << d7min << " " << d7max << endl;
    if ( xmin < 0 || xmax > ( header.dim[1]-1 ) || ymin < 0 || ymax > ( header.dim[2]-1 ) || zmin < 0 || zmax > ( header.dim[3]-1 ) ||  tmin < 0 || tmax > ( header.dim[4]-1 ) || d5min < 0 || d5max > ( header.dim[5]-1 ) || d6min < 0 || d6max > ( header.dim[6]-1 ) || d7min < 0 || d7max > ( header.dim[7]-1 ) )
      throw NiftiException("Error: ROI out of bounds for "+string(filename));
    if ( xmin > xmax || ymin > ymax || zmin > zmax || tmin > tmax || d5min > d5max || d6min > d6max || d7min > d7max )
      throw NiftiException("Error: Nonsensical ROI for "+string(filename));

    size_t bufferElements=( xmax-xmin+1 ) * ( ymax-ymin+1 ) * ( zmax-zmin+1 ) * ( tmax-tmin+1 ) * ( d5max-d5min+1 ) * ( d6max-d6min+1 ) * ( d7max-d7min+1 );
    buffer = new char[bufferElements*header.datumByteWidth()];
    char *movingBuffer(buffer);

    size_t voxelsToRead(0);
    size_t voxelsToSeek(0);
    bool wasInROI( 0 >= xmin && 0 <= xmax && 0 >= ymin && 0 <= ymax && 0 >= zmin && 0 <= zmax && 0 >= tmin && 0 <= tmax && 0 >= d5min && 0 <= d5max && 0 >= d6min && 0 <= d6max && 0 >= d7min && 0 <= d7max );
    if ( !header.singleFile() ) {
      //Need to check if header was compressed
      reader=fileIO(filename,true,false);
      nifti_1_header truncatedHeader;
      reader.readRawBytes( &truncatedHeader, (size_t)sizeof(truncatedHeader.sizeof_hdr) );
      //Note extra brackets required around macro to prevent invalid expansion
      bool wasCompressed( (NIFTI2_VERSION(truncatedHeader)) == 0 );
      reader=fileIO(filename.replace(filename.rfind(".hdr"),4,".img"), true, wasCompressed);
    }
    reader.seek(header.nominalVoxOffset(),SEEK_SET);
    for ( int64_t d7 = 0; d7 < header.dim[7]; d7++ )
      for ( int64_t d6 = 0; d6 < header.dim[6]; d6++ )
        for ( int64_t d5 = 0; d5 < header.dim[5]; d5++ )
          for ( int64_t t = 0; t < header.dim[4]; t++ )
            for ( int64_t z = 0; z < header.dim[3]; z++ )
              for ( int64_t y = 0; y < header.dim[2]; y++ )
                for ( int64_t x = 0; x < header.dim[1]; x++ ) {
                  bool currentlyInROI( x >= xmin && x <= xmax && y >= ymin && y <= ymax && z >= zmin && z <= zmax && t >= tmin && t <= tmax && d5 >= d5min && d5 <= d5max && d6 >= d6min && d6 <= d6max && d7 >= d7min && d7 <= d7max );
                  currentlyInROI ? voxelsToRead++ : voxelsToSeek++;
                  if ( wasInROI != currentlyInROI ) { //We have switched so flush according to wasInROI
                    if ( wasInROI ) { //read the accumulated bytes
                      //cerr << "reading " << voxelsToRead << " voxels" << x << " " << y << " " << z << endl;
                      reader.readRawBytes(movingBuffer, voxelsToRead*header.datumByteWidth() );
                      movingBuffer+=voxelsToRead*header.datumByteWidth();
                      voxelsToRead=0;
                    } else {
                      //cerr << "seeking " << voxelsToSeek << " voxels" << x << " " << y << " " << z <<endl;
                      reader.seek(voxelsToSeek*header.datumByteWidth(), SEEK_CUR );
                      voxelsToSeek=0;
                    }
                    wasInROI = currentlyInROI;
                  }
                }
    if ( voxelsToRead ) { //We still have some bytes to read - maybe the whole file?
      //cerr << "reading final " << voxelsToRead  << " voxels" << endl;
      reader.readRawBytes(movingBuffer, voxelsToRead*header.datumByteWidth() );
    }
    if ( voxelsToSeek ) {
      //cerr << "skipping final " << voxelsToSeek  << " voxels" << endl;
    }
    if ( header.wasWrongEndian )
      byteSwap( header.datumByteWidth(), buffer, bufferElements );
    header.dim[1] = 1+xmax-xmin;
    header.dim[2] = 1+ymax-ymin;
    header.dim[3] = 1+zmax-zmin;
    header.dim[4] = 1+tmax-tmin;
    header.dim[5] = 1+d5max-d5min;
    header.dim[6] = 1+d6max-d6min;
    header.dim[7] = 1+d7max-d7min;
    return header;
  }


  NiftiHeader loadImage(string filename, char*& buffer, vector<NiftiExtension>& extensions, bool allocateBuffer)
  {
    fileIO reader(filename,true);
    NiftiHeader header=reader.readHeader();
    reader.readExtensions(header, extensions );
    if ( !header.singleFile() )
      reader=fileIO(filename.replace(filename.rfind(".hdr"),4,".img"), true );
    if ( allocateBuffer )
      buffer = new char[header.nElements()*header.datumByteWidth()];
    reader.readData( header, buffer);
    return header;
  }


  NiftiHeader loadHeader(const string filename)
  {
    fileIO reader(filename,true);
    NiftiHeader header=reader.readHeader();
    return header;
  }


  NiftiHeader loadExtensions(const string filename, vector<NiftiExtension>& extensions)
  {
    fileIO reader(filename,true);
    NiftiHeader header=reader.readHeader();
    reader.readExtensions(header, extensions );
    return header;
  }


  void saveImage(string filename, const char* buffer, const vector<NiftiExtension>& extensions, NiftiHeader header, const bool useCompression)
  {
    fileIO writer(filename, false, useCompression);
    size_t sizeOfExtensions(4);
    header.vox_offset=header.sizeof_hdr;
    for ( unsigned int i(0); i < extensions.size(); i++ )
      sizeOfExtensions+=extensions[0].extensionSize();
    header.vox_offset+=sizeOfExtensions;
    if ( !header.singleFile() )
      header.vox_offset=0;
    writer.writeHeader(header);
    if ( !header.isAnalyze() )
      writer.writeExtensions(header,extensions);
    if ( !header.singleFile() )
      writer=fileIO(filename.replace(filename.rfind(".hdr"),4,".img"), false, useCompression);
    writer.writeData(header,buffer);
  }


  template <class T>
  HeaderType headerType(const T& header) {
    int version2(NIFTI2_VERSION(header));
    int version1(NIFTI_VERSION(header));
    return ((HeaderType)(version2 != 0 ? version2 == 1 ? version1 : version2: other));
  }


  //This method swaps the fields present in all image types
  template<class T>
  void byteSwapImageFields(T& rawHeader) {
    byteSwap(sizeof(rawHeader.sizeof_hdr),&rawHeader.sizeof_hdr);
    byteSwap(sizeof(rawHeader.dim[0]),rawHeader.dim,8);
    byteSwap(sizeof(rawHeader.datatype),&rawHeader.datatype);
    byteSwap(sizeof(rawHeader.bitpix),&rawHeader.bitpix);
    byteSwap(sizeof(rawHeader.pixdim[0]),rawHeader.pixdim,8);
    byteSwap(sizeof(rawHeader.vox_offset),&rawHeader.vox_offset);
    byteSwap(sizeof(rawHeader.cal_max),&rawHeader.cal_max);
    byteSwap(sizeof(rawHeader.cal_min),&rawHeader.cal_min);
  }


  //This method swaps the fields present in all nifti types
  template<class T>
  void byteSwapNiftiFields(T& rawNiftiHeader) {
    byteSwap(sizeof(rawNiftiHeader.intent_p1),&rawNiftiHeader.intent_p1);
    byteSwap(sizeof(rawNiftiHeader.intent_p2),&rawNiftiHeader.intent_p2);
    byteSwap(sizeof(rawNiftiHeader.intent_p3),&rawNiftiHeader.intent_p3);
    byteSwap(sizeof(rawNiftiHeader.intent_code),&rawNiftiHeader.intent_code);
    byteSwap(sizeof(rawNiftiHeader.slice_start),&rawNiftiHeader.slice_start);
    byteSwap(sizeof(rawNiftiHeader.scl_slope),&rawNiftiHeader.scl_slope);
    byteSwap(sizeof(rawNiftiHeader.scl_inter),&rawNiftiHeader.scl_inter);
    byteSwap(sizeof(rawNiftiHeader.slice_end),&rawNiftiHeader.slice_end);
    byteSwap(sizeof(rawNiftiHeader.slice_code),&rawNiftiHeader.slice_code); //char in Nifti1, int in Nifti2
    byteSwap(sizeof(rawNiftiHeader.xyzt_units),&rawNiftiHeader.xyzt_units); //char in Nifti1, int in Nifti2
    byteSwap(sizeof(rawNiftiHeader.slice_duration),&rawNiftiHeader.slice_duration);
    byteSwap(sizeof(rawNiftiHeader.toffset),&rawNiftiHeader.toffset);
    byteSwap(sizeof(rawNiftiHeader.sform_code),&rawNiftiHeader.sform_code);
    byteSwap(sizeof(rawNiftiHeader.qform_code),&rawNiftiHeader.qform_code);
    byteSwap(sizeof(rawNiftiHeader.quatern_b),&rawNiftiHeader.quatern_b);
    byteSwap(sizeof(rawNiftiHeader.quatern_c),&rawNiftiHeader.quatern_c);
    byteSwap(sizeof(rawNiftiHeader.quatern_d),&rawNiftiHeader.quatern_d);
    byteSwap(sizeof(rawNiftiHeader.qoffset_x),&rawNiftiHeader.qoffset_x);
    byteSwap(sizeof(rawNiftiHeader.qoffset_y),&rawNiftiHeader.qoffset_y);
    byteSwap(sizeof(rawNiftiHeader.qoffset_z),&rawNiftiHeader.qoffset_z);
    byteSwap(sizeof(rawNiftiHeader.srow_x[0]),rawNiftiHeader.srow_x,4);
    byteSwap(sizeof(rawNiftiHeader.srow_y[0]),rawNiftiHeader.srow_y,4);
    byteSwap(sizeof(rawNiftiHeader.srow_z[0]),rawNiftiHeader.srow_z,4);
  }


  //This method swaps the fields present only in NIFTI1/Analyze
  template<class T>
  void byteSwapLegacyFields(T& rawHeader) {
    byteSwap(sizeof(rawHeader.extents),&rawHeader.extents);
    byteSwap(sizeof(rawHeader.session_error),&rawHeader.session_error);
    byteSwap(sizeof(rawHeader.glmax),&rawHeader.glmax);
    byteSwap(sizeof(rawHeader.glmin),&rawHeader.glmin);
  }


  //This method swaps the fields present only in ANALYZE
  void byteSwapAnalyzeFields(analyzeHeader& rawHeader) {
    byteSwap(sizeof(rawHeader.dim_un0),&rawHeader.dim_un0);
    byteSwap(sizeof(rawHeader.compressed),&rawHeader.compressed);
    byteSwap(sizeof(rawHeader.verified),&rawHeader.verified);
    byteSwap(sizeof(rawHeader.views),&rawHeader.views);
    byteSwap(sizeof(rawHeader.vols_added),&rawHeader.vols_added);
    byteSwap(sizeof(rawHeader.start_field),&rawHeader.start_field);
    byteSwap(sizeof(rawHeader.field_skip),&rawHeader.field_skip);
    byteSwap(sizeof(rawHeader.omax),&rawHeader.omax);
    byteSwap(sizeof(rawHeader.omin),&rawHeader.omin);
    byteSwap(sizeof(rawHeader.smax),&rawHeader.smax);
    byteSwap(sizeof(rawHeader.smin),&rawHeader.smin);
    short *ptr((short *)rawHeader.originator);
    byteSwap(sizeof(ptr[0]),ptr,5);
  }


  template<>
  void byteSwap(nifti_1_header& rawHeader) {
    byteSwapImageFields(rawHeader);
    byteSwapLegacyFields(rawHeader);
    byteSwapNiftiFields(rawHeader);
  }


  template<>
  void byteSwap(nifti_2_header& rawHeader) {
    byteSwapImageFields(rawHeader);
    byteSwapNiftiFields(rawHeader);
  }


  template<>
  void byteSwap(analyzeHeader& rawHeader) {
    byteSwapImageFields(rawHeader);
    byteSwapLegacyFields(rawHeader);
    byteSwapAnalyzeFields(rawHeader);
  }


  void byteSwap(const size_t elementLength, void* vBuffer,const unsigned long nElements)
  {
    //cerr << "Low level byte swap: " << elementLength << " " << vBuffer << " " << nElements << endl;
    char *buffer(static_cast<char *>(vBuffer));
    for ( unsigned long current = 0; current < nElements; current ++ ) {
      reverse(buffer,buffer+elementLength);
      buffer+=elementLength;
    }
  }

  //Start of reporting functions
  //reportHeader overloads report in stored field order
  void reportHeader(const nifti_1_header& header)
  {
    cout << "sizeof_hdr\t" << header.sizeof_hdr << endl;
    cout << "data_type\t" << string(header.data_type) << endl;
    cout << "db_name\t" << string(header.db_name) << endl;
    cout << "extents\t\t" << header.extents << endl;
    cout << "session_error\t" << header.session_error << endl;
    cout << "regular\t\t" << header.regular << endl;
    cout << "dim_info\t" << (int)header.dim_info << endl;
    for ( int i=0; i<=7; i++ )
      cout << "dim" << i << "\t\t" << header.dim[i] << endl;
    cout << "intent_p1\t" << header.intent_p1 << endl;
    cout << "intent_p2\t" << header.intent_p2 << endl;
    cout << "intent_p3\t" << header.intent_p3 << endl;
    cout << "intent_code\t" << header.intent_code << endl;
    cout << "datatype\t" << header.datatype << endl;
    cout << "bitpix\t\t" << header.bitpix << endl;
    cout << "slice_start\t" << header.slice_start << endl;
    for ( int i=0; i<=7; i++ )
      cout << "pixdim" << i << "\t\t" << header.pixdim[i] << endl;
    cout << "vox_offset\t" << header.vox_offset << endl;
    cout << "scl_slope\t" << header.scl_slope << endl;
    cout << "scl_inter\t" << header.scl_inter << endl;
    cout << "slice_end\t" << header.slice_end << endl;
    cout << "slice_code\t" << (int)header.slice_code << endl;
    cout << "xyzt_units\t" << (int)header.xyzt_units << endl;
    cout << "cal_max\t\t" << header.cal_max << endl;
    cout << "cal_min\t\t" << header.cal_min << endl;
    cout << "slice_duration\t" << header.slice_duration << endl;
    cout << "toffset\t\t" << header.toffset << endl;
    cout << "glmax\t\t" << header.glmax << endl;
    cout << "glmin\t\t" << header.glmin << endl;
    cout << "descrip\t\t" << string(header.descrip) << endl;
    cout << "aux_file\t" << string(header.aux_file) << endl;
    cout << "qform_code\t" << header.qform_code << endl;
    cout << "sform_code\t" << header.sform_code << endl;
    cout << "quatern_b\t" << header.quatern_b << endl;
    cout << "quatern_c\t" << header.quatern_c << endl;
    cout << "quatern_d\t" << header.quatern_d << endl;
    cout << "qoffset_x\t" << header.qoffset_x << endl;
    cout << "qoffset_y\t" << header.qoffset_y << endl;
    cout << "qoffset_z\t" << header.qoffset_z << endl;
    cout << "srow_x:\t\t" << header.srow_x[0] << " "  << header.srow_x[1] << " " << header.srow_x[2] << " " << header.srow_x[3] << endl;
    cout << "srow_y:\t\t" << header.srow_y[0] << " "  << header.srow_y[1] << " " << header.srow_y[2] << " " << header.srow_y[3] << endl;
    cout << "srow_z:\t\t" << header.srow_z[0] << " "  << header.srow_z[1] << " " << header.srow_z[2] << " " << header.srow_z[3] << endl;
    cout << "intent_name\t" << string(header.intent_name) << endl;
    cout << "magic\t\t" << string(header.magic) << endl;
  }


  void reportHeader(const nifti_2_header& header)
  {
    cout << "sizeof_hdr\t" << header.sizeof_hdr << endl;
    cout << "magic\t\t" << string(header.magic) << endl;
    cout << "datatype\t" << header.datatype << endl;
    cout << "bitpix\t\t" << header.bitpix << endl;
    for ( int i=0; i<=7; i++ )
      cout << "dim" << i << "\t\t" << header.dim[i] << endl;
    cout << "intent_p1\t" << header.intent_p1 << endl;
    cout << "intent_p2\t" << header.intent_p2 << endl;
    cout << "intent_p3\t" << header.intent_p3 << endl;
    for ( int i=0; i<=7; i++ )
      cout << "pixdim" << i << "\t\t" << header.pixdim[i] << endl;
    cout << "vox_offset\t" << header.vox_offset << endl;
    cout << "scl_slope\t" << header.scl_slope << endl;
    cout << "scl_inter\t" << header.scl_inter << endl;
    cout << "cal_max\t\t" << header.cal_max << endl;
    cout << "cal_min\t\t" << header.cal_min << endl;
    cout << "slice_duration\t" << header.slice_duration << endl;
    cout << "toffset\t" << header.toffset << endl;
    cout << "slice_start\t" << header.slice_start << endl;
    cout << "slice_end\t" << header.slice_end << endl;
    cout << "descrip\t\t" << string(header.descrip) << endl;
    cout << "aux_file\t" << string(header.aux_file) << endl;
    cout << "qform_code\t" << header.qform_code << endl;
    cout << "sform_code\t" << header.sform_code << endl;
    cout << "quatern_b\t" << header.quatern_b << endl;
    cout << "quatern_c\t" << header.quatern_c << endl;
    cout << "quatern_d\t" << header.quatern_d << endl;
    cout << "qoffset_x\t" << header.qoffset_x << endl;
    cout << "qoffset_y\t" << header.qoffset_y << endl;
    cout << "qoffset_z\t" << header.qoffset_z << endl;
    cout << "srow_x:\t\t" << header.srow_x[0] << " "  << header.srow_x[1] << " " << header.srow_x[2] << " " << header.srow_x[3] << endl;
    cout << "srow_y:\t\t" << header.srow_y[0] << " "  << header.srow_y[1] << " " << header.srow_y[2] << " " << header.srow_y[3] << endl;
    cout << "srow_z:\t\t" << header.srow_z[0] << " "  << header.srow_z[1] << " " << header.srow_z[2] << " " << header.srow_z[3] << endl;
    cout << "slice_code\t" << (int)header.slice_code << endl;
    cout << "xyzt_units\t" << (int)header.xyzt_units << endl;
    cout << "intent_code\t" << header.intent_code << endl;
    cout << "intent_name\t" << string(header.intent_name) << endl;
    cout << "dim_info\t" << (int)header.dim_info << endl;
    cout << "unused_str\t" << string(header.unused_str);
  }


  void reportHeader(const analyzeHeader& header) {
    cout << "sizeof_hdr\t" << header.sizeof_hdr << endl;
    cout << "data_type\t" << header.data_type << endl;
    cout << "db_name\t\t" << header.db_name << endl;
    cout << "extents\t\t" << header.extents << endl;
    cout << "session_error\t" << header.session_error << endl;
    cout << "regular\t\t" << header.regular << endl;
    cout << "hkey_un0\t" << header.hkey_un0 << endl;
    for ( int i=0; i<=7; i++ )
      cout << "dim" << i << "\t\t" << header.dim[i] << endl;
    cout << "vox_units\t" << header.vox_units << endl;
    cout << "cal_units\t" << header.cal_units << endl;
    cout << "unused1\t\t" << header.unused1 << endl;
    cout << "datatype\t" << header.datatype << endl;
    cout << "bitpix\t\t" << header.bitpix << endl;
    cout << "dim_un0\t\t" << header.dim_un0 << endl;
    cout.setf(ios::fixed);
    for ( int i=0; i<=7; i++ )
      cout << "pixdim" << setprecision(6) << i << "\t\t" << header.pixdim[i] << endl;
    cout << "vox_offset\t" << header.vox_offset << endl;
    cout << "funused1\t" << header.funused1 << endl;
    cout << "funused2\t" << header.funused2 << endl;
    cout << "funused3\t" << header.funused3 << endl;
    cout << "cal_max\t\t" << header.cal_max << endl;
    cout << "cal_min\t\t" << header.cal_min << endl;
    cout << "compressed\t" << header.compressed << endl;
    cout << "verified\t" << header.verified << endl;
    cout << "glmax\t\t" << header.glmax << endl;
    cout << "glmin\t\t" << header.glmin << endl;
    cout << "descrip\t\t" << header.descrip << endl;
    cout << "aux_file\t" << header.aux_file << endl;
    cout << "orient\t\t" << static_cast<unsigned>(header.orient) << endl;
    for ( int i=0; i<5; i++ )
      cout << "origin" << setprecision(6) << i << "\t\t" << ((short *)header.originator)[i] << endl;
    cout << "generated\t" << header.generated << endl;
    cout << "scannum\t\t" << header.scannum << endl;
    cout << "patient_id\t" << header.patient_id << endl;
    cout << "exp_date\t" << header.exp_date << endl;
    cout << "exp_time\t" << header.exp_time << endl;
    cout << "hist_un0\t" << header.hist_un0 << endl;
    cout << "views\t\t" << header.views << endl;
    cout << "vols_added\t" << header.vols_added << endl;
    cout << "start_field\t" << header.start_field << endl;
    cout << "field_skip\t" << header.field_skip << endl;
    cout << "omax\t\t" << header.omax << endl;
    cout << "omin\t\t" << header.omin << endl;
    cout << "smax\t\t" << header.smax << endl;
    cout << "smin\t\t" << header.smin << endl;
  }
}
