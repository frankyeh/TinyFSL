//This file includes code from niftio1_io.c and niftio1_.h, as released to
//the public domain by Rober W Cox, August 2003 and modified by
//Mark Jenkinson, August 2004 and Rick Reynolds, December 2004 and
//Matthew Webster 2018
#if __cplusplus >= 201103L || __clang__
  #include <array>
using std::array;
#else
  #include <tr1/array>
using std::tr1::array;
#endif

#include "nifti1.h"

namespace NiftiIO {

  enum {
    NIFTI_L2R = 1,    /* Left to Right         */
    NIFTI_R2L = 2,    /* Right to Left         */
    NIFTI_P2A = 3,    /* Posterior to Anterior */
    NIFTI_A2P = 4,    /* Anterior to Posterior */
    NIFTI_I2S = 5,    /* Inferior to Superior  */
    NIFTI_S2I = 6,    /* Superior to Inferior  */
  };/* enum */

//The following is the Mayo Analyze 7.5 specification, with SPM99 additions
  struct analyzeHeader {
    //Start of header_key
    int sizeof_hdr;             //Byte size of header ( always 348 )
    char data_type[10];
    char db_name[18];
    int extents;                //Should be 16384
    short int session_error;
    char regular;               //'r' to indicate all volumes have same size
    char hkey_un0;
    //Start of image_dimension
    short int dim[8];
    char vox_units[4];          //Originally defined as unused fields, but
    char cal_units[8];          //implemented as ???_units in header
    short int unused1;
    short int datatype;
    short int bitpix;
    short int dim_un0;
    float pixdim[8];
    float vox_offset;
    float funused1;             //SPM99: Scale factor, unused in FSL
    float funused2;
    float funused3;
    float cal_max;              //Maximum calibration value
    float cal_min;              //Minimum calibration value
    int compressed;             //Listed as float in PDF, but int in dbh.h
    int verified;               //Listed as float in PDF, but int in dbh.h
    int glmax;                  //Maximum value for entire volume
    int glmin;                  //Minimum value for entire volume
    //Start of data_history
    char descrip[80];
    char aux_file[24];
    char orient;
    //CAUTION: Redefining as short[5] breaks packing order: sizeof returns 352
    char originator[10];        //SPM99:short X,Y,Z,?,? near Anterior Commissure
    char generated[10];
    char scannum[10];
    char patient_id[10];
    char exp_date[10];
    char exp_time[10];
    char hist_un0[3];
    int views;
    int vols_added;
    int start_field;
    int field_skip;
    int omax;
    int omin;
    int smax;
    int smin;
  };


  class LegacyFields {
  public:
    //Methods
    LegacyFields();
    short * origin(void) { return reinterpret_cast<short *>(originator.data()); }
    void readFieldsFromRaw(const analyzeHeader& rawHeader);
    void readFieldsFromRaw(const nifti_1_header& rawHeader);
    void writeFieldsToRaw(analyzeHeader& rawHeader) const;
    void writeFieldsToRaw(nifti_1_header& rawHeader) const;
    //Members
    array<char,10> data_type;
    array<char,18> db_name;
    int extents;
    short int session_error;
    char regular;
    char hkey_un0;
    array<char,4> vox_units;
    array<char,8> cal_units;
    short int unused1;
    short int dim_un0;
    float funused1;
    float funused2;
    float funused3;
    int compressed;
    int verified;
    int glmax;
    int glmin;
    char orient;
    array<char,10> originator;
    array<char,10> generated;
    array<char,10> scannum;
    array<char,10> patient_id;
    array<char,10> exp_date;
    array<char,10> exp_time;
    array<char,3> hist_un0;
    int views;
    int vols_added;
    int start_field;
    int field_skip;
    int omax;
    int omin;
    int smax;
    int smin;
  };


  typedef struct {                   /** 4x4 matrix struct **/
    float m[4][4] ;
  } mat44 ;


  typedef struct {                   /** 3x3 matrix struct **/
    float m[3][3] ;
  } mat33 ;

  namespace legacy {
    mat44 nifti_quatern_to_mat44( float qb, float qc, float qd, float qx, float qy, float qz, float dx, float dy, float dz, float qfac );
    void nifti_mat44_to_quatern( const mat44& R , double& qb, double& qc, double& qd, double& qx, double& qy, double& qz, double& dx, double& dy, double& dz, double& qfac );
    void nifti_mat44_to_orientation( mat44 R , int *icod, int *jcod, int *kcod );
    mat33 mat44_to_mat33(mat44 x);
    float nifti_mat33_determ( mat33 R );
    mat33 nifti_mat33_inverse( mat33 R );
    mat44 nifti_mat44_inverse( mat44 R );
  }
}
