/*! \file EddyHelperClasses.h
    \brief Contains declaration of classes that implements useful functionality for the eddy project.

    \author Jesper Andersson
    \version 1.0b, Sep., 2012.
*/
// Declarations of classes that implements useful
// concepts for the eddy current project.
//
// EddyHelperClasses.h
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2011 University of Oxford
//

#ifndef EddyHelperClasses_h
#define EddyHelperClasses_h

#include <cstdlib>
#include <cstring>
#include <string>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>

#include <boost/current_function.hpp>
#include "armawrap/newmat.h"
#include "newimage/newimageall.h"
#include "miscmaths/miscmaths.h"


#ifndef TicToc
#define TicToc(task) { timeval tim;		\
  gettimeofday(&tim,NULL); \
  task; \
  timeval tim2; \
  gettimeofday(&tim2,NULL); \
  std::cout << "Call to " #task " took " << 1000000*(tim2.tv_sec-tim.tv_sec) + tim2.tv_usec - tim.tv_usec << " usec" << std::endl; }
#endif


#define UseEddyTry

#ifdef UseEddyTry
  #ifndef EddyTry
    #define EddyTry try
    #define EddyCatch catch (const std::exception& e) { std::cout << e.what() << std::endl; throw EDDY::EddyException(std::string(__FILE__) + ":::  " + std::string(__func__) + ":  Exception thrown"); }
  #endif
#else // If not UseEddyTry
  #ifndef EddyTry
    #define EddyTry
    #define EddyCatch
  #endif
# endif // End #ifdef UseEddyTry

namespace EDDY {

enum Parameters { ZERO_ORDER_MOVEMENT, MOVEMENT, EC, ALL };
enum ECModel { NoEC, Linear, Quadratic, Cubic, Unknown };
enum SecondLevelECModel { No_2nd_lvl_mdl, Linear_2nd_lvl_mdl, Quadratic_2nd_lvl_mdl, Unknown_2nd_lvl_mdl };
enum OffsetModel { LinearOffset, QuadraticOffset, UnknownOffset };
enum OLType { SliceWise, GroupWise, Both };
enum ScanType { DWI, B0 , ANY };
enum FinalResampling { JAC, LSR, UNKNOWN_RESAMPLING};

/****************************************************************//**
*
* \brief This is the exception that is being thrown by routines
*  in the core code of eddy.
*
* This is the exception that is being thrown by routines
* in the core code of eddy.
*
********************************************************************/
class EddyException: public std::exception
{
public:
  EddyException(const std::string& msg) noexcept : message(msg) {}
  ~EddyException() noexcept {}
  virtual const char * what() const noexcept { return std::string("EDDY:::  " + message).c_str(); }
private:
  std::string message;
};

/****************************************************************//**
*
* \brief This class reads a JSON file.
*
* This class reads a JSON file and can subsequently be interrogated
* for a very limited subset of parameters from that file.
*
********************************************************************/
class JsonReader
{
public:
  JsonReader();
  JsonReader(const std::string& fname) EddyTry : _fname(fname) { common_read(); } EddyCatch
  void Read(const std::string& fname) EddyTry { _fname = fname; common_read(); } EddyCatch
  /// Return a slice order matrix in the same format as a --slspec text file
  NEWMAT::Matrix SliceOrdering() const;
  /// Return a Phase-encode vector in the format expected in an --acqp file
  NEWMAT::ColumnVector PEVector() const;
  /// Return the total readout time (fourth column in an --acqp file). Not properly implemented yet.
  double TotalReadoutTime() const { return(0.05); }
private:
  void common_read();
  std::string _fname;
  std::string _content;
};

/****************************************************************//**
*
* \brief This class manages the diffusion parameters for one scan
*
* This class manages the diffusion parameters for one scan
*
********************************************************************/
class DiffPara
{
public:
  /// Default constructor. Sets b-vector to [1 0 0] and b-value to zero.
  DiffPara() EddyTry { _bvec.ReSize(3); _bvec(1)=1; _bvec(2)=0; _bvec(3)=0; _bval = 0; } EddyCatch
  /// Constructs a diffpara object with b-vec [1 0 0] and specified b-value
  DiffPara(double bval) EddyTry : _bval(bval) { _bvec.ReSize(3); _bvec(1)=1; _bvec(2)=0; _bvec(3)=0; } EddyCatch
  /// Constructs a DiffPara object from a b-vector and a b-value.
  DiffPara(const NEWMAT::ColumnVector&   bvec,
	   double                        bval) EddyTry : _bvec(bvec), _bval(bval)
  {
    if (_bvec.Nrows() != 3) throw EddyException("DiffPara::DiffPara: b-vector must be three elements long");
    if (_bval < 0) throw EddyException("DiffPara::DiffPara: b-value must be non-negative");
    if (_bval) _bvec /= _bvec.NormFrobenius();
  } EddyCatch
  /// Prints out b-vector and b-value in formatted way
  friend std::ostream& operator<<(std::ostream& op, const DiffPara& dp) EddyTry { op << "b-vector: " << dp._bvec.t() << std::endl << "b-value:  " << dp._bval << std::endl; return(op); } EddyCatch
  /// Returns true if the b-value AND the direction are the same
  bool operator==(const DiffPara& rhs) const;
  /// Same as !(a==b)
  bool operator!=(const DiffPara& rhs) const EddyTry { return(!(*this==rhs)); } EddyCatch
  /// Compares the b-values
  bool operator>(const DiffPara& rhs) const EddyTry { return(this->bVal()>rhs.bVal()); } EddyCatch
  /// Compares the b-values
  bool operator<(const DiffPara& rhs) const EddyTry { return(this->bVal()<rhs.bVal()); } EddyCatch
  /// Returns a normalised b-vector
  NEWMAT::ColumnVector bVec() const EddyTry { return(_bvec); } EddyCatch
  /// Returns the b-value
  double bVal() const { return(_bval); }
private:
  NEWMAT::ColumnVector _bvec;
  double               _bval;
};

/****************************************************************//**
*
* \brief This class manages the acquisition parameters for one scan
*
* This class manages the acquisition parameters for one scan
*
********************************************************************/
class AcqPara
{
public:
  /// Constructs an AcqPara object from a phase-encode vector and a total readout-time (sec)
  AcqPara(const NEWMAT::ColumnVector&   pevec,
          double                        rotime);
  /// Prints out phase-encode vactor and readout-time (sec) in formatted way
  friend std::ostream& operator<<(std::ostream& op, const AcqPara& ap) EddyTry { op << "Phase-encode vector: " << ap._pevec.t() << std::endl << "Read-out time: " << ap._rotime; return(op); } EddyCatch
  /// Returns true if both PE direction and readout time are the same.
  bool operator==(const AcqPara& rh) const;
  /// Same as !(a==b)
  bool operator!=(const AcqPara& rh) const EddyTry { return(!(*this == rh)); } EddyCatch
  /// Returns the phase-enocde vector
  NEWMAT::ColumnVector PhaseEncodeVector() const EddyTry { return(_pevec); } EddyCatch
  /// Returns the a binarised version of the phase-encode vector as a std::vector<unsigned int>
  std::vector<unsigned int> BinarisedPhaseEncodeVector() const;
  /// Returns the readout-time in seconds
  double ReadOutTime() const { return(_rotime); }
private:
  NEWMAT::ColumnVector _pevec;
  double               _rotime;
};

class PolationPara
{
public:
  PolationPara() EddyTry : _int(NEWIMAGE::spline), _ext(NEWIMAGE::periodic), _evip(true), _s2v_int(NEWIMAGE::trilinear), _sp_lambda(0.005) {} EddyCatch
  PolationPara(NEWIMAGE::interpolation ip, NEWIMAGE::extrapolation ep, bool evip, NEWIMAGE::interpolation s2v_ip, double sp_lambda=0.005) EddyTry : _sp_lambda(sp_lambda)
  {
    SetInterp(ip); SetExtrap(ep); SetExtrapValidity(evip); SetS2VInterp(s2v_ip);
  } EddyCatch
  NEWIMAGE::interpolation GetInterp() const { return(_int); }
  NEWIMAGE::interpolation GetS2VInterp() const { return(_s2v_int); }
  NEWIMAGE::extrapolation GetExtrap() const { return(_ext); }
  bool GetExtrapValidity() const { return(_evip); }
  double GetSplineInterpLambda() const { return(_sp_lambda); }
  void SetInterp(NEWIMAGE::interpolation ip) EddyTry {
    if (ip!=NEWIMAGE::trilinear && ip!=NEWIMAGE::spline) throw EddyException("PolationPara::SetInterp: Invalid interpolation");
    _int = ip;
  } EddyCatch
  void SetS2VInterp(NEWIMAGE::interpolation ip) EddyTry {
    if (ip!=NEWIMAGE::trilinear && ip!=NEWIMAGE::spline) throw EddyException("PolationPara::SetS2VInterp: Invalid interpolation");
    _s2v_int = ip;
  } EddyCatch
  void SetExtrap(NEWIMAGE::extrapolation ep) EddyTry {
    if (ep!=NEWIMAGE::mirror && ep!=NEWIMAGE::periodic) throw EddyException("PolationPara::SetExtrap: Invalid extrapolation");
    if (ep!=NEWIMAGE::periodic && _evip) throw EddyException("PolationPara::SetExtrap: Invalid extrapolation and validity combo");
    _ext = ep;
  } EddyCatch
  void SetExtrapValidity(bool evip) EddyTry {
    if (evip && _ext!=NEWIMAGE::periodic) throw EddyException("PolationPara::SetExtrapValidity: Invalid extrapolation and validity combo");
    _evip = evip;
  } EddyCatch
  /// Writes some useful debug info
  friend std::ostream& operator<<(std::ostream& out, const PolationPara& pp) EddyTry {
    out << "PolationPara:" << std::endl;
    if (pp._int == NEWIMAGE::trilinear) out << "Interpolation: trilinear" << std::endl;
    else out << "Interpolation: spline" << std::endl;
    if (pp._ext == NEWIMAGE::mirror) out << "Extrapolation: mirror" << std::endl;
    else out << "Extrapolation: periodic" << std::endl;
    if (pp._evip) out << "Extrapolation along EP is valid" << std::endl;
    else out << "Extrapolation along EP is not valid" << std::endl;
    if (pp._s2v_int == NEWIMAGE::trilinear) out << "Slice-to-vol interpolation: trilinear" << std::endl;
    else out << "Slice-to-vol interpolation: spline" << std::endl;
    out << "Lambda for spline interpolation in z-direction: " << pp._sp_lambda << std::endl;
    return(out);
  } EddyCatch
private:
  NEWIMAGE::interpolation _int;        ///< Interpolation method
  NEWIMAGE::extrapolation _ext;        ///< Extrapolation method
  bool                    _evip;       ///< Specifies if extrapolation is valid in PE-direction
  NEWIMAGE::interpolation _s2v_int;    ///< z-direction interpolation for slice-to-vol
  double                  _sp_lambda;  ///< Lambda when doing spline interpolation in z-direction for slice-to-vol
};

class JacMasking
{
public:
  JacMasking(bool doit, double ll, double ul) : _doit(doit), _ll(ll), _ul(ul) {}
  ~JacMasking() {}
  bool DoIt() const { return(_doit); }
  double LowerLimit() const { return(_ll); }
  double UpperLimit() const { return(_ul); }
private:
  bool    _doit;
  double  _ll;
  double  _ul;
};

class ReferenceScans
{
public:
  ReferenceScans() EddyTry : _loc_ref(0), _b0_loc_ref(0), _b0_shape_ref(0), _dwi_loc_ref(0), _shell_loc_ref(1,0), _shell_shape_ref(1,0) {} EddyCatch
  ReferenceScans(std::vector<unsigned int> b0indx, std::vector<std::vector<unsigned int> > shindx) EddyTry : _loc_ref(0), _shell_loc_ref(shindx.size()), _shell_shape_ref(shindx.size()) {
    _b0_loc_ref = (b0indx.size() > 0 ? b0indx[0] : 0); _b0_shape_ref = (b0indx.size() > 0 ? b0indx[0] : 0);
    _dwi_loc_ref = ((shindx.size() && shindx[0].size()) ? shindx[0][0] : 0);
    for (unsigned int i=0; i<shindx.size(); i++) { _shell_loc_ref[i] = shindx[i][0]; _shell_shape_ref[i] = shindx[i][0]; }
  } EddyCatch
  unsigned int GetLocationReference() const { return(_loc_ref); }
  unsigned int GetB0LocationReference() const { return(_b0_loc_ref); }
  unsigned int GetB0ShapeReference() const { return(_b0_shape_ref); }
  unsigned int GetDWILocationReference() const { return(_dwi_loc_ref); }
  unsigned int GetShellLocationReference(unsigned int si) const EddyTry {
    if (si>=_shell_loc_ref.size()) throw EddyException("ReferenceScans::GetShellLocationReference: Shell index out of range");
    else return(_shell_loc_ref[si]);
  } EddyCatch
  unsigned int GetShellShapeReference(unsigned int si) const EddyTry {
    if (si>=_shell_shape_ref.size()) throw EddyException("ReferenceScans::GetShellShapeReference: Shell index out of range");
    else return(_shell_shape_ref[si]);
  } EddyCatch
  void SetLocationReference(unsigned int indx) { _loc_ref=indx; }
  void SetB0LocationReference(unsigned int indx) { _b0_loc_ref=indx; }
  void SetB0ShapeReference(unsigned int indx) { _b0_shape_ref=indx; }
  void SetDWILocationReference(unsigned int indx) { _dwi_loc_ref=indx; }
  void SetShellLocationReference(unsigned int si, unsigned int indx) EddyTry {
    if (si>=_shell_loc_ref.size()) throw EddyException("ReferenceScans::SetShellLocationReference: Shell index out of range");
    _shell_loc_ref[si] = indx;
  } EddyCatch
  void SetShellShapeReference(unsigned int si, unsigned int indx) EddyTry {
    if (si>=_shell_shape_ref.size()) throw EddyException("ReferenceScans::SetShellShapeReference: Shell index out of range");
    _shell_shape_ref[si] = indx;
  } EddyCatch
private:
  /// All indicies are indicies of type ScanType ANY
  unsigned int                _loc_ref;          ///< Overall location reference scan
  unsigned int                _b0_loc_ref;       ///< Index for location reference b0 scan
  unsigned int                _b0_shape_ref;     ///< Index for shape reference b0 scan
  unsigned int                _dwi_loc_ref;      ///< Index for overall dwi location reference scan
  std::vector<unsigned int>   _shell_loc_ref;    ///< Indicies for shell location reference scans
  std::vector<unsigned int>   _shell_shape_ref;  ///< Indicies for shell shape reference scans
};

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//
// Class MaskManager
//
// This class manages an And-mask.
// scan.
//
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

class MaskManager
{
public:
  MaskManager(const NEWIMAGE::volume<float>& mask) EddyTry : _mask(mask) {} EddyCatch
  MaskManager(int xs, int ys, int zs) EddyTry : _mask(xs,ys,zs) { _mask = 1.0; } EddyCatch
  void ResetMask() EddyTry { _mask = 1.0; } EddyCatch
  void SetMask(const NEWIMAGE::volume<float>& mask) EddyTry { if (!NEWIMAGE::samesize(_mask,mask)) throw EddyException("EDDY::MaskManager::SetMask: Wrong dimension"); else _mask = mask;} EddyCatch
  void UpdateMask(const NEWIMAGE::volume<float>& mask) EddyTry { if (!NEWIMAGE::samesize(_mask,mask)) throw EddyException("EDDY::MaskManager::UpdateMask: Wrong dimension"); else _mask *= mask;} EddyCatch
  const NEWIMAGE::volume<float>& GetMask() const EddyTry { return(_mask); } EddyCatch
private:
  NEWIMAGE::volume<float> _mask;
};

/****************************************************************//**
*
* \brief This class manages stats on slice wise differences.
*
* This class calculates and serves up information about slice-wise
*  (in observation space) statistics on the difference between
*  an observation (scan) and the prediction.
*
********************************************************************/
class DiffStats
{
public:
  DiffStats() {}
  /// Constructs a Diffstats object given a difference volume and a mask.
  DiffStats(const NEWIMAGE::volume<float>& diff, const NEWIMAGE::volume<float>& mask);
  /// Returns the mean (across all valid voxels in the volume) difference.
  double MeanDifference() const EddyTry { return(mean_stat(_md)); } EddyCatch
  /// Returns the mean (across all valid voxels in slice sl (zero-offset)) difference.
  double MeanDifference(int sl) const EddyTry { if (index_ok(sl)) return(_md[sl]); else return(0.0); } EddyCatch
  /// Returns a vector with the mean difference for all slices
  NEWMAT::RowVector MeanDifferenceVector() const EddyTry { return(get_vector(_md)); } EddyCatch
  /// Returns the mean (across all valid voxels in the volume) squared difference.
  double MeanSqrDiff() const EddyTry { return(mean_stat(_msd)); } EddyCatch
  /// Returns the mean (across all valid voxels in slice sl (zero-offset)) squared difference.
  double MeanSqrDiff(int sl) const EddyTry { if (index_ok(sl)) return(_msd[sl]); else return(0.0); } EddyCatch
  /// Returns a vector with the mean squared difference for all slices
  NEWMAT::RowVector MeanSqrDiffVector() const EddyTry { return(get_vector(_msd)); } EddyCatch
  /// Number of valid voxels in the whole volume (as determined by the mask passed to the constructor)
  unsigned int NVox() const EddyTry { unsigned int n=0; for (int i=0; i<int(_n.size()); i++) n+=_n[i]; return(n); } EddyCatch
  /// Number of valid voxels in slice sl (zero offset).
  unsigned int NVox(int sl) const EddyTry { if (index_ok(sl)) return(_n[sl]); else return(0); } EddyCatch
  /// Vector with the number of valid voxels in each slice.
  NEWMAT::RowVector NVoxVector() const EddyTry { return(get_vector(_n)); } EddyCatch
  /// Number of slices.
  unsigned int NSlice() const EddyTry { return(_n.size()); } EddyCatch
private:
  std::vector<double>        _md;  // Slice wise mean difference
  std::vector<double>        _msd; // Slice wise mean squared difference
  std::vector<unsigned int>  _n;   // Slice wise # of valid pixels

  bool index_ok(int sl) const EddyTry
  { if (sl<0 || sl>=int(_n.size())) throw EddyException("DiffStats::index_ok: Index out of range"); return(true); } EddyCatch

  double mean_stat(const std::vector<double>& stat) const EddyTry
  { double ms=0; for (int i=0; i<int(_n.size()); i++) ms+=_n[i]*stat[i]; ms/=double(NVox()); return(ms); } EddyCatch

  template<class T>
  NEWMAT::RowVector get_vector(const std::vector<T>& stat) const EddyTry
  { NEWMAT::RowVector ov(stat.size()); for (unsigned int i=0; i<stat.size(); i++) ov(i+1) = double(stat[i]); return(ov); } EddyCatch
};

/****************************************************************//**
*
* \brief This class describes a multi-band structure
*
* This class describes a multi-band structure. Given a group index it
* will return a vector of slice indicies belonging to this group.
* In the simplest (degenerate) case with mb=1 it will simply return
* the same index.
*
********************************************************************/
class MultiBandGroups
{
public:
  MultiBandGroups(unsigned int nsl, unsigned int mb=1, int offs=0);
  /// Takes a file where each slice is numbered 0--n-1 and each row contains the slices acquired at a timepoint corresponding to row
  MultiBandGroups(const std::string& fname);
  /// Takes a NEWMAT matrix with the same content as described for the file above.
  MultiBandGroups(const NEWMAT::Matrix& mat);
  void SetTemporalOrder(const std::vector<unsigned int>& to) EddyTry {
    if (to.size() != _to.size()) throw EddyException("MultiBandGroups::SetTemporalOrder: to size mismatch"); else _to=to;
  } EddyCatch
  unsigned int NSlices() const { return(_nsl); }
  unsigned int MBFactor() const { return(_mb); }
  unsigned int NGroups() const EddyTry { return(_grps.size()); } EddyCatch
  unsigned int WhichGroupIsSliceIn(unsigned int sl) const EddyTry {
    if (sl >= _nsl) throw EddyException("MultiBandGroups::WhichGroupIsSliceIn: Slice index out of range");
    for (unsigned int g=0; g<_grps.size(); g++) {
      for (unsigned int sli=0; sli<_grps[g].size(); sli++) { if (_grps[g][sli] == sl) return(g); }
    }
    throw EddyException("MultiBandGroups::WhichGroupIsSliceIn: Slice not found");
  } EddyCatch
  const std::vector<unsigned int>& SlicesInGroup(unsigned int grp_i) const EddyTry {
    if (grp_i >= _grps.size()) throw EddyException("MultiBandGroups::SlicesInGroup: Group index out of range");
    else return(_grps[grp_i]);
  } EddyCatch
  const std::vector<unsigned int>& SlicesAtTimePoint(unsigned int time_i) const EddyTry {
    if (time_i >= _grps.size()) throw EddyException("MultiBandGroups::SlicesAtTimePoint: Time index out of range");
    else return(_grps[_to[time_i]]);
  } EddyCatch
  friend std::ostream& operator<<(std::ostream& os, const MultiBandGroups& mbg) EddyTry
  {
    for (unsigned int i=0; i<mbg._grps.size(); i++) {
      for (unsigned int j=0; j<mbg._grps[i].size(); j++) os << std::setw(5) << mbg._grps[i][j];
      os << std::endl;
    }
    return(os);
  } EddyCatch
private:
  unsigned int                            _nsl;  /// Number of slices
  unsigned int                            _mb;   /// Multi-band factor
  int                                     _offs; ///
  std::vector<std::vector<unsigned int> > _grps; ///
  /// Temporal order. For example if _to[0]==5 it means that the sixth slice/group was aquired first.
  std::vector<unsigned int>               _to;

  /// Checks _grps for internal consistency, like no duplicate slices, all slices accounted for etc.
  void assert_grps();
};

/****************************************************************//**
*
* \brief This class manages a set (one for each scan) of DiffStats objects.
*
* This class manages a vector of DiffStats objects (one for each scan).
* This means that it can look across scans (for a given slice) and
* build up statistics of the statistics from the DiffStats objects.
* It can for example calculate the mean and standard deviation (across)
* subjects of the slice-wise mean differences from the DiffStat objects.
* From that it can the determine how many standard deviations away
* a given scan and slice is from the mean and hence identify outliers.
*
********************************************************************/
class DiffStatsVector
{
public:
  /// Constructs an object with n (empty) slots for DiffStats objects.
  DiffStatsVector(unsigned int n) EddyTry : _n(n) { _ds = new DiffStats[_n]; } EddyCatch
  /// Copy constructor
  DiffStatsVector(const DiffStatsVector& rhs) EddyTry : _n(rhs._n) { _ds = new DiffStats[_n]; for (unsigned int i=0; i<_n; i++) _ds[i] = rhs._ds[i]; } EddyCatch
  ~DiffStatsVector() { delete [] _ds; }
  /// Assignment
  DiffStatsVector& operator=(const DiffStatsVector& rhs) EddyTry {
    delete [] _ds; _n = rhs._n; _ds = new DiffStats[_n]; for (unsigned int i=0; i<_n; i++) _ds[i] = rhs._ds[i]; return(*this);
  } EddyCatch
  /// Gives read-access to the ith (zero offset) DiffStats object in the vector.
  const DiffStats& operator[](unsigned int i) const EddyTry { throw_if_oor(i); return(_ds[i]); } EddyCatch
  /// Gives read/write-access to the ith (zero offset) DiffStats object in the vector. This is used to populate the vector.
  DiffStats& operator[](unsigned int i) EddyTry { throw_if_oor(i); return(_ds[i]); } EddyCatch
  /// Returns the number of DiffStats objects in the vector.
  unsigned int NScan() const { return(_n); }
  /// Returns the number of slices in each of the DiffStats objects.
  unsigned int NSlice() const EddyTry { return(_ds[0].NSlice()); } EddyCatch
  /// Returns the mean difference in scan sc, slice sl
  double MeanDiff(unsigned int sc, unsigned int sl) const EddyTry { throw_if_oor(sc); return(_ds[sc].MeanDifference(int(sl))); } EddyCatch
  /// Returns the mean square difference in scan sc, slice sl
  double MeanSqrDiff(unsigned int sc, unsigned int sl) const EddyTry { throw_if_oor(sc); return(_ds[sc].MeanSqrDiff(int(sl))); } EddyCatch
  /// Returns the number of "inside mask" voxels in scan sc, slice sl
  unsigned int NVox(unsigned int sc, unsigned int sl) const EddyTry { throw_if_oor(sc); return(_ds[sc].NVox(int(sl))); } EddyCatch
  /// Writes three files with information relevant for debugging.
  void Write(const std::string& bfname) const;
private:
  unsigned int _n;   // Length of vector
  DiffStats    *_ds; // Old fashioned C vector of DiffStats objects

  void throw_if_oor(unsigned int i) const EddyTry { if (i >= _n) throw EddyException("DiffStatsVector::throw_if_oor: Index out of range"); } EddyCatch
};

/****************************************************************//**
*
* \brief This class defines what is considered an outlier.
*
********************************************************************/
class OutlierDefinition {
public:
  OutlierDefinition(double        nstdev,    // # of std-dev away to qualify as outlier
		    unsigned int  minn,      // min # of intracerebral voxels to be considered
		    bool          pos,       // Flag also positive outliers if true
		    bool          sqr)       // Flag also sum-of-squares outliers if true
  : _nstdev(nstdev), _minn(minn), _pos(pos), _sqr(sqr) {}
  OutlierDefinition() : _nstdev(4.0), _minn(250), _pos(false), _sqr(false) {}
  double NStdDev() const { return(_nstdev); }
  unsigned int MinVoxels() const { return(_minn); }
  bool ConsiderPosOL() const { return(_pos); }
  bool ConsiderSqrOL() const { return(_sqr); }
private:
  double        _nstdev;    // # of std-dev away to qualify as outlier
  unsigned int  _minn;      // min # of intracerebral voxels to be considered
  bool          _pos;       // Flag also positive outliers if true
  bool          _sqr;       // Flag also sum-of-squares outliers if true
};

/****************************************************************//**
*
* \brief This class decides and keeps track of which slices in which
* scans should be replaced by their predictions
*
********************************************************************/
class ReplacementManager {
public:
  ReplacementManager(unsigned int              nscan,  // # of scans
		     unsigned int              nsl,    // # of slices
		     const OutlierDefinition&  old,    // Class defining an outlier
		     unsigned int              etc,    // =1 -> constant (across slices) type 1 error, =2 -> const type 2 error
		     OLType                    olt,   // If set, consider mb-groups instead of slices
		     const MultiBandGroups&    mbg)    // multi-band structure
  EddyTry : _old(old), _etc(etc), _olt(olt), _mbg(mbg), _sws(nsl,nscan), _gws(mbg.NGroups(),nscan), _swo(nsl,nscan), _gwo(mbg.NGroups(),nscan)
  {
  if (_etc != 1 && _etc != 2) throw  EddyException("ReplacementManager::ReplacementManager: etc must be 1 or 2");
  } EddyCatch
  ~ReplacementManager() {}
  unsigned int NSlice() const EddyTry { return(_swo._ovv.size()); } EddyCatch
  unsigned int NScan() const EddyTry { unsigned int rval = (_swo._ovv.size()) ? _swo._ovv[0].size() : 0; return(rval); } EddyCatch
  unsigned int NGroup() const EddyTry { return(_mbg.NGroups()); } EddyCatch
  void Update(const DiffStatsVector& dsv);
  std::vector<unsigned int> OutliersInScan(unsigned int scan) const;
  bool ScanHasOutliers(unsigned int scan) const;
  bool IsAnOutlier(unsigned int slice, unsigned int scan) const EddyTry { return(_swo._ovv[slice][scan]); } EddyCatch
  void WriteReport(const std::vector<unsigned int>& i2i,
		   const std::string&               bfname) const;
  void WriteMatrixReport(const std::vector<unsigned int>& i2i,
			 unsigned int                     nscan,
			 const std::string&               om_fname,
			 const std::string&               nstdev_fname,
			 const std::string&               n_sqr_stdev_fname) const;
  // For debugging
  void DumpOutlierMaps(const std::string& fname) const;
  // Struct that is instantiated in the private section
  struct OutlierInfo {
    std::vector<std::vector<bool> >           _ovv;     // _ovv[slice/group][scan] is an outlier if set
    std::vector<std::vector<double> >         _nsv;     // _nsv[slice/group][scan] tells how many stdev off that slice-scan is.
    std::vector<std::vector<double> >         _nsq;     // _nsq[slice/group][scan] tells how many stdev off the sum-of-squared differences of that slice-scan is.
    OutlierInfo(unsigned int nsl, unsigned int nscan) EddyTry : _ovv(nsl), _nsv(nsl), _nsq(nsl) {
      for (unsigned int i=0; i<nsl; i++) { _ovv[i].resize(nscan,false); _nsv[i].resize(nscan,0.0); _nsq[i].resize(nscan,0.0); }
    } EddyCatch
  };
  // Struct that is instantiated in the private section
  struct StatsInfo {
    std::vector<std::vector<unsigned int> >   _nvox;    // _nvox[slice/group][scan] is # of valid voxels in that slice-scan
    std::vector<std::vector<double> >         _mdiff;   // _mdiff[slice/group][scan] is the mean difference in that slice-scan
    std::vector<std::vector<double> >         _msqrd;   // _msqrd[slice/group][scan] is the mean squared difference in that slice-scan
    StatsInfo(unsigned int nsl, unsigned int nscan) EddyTry : _nvox(nsl), _mdiff(nsl), _msqrd(nsl) {
      for (unsigned int i=0; i<nsl; i++) { _nvox[i].resize(nscan,0); _mdiff[i].resize(nscan,0.0); _msqrd[i].resize(nscan,0.0); }
    } EddyCatch
  };
private:
  OutlierDefinition                 _old;     // Class defining an outlier
  unsigned int                      _etc;     // ErrorTypeConstant (keep type 1 or type 2 errors constant)
  OLType                            _olt;     // Loook for outliers slicewise, groupwise or both.
  MultiBandGroups                   _mbg;     // Structure of mb-groups
  StatsInfo                         _sws;     // Slice-wise stats
  StatsInfo                         _gws;     // Group-wise stats
  OutlierInfo                       _swo;     // Slice-wise outlier info
  OutlierInfo                       _gwo;     // group-wise outlier info

  void throw_if_oor(unsigned int scan) const EddyTry { if (scan >= this->NScan()) throw EddyException("ReplacementManager::throw_if_oor: Scan index out of range"); } EddyCatch
  double sqr(double a) const EddyTry { return(a*a); } EddyCatch
  std::pair<double,double> mean_and_std(const EDDY::ReplacementManager::StatsInfo& sws, unsigned int minvox, unsigned int etc,
					const std::vector<std::vector<bool> >& ovv, std::pair<double,double>& stdev) const;
};

/****************************************************************//**
*
* \brief Helper class that manages a set of image coordinates in a way
* that, among other things, enables calculation/implementation of partial
* derivatives of images w.r.t. transformation parameters.
*
********************************************************************/
class ImageCoordinates
{
public:
  ImageCoordinates(const NEWIMAGE::volume<float>& ima)
  EddyTry : ImageCoordinates(static_cast<unsigned int>(ima.xsize()),static_cast<unsigned int>(ima.ysize()),static_cast<unsigned int>(ima.zsize())) {} EddyCatch
  ImageCoordinates(unsigned int xn, unsigned int yn, unsigned int zn)
  EddyTry : _xn(xn), _yn(yn), _zn(zn)
  {
    _x = new float[_xn*_yn*_zn];
    _y = new float[_xn*_yn*_zn];
    _z = new float[_xn*_yn*_zn];
    for (unsigned int k=0, indx=0; k<_zn; k++) {
      for (unsigned int j=0; j<_yn; j++) {
	for (unsigned int i=0; i<_xn; i++) {
	  _x[indx] = float(i);
	  _y[indx] = float(j);
	  _z[indx++] = float(k);
	}
      }
    }
  } EddyCatch
  ImageCoordinates(const ImageCoordinates& inp)
  EddyTry : _xn(inp._xn), _yn(inp._yn), _zn(inp._zn)
  {
    _x = new float[_xn*_yn*_zn]; std::memcpy(_x,inp._x,_xn*_yn*_zn*sizeof(float));
    _y = new float[_xn*_yn*_zn]; std::memcpy(_y,inp._y,_xn*_yn*_zn*sizeof(float));
    _z = new float[_xn*_yn*_zn]; std::memcpy(_z,inp._z,_xn*_yn*_zn*sizeof(float));
  } EddyCatch
  ImageCoordinates(ImageCoordinates&& inp)
  EddyTry : _xn(inp._xn), _yn(inp._yn), _zn(inp._zn)
  {
    _x = inp._x; inp._x = nullptr;
    _y = inp._y; inp._y = nullptr;
    _z = inp._z; inp._z = nullptr;
  } EddyCatch
  ~ImageCoordinates() { delete[] _x; delete[] _y; delete[] _z; }
  ImageCoordinates& operator=(const ImageCoordinates& rhs) EddyTry {
    if (this == &rhs) return(*this);
    delete[] _x; delete[] _y; delete[] _z;
    _xn = rhs._xn; _yn = rhs._yn; _zn = rhs._zn;
    _x = new float[_xn*_yn*_zn]; std::memcpy(_x,rhs._x,_xn*_yn*_zn*sizeof(float));
    _y = new float[_xn*_yn*_zn]; std::memcpy(_y,rhs._y,_xn*_yn*_zn*sizeof(float));
    _z = new float[_xn*_yn*_zn]; std::memcpy(_z,rhs._z,_xn*_yn*_zn*sizeof(float));
    return(*this);
  } EddyCatch
  ImageCoordinates& operator=(ImageCoordinates&& rhs) EddyTry {
    if (this != &rhs) {
      delete[] _x; delete[] _y; delete[] _z;
      _xn = rhs._xn; _yn = rhs._yn; _zn = rhs._zn;
      _x = rhs._x; rhs._x = nullptr;
      _y = rhs._y; rhs._y = nullptr;
      _z = rhs._z; rhs._z = nullptr;
    }
    return(*this);
  } EddyCatch
  ImageCoordinates& operator+=(const ImageCoordinates& rhs) EddyTry {
    if (_xn != rhs._xn || _yn != rhs._yn || _zn != rhs._zn) throw EddyException("ImageCoordinates::operator-= size mismatch");
    for (unsigned int i=0; i<_xn*_yn*_zn; i++) { _x[i]+=rhs._x[i]; _y[i]+=rhs._y[i]; _z[i]+=rhs._z[i]; }
    return(*this);
  } EddyCatch
  ImageCoordinates& operator-=(const ImageCoordinates& rhs) EddyTry {
    if (_xn != rhs._xn || _yn != rhs._yn || _zn != rhs._zn) throw EddyException("ImageCoordinates::operator-= size mismatch");
    for (unsigned int i=0; i<_xn*_yn*_zn; i++) { _x[i]-=rhs._x[i]; _y[i]-=rhs._y[i]; _z[i]-=rhs._z[i]; }
    return(*this);
  } EddyCatch
  ImageCoordinates operator+(const ImageCoordinates& rhs) const EddyTry {
    return(ImageCoordinates(*this)+=rhs);
  } EddyCatch
  ImageCoordinates operator-(const ImageCoordinates& rhs) const EddyTry {
    return(ImageCoordinates(*this)-=rhs);
  } EddyCatch
  ImageCoordinates& operator/=(double div) EddyTry {
    if (div==0) throw EddyException("ImageCoordinates::operator/= attempt to divide by zero");
    for (unsigned int i=0; i<_xn*_yn*_zn; i++) { _x[i]/=div; _y[i]/=div; _z[i]/=div; }
    return(*this);
  } EddyCatch
  NEWIMAGE::volume<float> operator*(const NEWIMAGE::volume4D<float>& vol) EddyTry {
    if (int(_xn) != vol.xsize() || int(_yn) != vol.ysize() || int(_zn) != vol.zsize() || vol.tsize() != 3) {
      throw EddyException("ImageCoordinates::operator* size mismatch");
    }
    NEWIMAGE::volume<float> ovol = vol[0];
    for (unsigned int k=0, indx=0; k<_zn; k++) {
      for (unsigned int j=0; j<_yn; j++) {
	for (unsigned int i=0; i<_xn; i++) {
          ovol(i,j,k) = _x[indx]*vol(i,j,k,0) + _y[indx]*vol(i,j,k,1) + _z[indx]*vol(i,j,k,2);
          indx++;
	}
      }
    }
    return(ovol);
  } EddyCatch
  void Transform(const NEWMAT::Matrix& M) EddyTry {
    if (M.Nrows() != 4 || M.Ncols() != 4) throw EddyException("ImageCoordinates::Transform: Matrix M must be 4x4");
    float M11 = M(1,1); float M12 = M(1,2); float M13 = M(1,3); float M14 = M(1,4);
    float M21 = M(2,1); float M22 = M(2,2); float M23 = M(2,3); float M24 = M(2,4);
    float M31 = M(3,1); float M32 = M(3,2); float M33 = M(3,3); float M34 = M(3,4);
    float *xp = _x; float *yp = _y; float *zp = _z;
    for (unsigned int i=0; i<N(); i++) {
      float ox = M11 * *xp + M12 * *yp + M13 * *zp + M14;
      float oy = M21 * *xp + M22 * *yp + M23 * *zp + M24;
      float oz = M31 * *xp + M32 * *yp + M33 * *zp + M34;
      *xp = ox; *yp = oy; *zp = oz;
      xp++; yp++; zp++;
    }
  } EddyCatch
  void Transform(const std::vector<NEWMAT::Matrix>&             M,    // Array of matrices
		 const std::vector<std::vector<unsigned int> >& grps) // Array of MB-groups of slices
  EddyTry {
    if (M.size() != grps.size()) throw EddyException("ImageCoordinates::Transform: Mismatch between M and grps");
    for (unsigned int grp=0; grp<grps.size(); grp++) {
      if (M[grp].Nrows() != 4 || M[grp].Ncols() != 4) throw EddyException("ImageCoordinates::Transform: All Matrices M must be 4x4");
      std::vector<unsigned int> slices = grps[grp];
      float M11 = M[grp](1,1); float M12 = M[grp](1,2); float M13 = M[grp](1,3); float M14 = M[grp](1,4);
      float M21 = M[grp](2,1); float M22 = M[grp](2,2); float M23 = M[grp](2,3); float M24 = M[grp](2,4);
      float M31 = M[grp](3,1); float M32 = M[grp](3,2); float M33 = M[grp](3,3); float M34 = M[grp](3,4);
      for (unsigned int i=0; i<slices.size(); i++) {
	for (unsigned int indx=slstart(slices[i]); indx<slend(slices[i]); indx++) {
	  float ox = M11 * _x[indx] + M12 * _y[indx] + M13 * _z[indx] + M14;
	  float oy = M21 * _x[indx] + M22 * _y[indx] + M23 * _z[indx] + M24;
	  float oz = M31 * _x[indx] + M32 * _y[indx] + M33 * _z[indx] + M34;
	  _x[indx] = ox; _y[indx] = oy; _z[indx] = oz;
	}
      }
    }
  } EddyCatch
  ImageCoordinates MakeTransformed(const NEWMAT::Matrix& M) const EddyTry {
    ImageCoordinates rval = *this;
    rval.Transform(M);
    return(rval);
  } EddyCatch
  ImageCoordinates MakeTransformed(const std::vector<NEWMAT::Matrix>&             M,
				   const std::vector<std::vector<unsigned int> >& grps) const EddyTry {
    ImageCoordinates rval = *this;
    rval.Transform(M,grps);
    return(rval);
  } EddyCatch
  void Write(const std::string& fname) const EddyTry
  {
    NEWMAT::Matrix omat(N(),3);
    for (unsigned int i=0; i<N(); i++) {
      omat(i+1,1) = x(i); omat(i+1,2) = y(i); omat(i+1,3) = z(i);
    }
    MISCMATHS::write_ascii_matrix(fname,omat);
  } EddyCatch

  unsigned int N() const EddyTry { return(_xn*_yn*_zn); } EddyCatch
  unsigned int NX() const EddyTry { return(_xn); } EddyCatch
  unsigned int NY() const EddyTry { return(_yn); } EddyCatch
  unsigned int NZ() const EddyTry { return(_zn); } EddyCatch
  bool IsInBounds(unsigned int i) const EddyTry { return(_x[i] >= 0 && _x[i] <= (_xn-1) && _y[i] >= 0 && _y[i] <= (_yn-1) && _z[i] >= 0 && _z[i] <= (_zn-1)); }
EddyCatch
  const float& x(unsigned int i) const EddyTry { return(_x[i]); } EddyCatch
  const float& y(unsigned int i) const EddyTry { return(_y[i]); } EddyCatch
  const float& z(unsigned int i) const EddyTry { return(_z[i]); } EddyCatch
  float& x(unsigned int i) EddyTry { return(_x[i]); } EddyCatch
  float& y(unsigned int i) EddyTry { return(_y[i]); } EddyCatch
  float& z(unsigned int i) EddyTry { return(_z[i]); } EddyCatch

private:
  unsigned int _xn;
  unsigned int _yn;
  unsigned int _zn;
  float *_x;
  float *_y;
  float *_z;
  unsigned int slstart(unsigned int sl) const EddyTry { return(sl*_xn*_yn); } EddyCatch
  unsigned int slend(unsigned int sl) const EddyTry { return((sl+1)*_xn*_yn); } EddyCatch
};

/****************************************************************//**
*
* \brief Helper class that manages histograms and calculates
* mutual information.
*
********************************************************************/
class MutualInfoHelper
{
public:
  MutualInfoHelper(unsigned int nbins) EddyTry : _nbins(nbins), _lset(false) {
    _mhist1 = new double[_nbins];
    _mhist2 = new double[_nbins];
    _jhist = new double[_nbins*_nbins];
  } EddyCatch
  MutualInfoHelper(unsigned int nbins, float min1, float max1, float min2, float max2) EddyTry
    : _nbins(nbins), _min1(min1), _max1(max1), _min2(min2), _max2(max2), _lset(true) {
    _mhist1 = new double[_nbins];
    _mhist2 = new double[_nbins];
    _jhist = new double[_nbins*_nbins];
  } EddyCatch
  virtual ~MutualInfoHelper() { delete[] _mhist1; delete[] _mhist2; delete[] _jhist; }
  void SetLimits(float min1, float max1, float min2, float max2) EddyTry {
    _min1 = min1; _max1 = max1; _min2 = min2; _max2 = max2; _lset = true;
  } EddyCatch
  double MI(const NEWIMAGE::volume<float>& ima1,
	    const NEWIMAGE::volume<float>& ima2,
	    const NEWIMAGE::volume<float>& mask) const;
  double SoftMI(const NEWIMAGE::volume<float>& ima1,
		const NEWIMAGE::volume<float>& ima2,
		const NEWIMAGE::volume<float>& mask) const;
private:
  double plogp(double p) const EddyTry { if (p) return( - p*std::log(p)); else return(0.0); } EddyCatch
  unsigned int val_to_indx(float val, float min, float max, unsigned int nbins) const EddyTry
  {
    int tmp = static_cast<int>((val-min)*static_cast<float>(nbins-1)/(max-min) + 0.5);
    if (tmp < 0) tmp = 0;
    else if (static_cast<unsigned int>(tmp) > (nbins-1)) tmp = nbins-1;
    return(static_cast<unsigned int>(tmp));
  } EddyCatch
  unsigned int val_to_floor_indx(float val, float min, float max, unsigned int nbins, float *rem) const EddyTry
  {
    unsigned int rval=0;
    float x = (val-min)*static_cast<float>(nbins)/(max-min); // 0 <= x <= nbins for min <= val <= max
    if (x <= 0.5) { *rem = 0.0; rval = 0; }
    else if (x >= static_cast<float>(nbins-0.5)) { *rem = 0.0; rval = nbins - 1; }
    else { rval = static_cast<unsigned int>(x-0.5); *rem = x - 0.5 - static_cast<float>(rval); }
    return(rval);
  } EddyCatch

  unsigned int          _nbins;   /// No. of bins of histograms
  float                 _min1;    /// Minimum of ima1
  float                 _max1;    /// Maximum of ima1
  float                 _min2;    /// Minimum of ima2
  float                 _max2;    /// Maximum of ima2
  bool                  _lset;    /// True if the _min _max values are to be used
  mutable double        *_mhist1; /// Marginal histogram of ima1
  mutable double        *_mhist2; /// Marginal histogram of ima2
  mutable double        *_jhist;  /// Joint histogram of ima1 and ima2
};

/****************************************************************//**
*
* \brief Helper class that repacks from a stack of 2D images and a
* stack of z-coords to a format suitable for doing a scattered
* data recon GP-prediction.
*
********************************************************************/
class Stacks2YVecsAndWgts
{
public:
  Stacks2YVecsAndWgts(unsigned int zsize, unsigned int tsize) : _y(zsize), _wgt(zsize), _sqrtwgt(zsize), _bvi(zsize), _n(zsize,0)
  {
    for (unsigned int i=0; i<zsize; i++) {
      _y[i].resize(3*tsize,0.0);
      _wgt[i].resize(3*tsize,0.0);
      _sqrtwgt[i].resize(3*tsize,0.0);
      _bvi[i].resize(3*tsize,std::make_pair(-1,-1));
    }
  }
  /// Makes arrays of y-vecs, wgt-vecs and bvec/bvals index-vecs for given [i,j] pair in volume
  void MakeVectors(const NEWIMAGE::volume4D<float>& stacks,
		   const NEWIMAGE::volume4D<float>& masks,
		   const NEWIMAGE::volume4D<float>& zcoord,
		   unsigned int                     i,
		   unsigned int                     j);
  NEWMAT::ColumnVector YVec(unsigned int indx) const {
    if (indx >= _y.size()) throw EddyException("Stacks2YVecsAndWgts::YVec: indx out of range");
    return(return_vec(_y[indx],_n[indx]));
  }
  NEWMAT::ColumnVector Wgt(unsigned int indx) const {
    if (indx >= _wgt.size()) throw EddyException("Stacks2YVecsAndWgts::Wgt: indx out of range");
    return(return_vec(_wgt[indx],_n[indx]));
  }
  NEWMAT::ColumnVector SqrtWgt(unsigned int indx) const {
    if (indx >= _sqrtwgt.size()) throw EddyException("Stacks2YVecsAndWgts::SqrtWgt: indx out of range");
    return(return_vec(_sqrtwgt[indx],_n[indx]));
  }
  std::vector<double> StdSqrtWgt(unsigned int indx) const {
    if (indx >= _sqrtwgt.size()) throw EddyException("Stacks2YVecsAndWgts::StdSqrtWgt: indx out of range");
    std::vector<double> rval(_n[indx]); std::copy_n(_sqrtwgt[indx].begin(),_n[indx],rval.begin());
    return(rval);
  }
  NEWMAT::ColumnVector SqrtWgtYVec(unsigned int indx) const {
    if (indx >= _y.size()) throw EddyException("Stacks2YVecsAndWgts::SqrtWgtYVec: indx out of range");
    return(NEWMAT::SP(return_vec(_sqrtwgt[indx],_n[indx]),return_vec(_y[indx],_n[indx])));
  }
  const std::vector<std::pair<int,int> >& Indx(unsigned int indx) const { return(_bvi[indx]); }
  unsigned int NVal(unsigned int indx) const { return(_n[indx]); }
  unsigned int NVox() const { return(_y.size()); }
private:
  std::vector<std::vector<double> >              _y;       // array of y-vectors
  std::vector<std::vector<double> >              _wgt;     // array of weight-vectors
  std::vector<std::vector<double> >              _sqrtwgt; // array of square root of weight-vectors
  std::vector<std::vector<std::pair<int,int> > > _bvi;     // array of vectors of indicies into volume and slice
  std::vector<unsigned int>                      _n;       // Number of elements for a given voxel

  NEWMAT::ColumnVector return_vec(const std::vector<double>& vec,
				  unsigned int               n) const
  {
    NEWMAT::ColumnVector rval(n); for (unsigned int i=0; i<n; i++) rval(i+1) = vec[i]; return(rval);
  }
};

/****************************************************************//**
*
* \brief Helper class that turns a vector of volume and slice
* indices into a (weighted) K-matrix.
*
********************************************************************/
class Indicies2KMatrix
{
public:
  Indicies2KMatrix(const std::vector<std::vector<NEWMAT::ColumnVector> >& bvecs,
		   const std::vector<unsigned int>&                       grpi,
		   const std::vector<double>&                             grpb,
		   const std::vector<std::pair<int,int> >&                indx,
		   unsigned int                                           nval,
		   const std::vector<double>&                             hpar) EddyTry {
    common_construction(bvecs,grpi,grpb,indx,nval,hpar,nullptr);
  } EddyCatch

  Indicies2KMatrix(const std::vector<std::vector<NEWMAT::ColumnVector> >& bvecs,
		   const std::vector<unsigned int>&                       grpi,
		   const std::vector<double>&                             grpb,
		   const std::vector<std::pair<int,int> >&                indx,
		   const std::vector<double>&                             wgt,
		   unsigned int                                           nval,
		   const std::vector<double>&                             hpar) EddyTry {
    common_construction(bvecs,grpi,grpb,indx,nval,hpar,&wgt);
  } EddyCatch
  const NEWMAT::Matrix& GetKMatrix() const { return(_K); }
  NEWMAT::RowVector GetkVector(const NEWMAT::ColumnVector& bvec,
			       unsigned int                grp) const;
private:
  std::vector<NEWMAT::ColumnVector> _bvecs;
  std::vector<double>               _grpb;
  std::vector<double>               _log_grpb;
  std::vector<unsigned int>         _grpi;
  std::vector<double>               _hpar;
  std::vector<double>               _thpar;
  std::vector<double>               _wgt;
  NEWMAT::Matrix                    _K;
  void common_construction(const std::vector<std::vector<NEWMAT::ColumnVector> >& bvecs,
			   const std::vector<unsigned int>&                       grpi,
			   const std::vector<double>&                             grpb,
			   const std::vector<std::pair<int,int> >&                indx,
			   unsigned int                                           nval,
			   const std::vector<double>&                             hpar,
			   const std::vector<double>                              *wgt);
};

} // End namespace EDDY

#endif // End #ifndef EddyHelperClasses_h

////////////////////////////////////////////////
//
// Here starts Doxygen documentation
//
////////////////////////////////////////////////

/*!
 * \fn EDDY::DiffPara::DiffPara(const NEWMAT::ColumnVector& bvec, double bval)
 * Contructs a DiffPara object from a b-vector and a b-value.
 * \param bvec ColumnVector with three elements. Will be normalised, but must have norm different from zero.
 * \param bval b-value. Must be non-negative.
 */

/*!
 * \fn EDDY::DiffPara::operator==(const DiffPara& rhs) const
 *  Will return true if calls to both EddyUtils::AreInSameShell and EddyUtils::HaveSameDirection are true.
 */

/*!
 * \fn AcqPara::AcqPara(const NEWMAT::ColumnVector& pevec, double rotime)
 * Constructs an AcqPara object from phase-encode direction and total read-out time.
 * \param pevec Normalised vector desribing the direction of the phase-encoding. At present the third element
 * (the z-direction) must be zero (i.e. it only allows phase-encoding in the xy-plane.
 * \param rotime The time between the collection of the midpoint of the first echo and the last echo.
 */
