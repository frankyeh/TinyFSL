// Declarations of cost-function classes used by topup
//
// topup_costfunctions.h
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2009 University of Oxford
//
/*  CCOPYRIGHT  */
#ifndef topup_costfunctions_h
#define topup_costfunctions_h

#include <string>
#include <vector>
#include <memory>
#include "armawrap/newmat.h"
#include "miscmaths/bfmatrix.h"
#include "miscmaths/nonlin.h"
#include "basisfield/basisfield.h"
#include "basisfield/splinefield.h"
#include "basisfield/dctfield.h"

namespace TOPUP {

class TopupException: public std::exception
{
private:
  std::string m_msg;
public:
  TopupException(const std::string& msg) throw(): m_msg(msg) {}

  virtual const char * what() const throw() {
    return std::string("Topup: msg=" + m_msg).c_str();
  }

  ~TopupException() throw() {}
};

enum SizeType {Original=0, Regridded=1, Subsampled=2, Target=3};
enum RegularisationType {MembraneEnergy, BendingEnergy};
enum TopupInterpolationType {LinearInterp, SplineInterp, UnknownInterp};

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//
// Class TiledMatrix
//
// Helper class to organize the creation/calculation of the
// movement part of the Hessian.
//
// {{{ @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

class TiledMatrix
{
public:
  TiledMatrix(const std::vector<unsigned int>& tile_sizes)
  : _ts(tile_sizes)
  {
    unsigned int tots = _ts[0];
    for (unsigned int i=1; i<_ts.size(); i++) tots += _ts[i];
    _m.ReSize(tots,tots);
    _m = 0.0;
  }
  ~TiledMatrix() {}
  unsigned int NoOfTiles() const { return(_ts.size()); }
  unsigned int TileSizeRow(unsigned int r, unsigned int c) const {
    if (r>=_ts.size() || c>=_ts.size()) throw TopupException("TiledMatrix::TileSizeRow: Index out of range"); return(_ts[r]);
  }
  unsigned int TileSizeCol(unsigned int r, unsigned int c) const {
    if (r>=_ts.size() || c>=_ts.size()) throw TopupException("TiledMatrix::TileSizeCol: Index out of range"); return(_ts[c]);
  }
  // Read and read-write access functions. N.B. that the tiles are
  // indexed from 0 to no_tiles-1, i.e. C-convention rather than NEWMAT.
  NEWMAT::Matrix GetTile(unsigned int r, unsigned int c) const {
    if (r>=_ts.size() || c>=_ts.size()) throw TopupException("TiledMatrix::GetTile: Index out of range");
    unsigned int rs = 0; for (unsigned i=0; i<r; i++) rs += _ts[i];
    unsigned int cs = 0; for (unsigned i=0; i<c; i++) cs += _ts[i];
    return(_m.SubMatrix(rs+1,rs+_ts[r],cs+1,cs+_ts[c]));
  }
  void SetTile(unsigned int r, unsigned int c, const NEWMAT::Matrix& tile) // Read-write
  {
    if (r>=_ts.size() || c>=_ts.size()) throw TopupException("TiledMatrix::SetTile: Index out of range");
    if (tile.Nrows() != int(_ts[r]) || tile.Ncols() != int(_ts[c])) throw TopupException("TiledMatrix::SetTile: Wrong size matrix");
    unsigned int rs = 0; for (unsigned i=0; i<r; i++) rs += _ts[i];
    unsigned int cs = 0; for (unsigned i=0; i<c; i++) cs += _ts[i];
    _m.SubMatrix(rs+1,rs+_ts[r],cs+1,cs+_ts[c]) = tile;
  }
  const NEWMAT::Matrix& Untile() const { return(_m); }
private:
  std::vector<unsigned int> _ts;
  NEWMAT::Matrix            _m;
};

/*
class TiledMatrix
{
public:
  TiledMatrix(unsigned int tile_size_rows,
              unsigned int tile_size_columns,
              unsigned int no_tiles_rows,
              unsigned int no_tiles_columns)
  : _tsr(tile_size_rows), _tsc(tile_size_columns), _ntr(no_tiles_rows), _ntc(no_tiles_columns), _m(_tsr*_ntr,_tsc*_ntc)
  {
    _m = 0.0;
  }
  ~TiledMatrix() {}
  unsigned int NoOfTilesRow() const { return(_ntr); }
  unsigned int NoOfTilesCol() const { return(_ntc); }
  unsigned int TileSizeRow() const { return(_tsr); }
  unsigned int TileSizeCol() const { return(_tsc); }
  // Read and read-write access functions. N.B. that the tiles are
  // indexed from 0 to no_tiles-1, i.e. C-convention rather than NEWMAT.
  NEWMAT::Matrix GetTile(unsigned int ri, unsigned int ci) const
  {
    if (ri>=_ntr || ci>=_ntc) throw TopupException("TiledMatrix::GetTile: Index out of range");
    return(_m.SubMatrix(ri*_tsr+1,(ri+1)*_tsr,ci*_tsc+1,(ci+1)*_tsc));
  }
  void SetTile(unsigned int ri, unsigned int ci, const NEWMAT::Matrix& tile) // Read-write
  {
    if (ri>=_ntr || ci>=_ntc) throw TopupException("TiledMatrix::SetTile: Index out of range");
    if (tile.Nrows() != int(_tsr) || tile.Ncols() != int(_tsc)) throw TopupException("TiledMatrix::SetTile: Index out of range");
    _m.SubMatrix(ri*_tsr+1,(ri+1)*_tsr,ci*_tsc+1,(ci+1)*_tsc) = tile;
  }
  const NEWMAT::Matrix& Untile() const { return(_m); }

private:
  unsigned int     _tsr;
  unsigned int     _tsc;
  unsigned int     _ntr;
  unsigned int     _ntc;
  NEWMAT::Matrix   _m;
};
*/
// }}} End of fold.

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//
// Class TopupScan
//
// Has the data, acquisition info and movement parameters for
// one scan. Responsible for managing the scan, resampling the
// scan (and serve it up) and keeping track of update status.
//
// {{{ @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

class TopupScan
{
public:
  TopupScan(const NEWIMAGE::volume<float>& scan, const NEWMAT::ColumnVector& pevec, double rotime);
  ~TopupScan() {}
  std::vector<unsigned int> ImageSize(SizeType type=Original) const;
  std::vector<double>       ImageVxs(SizeType type=Original) const;
  bool TracePrint() const { return(_tp); }
  void SetTracePrint(bool val=true) { _tp=val; }
  bool UpToDate() const { return(_uptodate); }
  bool HasBeta() const { if (_pevec(1)) return(true); else return(false); }
  bool HasGamma() const { if (_pevec(2)) return(true); else return(false); }
  NEWMAT::ColumnVector GetPhaseEncodeDirection() const { return(_pevec); }
  NEWIMAGE::volume<float> GetResampled(const BASISFIELD::splinefield& field) const;
  NEWIMAGE::volume<char>  GetMask(const BASISFIELD::splinefield& field) const;
  NEWIMAGE::volume<float> GetAlpha(const BASISFIELD::splinefield& field) const;
  NEWIMAGE::volume<float> GetBeta(const BASISFIELD::splinefield& field) const;
  NEWIMAGE::volume<float> GetGamma(const BASISFIELD::splinefield& field) const;


  inline float GetResampled(size_t index) const
  {return _jac.at(index) * _resampled.at(index);}

  inline float GetMask(size_t index) const
  {return _mask.at(index);}

  inline float GetAlpha(size_t index,float s0,float s1) const
  {
      if(s0 == 0.0f) return s1 * _derivs[1].at(index) * _jac.at(index);
      if(s1 == 0.0f) return s0 * _derivs[0].at(index) * _jac.at(index);
      return (s1 * _derivs[1].at(index) + s0 * _derivs[0].at(index)) * _jac.at(index);
  }
  inline float GetBeta(size_t index,float s) const
  {return (s != 0.0f) ? s * _resampled.at(index) : 0.0f;}

  inline float GetGamma(size_t index,float s) const
  {return (s != 0.0f) ? s * _resampled.at(index) : 0.0f;}


  NEWIMAGE::volume<float> GetMovementDerivative(unsigned int i,
                                                const BASISFIELD::splinefield& field) const;
  NEWIMAGE::volume<float> GetNumericalMovementDerivative(unsigned int i,
                                                         const BASISFIELD::splinefield& field) const;
  NEWIMAGE::volume<float> GetJacobian(const BASISFIELD::splinefield& field) const;
  NEWIMAGE::volume4D<float> GetDisplacementField(const BASISFIELD::splinefield&  field) const;
  NEWMAT::ColumnVector GetMovementParameters() const { return(_mp); }  // Will always serve up six elements
  NEWMAT::Matrix GetRigidBodyMatrix() const { return(mp_to_matrix(_mp)); }
  void SetUpToDate(bool flag) const { _uptodate=flag; }
  void SetMovementParameters(const NEWMAT::ColumnVector& mp) const;    // mp must contain six elements
  void SetInterpolationModel(TopupInterpolationType it) const;
  void ReGrid(int xsz, int ysz, int zsz);
  void ReGrid(const std::vector<unsigned int>& sz) { ReGrid(int(sz[0]),int(sz[1]),int(sz[2])); }
  void Smooth(double fwhm);
  void SubSample(unsigned int ss);
//private:
  std::shared_ptr<NEWIMAGE::volume<float> >   _orig;
  std::shared_ptr<NEWIMAGE::volume<float> >   _regrid;      // May be identical to _orig
  std::shared_ptr<NEWIMAGE::volume<float> >   _subsamp;     // May be identical to _regrid
  std::shared_ptr<NEWIMAGE::volume<float> >   _smooth;      // May be identical to _subsamp
  mutable NEWIMAGE::volume<float>               _resampled;
  mutable NEWIMAGE::volume4D<float>             _derivs;
  mutable NEWIMAGE::volume<float>               _jac;
  mutable NEWIMAGE::volume<char>                _mask;
  NEWMAT::ColumnVector                          _pevec;
  double                                        _rotime;
  mutable bool                                  _uptodate;
  mutable NEWMAT::ColumnVector                  _mp;        // dx dy dz rx ry rz, N.B. different from MJ
  bool                                          _tp;        // Trace-print

  NEWMAT::Matrix mp_to_matrix(const NEWMAT::ColumnVector& mp) const;
  NEWIMAGE::interpolation translate_interp_type(TopupInterpolationType it) const { if (it==LinearInterp) return(NEWIMAGE::trilinear); else return(NEWIMAGE::spline); }
  TopupInterpolationType translate_interp_type(NEWIMAGE::interpolation it) const { if (it==NEWIMAGE::trilinear) return(LinearInterp); else if (it==NEWIMAGE::spline) return(SplineInterp); else return(UnknownInterp); }

  void update(const BASISFIELD::splinefield& field) const; // This is where all the work gets done
};

// }}} End of fold.

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//
// Class TopupScanManager
//
// Responsible for managing the scans, updating the resampled ones
// as neccessary, keep track of update-status and to serve up
// resampled scans and a mask that is the intersection of the
// individual masks.
//
// {{{ @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

class TopupScanManager
{
public:
  TopupScanManager(const NEWIMAGE::volume4D<float>&   scans,
                   const NEWMAT::Matrix&              pevecs,
                   const NEWMAT::ColumnVector&        rotimes);
  ~TopupScanManager();

  // Routines that request information about the scans
  unsigned int NoOfScans() const { return(_scans.size()); }
  // unsigned int NoOfMovementParameters() const { return((NoOfScans()-1) * _mpi.size()); }
  unsigned int NoOfMovementParameters() const { unsigned int n=_mpi[0].size(); for (unsigned int i=1; i<NoOfScans(); i++) n+=_mpi[i].size(); return(n); }
  unsigned int NoOfMovementParametersForScan(unsigned int scan) const;
  NEWMAT::ReturnMatrix GetMovementParameters() const;    // Returns the movement parameters that are being estimated
  NEWMAT::ReturnMatrix GetAllMovementParameters() const; // Returns ALL movement paramaters, including those held at zero
  NEWMAT::ReturnMatrix GetRigidBodyMatrix(unsigned int i) const;
  std::vector<unsigned int> ImageSize(SizeType type=Original) const { return(_scans[0]->ImageSize(type)); }
  std::vector<double>       ImageVxs(SizeType type=Original) const { return(_scans[0]->ImageVxs(type)); }
  unsigned int SubSampling() const { return(_ss); }
  double FWHM() const { return(_fwhm); }
  TopupInterpolationType InterpolationModel() const { return(_it); }
  bool HasBeta() const { return(_hasx); }
  bool HasGamma() const { return(_hasy); }
  bool HasBeta(unsigned int i) const;
  bool HasGamma(unsigned int i) const;

  bool TracePrint() const { return(_tp); }
  void SetTracePrint(bool val=true) { _tp=val; for (unsigned int i=0; i<_scans.size(); i++) _scans[i]->SetTracePrint(val); }

  // Routines that return data from the scans
  // Get average of all warped and modulated scans
  const NEWIMAGE::volume<float>& GetMean(const BASISFIELD::splinefield& field) const { update(field); return(_mean); }
  // Get area where data is present for all scans
  const NEWIMAGE::volume<char>& GetMask(const BASISFIELD::splinefield& field) const { update(field); return(_mask); }
  // See paper/techreport for explanation of alpha, beta and gamma
  const NEWIMAGE::volume<float>& GetMeanAlpha(const BASISFIELD::splinefield& field) const { update(field); return(_mean_alpha); }
  const NEWIMAGE::volume<float>& GetMeanBeta(const BASISFIELD::splinefield& field) const { update(field); return(_mean_beta); }
  const NEWIMAGE::volume<float>& GetMeanGamma(const BASISFIELD::splinefield& field) const { update(field); return(_mean_gamma); }
  NEWIMAGE::volume<float> GetScan(unsigned int                   i,
                                  const BASISFIELD::splinefield& field,
				  bool                           masked=false) const;
  NEWIMAGE::volume<float> GetAlpha(unsigned int i,
                                   const BASISFIELD::splinefield& field) const;
  NEWIMAGE::volume<float> GetBeta(unsigned int i,
                                  const BASISFIELD::splinefield& field) const;
  NEWIMAGE::volume<float> GetGamma(unsigned int i,
                                   const BASISFIELD::splinefield& field) const;
  NEWIMAGE::volume<float> GetMovementDerivative(unsigned int scan,
                                                unsigned int deriv,
                                                const BASISFIELD::splinefield& field) const;
  NEWIMAGE::volume<float> GetNumericalMovementDerivative(unsigned int scan,
                                                         unsigned int deriv,
                                                         const BASISFIELD::splinefield& field) const;
  NEWIMAGE::volume<float> GetJacobian(unsigned int i,
                                      const BASISFIELD::splinefield& field) const;

  NEWIMAGE::volume4D<float> GetDisplacementField(unsigned int                    scanindx,
						 const BASISFIELD::splinefield&  field) const
  {
    if (scanindx >= _scans.size()) throw TopupException("TopupScanManager::GetDisplacementField: index scanindx out of range");
    return(_scans[scanindx]->GetDisplacementField(field));
  }


  // Routines that set/change the state of the scans
  void FieldUpdated() const;
  void ReGrid(unsigned int xsz, unsigned int ysz, unsigned int zsz);
  void ReGrid(const std::vector<unsigned int>& sz) { ReGrid(sz[0],sz[1],sz[2]); }
  void SubSample(unsigned int ss);
  void Smooth(double fwhm);
  void SetMovementParameters(const NEWMAT::ColumnVector& mp) const; // Sets the parameters that are being estimated
  void SetInterpolationModel(TopupInterpolationType it) const;

//private:
  std::vector<TopupScan* >                      _scans;  // Vector of pointers
  mutable NEWIMAGE::volume<char>                _mask;
  mutable NEWIMAGE::volume<float>               _mean;
  mutable NEWIMAGE::volume<float>               _mean_alpha;
  mutable NEWIMAGE::volume<float>               _mean_beta;
  mutable NEWIMAGE::volume<float>               _mean_gamma;
  mutable bool                                  _up_to_date;
  std::vector<unsigned int>                     _regrid_sz;
  unsigned int                                  _ss;
  mutable TopupInterpolationType                _it;
  double                                        _fwhm;
  std::vector<std::vector<unsigned int> >       _mpi;    // Indicies to movement parameters
  bool                                          _hasx;
  bool                                          _hasy;
  bool                                          _tp;     // Trace-print

  void update(const BASISFIELD::splinefield& field) const;
  bool have_different_signs(double f1, double f2) const { return((f1>0 && f2<0) || (f1<0 && f2>0)); }
  NEWIMAGE::volume<float> char_to_float(const NEWIMAGE::volume<char>& invol) const;
};

// }}} End of fold.

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//
// Class TopupCF
//
// The "main" class of the Topup application. Passed to the nonlin
// library.
//
// {{{ @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


class TopupCF: public MISCMATHS::NonlinCF
{
public:
  TopupCF(const NEWIMAGE::volume4D<float>&     scans,
          const NEWMAT::Matrix&                pevecs,
          const NEWMAT::ColumnVector&          rotimes,
          double                               warpres,
          unsigned int                         sporder=3);

  // Routines to get information about the run
  // These will (probably) mostly be used internally

  unsigned int NPar() const { if (_mf) return(_field.CoefSz()); else return(_field.CoefSz()+NMovPar()); }
  NEWMAT::ReturnMatrix Par() const;
  unsigned int NDefPar() const { return(_field.CoefSz()); }
  unsigned int NMovPar() const { if (!_mf) return(_sm.NoOfMovementParameters()); else return(0); }
  double WarpResolution() const { return(_wr); }
  MISCMATHS::BFMatrixPrecisionType HessianPrecision() const { return(_hp); }
  bool TracePrint() const { return(_tp); }
  bool Verbose() const { return(_verb); }
  double Lambda() const { if (_ssql) return(_lssd * _lambda); else return(_lambda); }
  bool MovementsFixed() const { return(_mf); }
  TopupInterpolationType InterpolationModel() const { return(_sm.InterpolationModel()); }

  // Routines used to set parameters that determine how topup is run.
  // Most of these correspond to user settable parameters on the command line.

  void SetTracePrint(bool val=true) { _tp=val; _sm.SetTracePrint(val); }
  void SetRegridding(const std::vector<unsigned int>& ims);
  void Smooth(double fwhm) { if (fwhm != _sm.FWHM()) _sm.Smooth(fwhm); }
  void SubSample(unsigned int ss);
  void SetWarpResolution(double wr); // N.B. wr should be in mm
  void SetVerbose(bool val=true) { _verb=val; }
  void SetMovementsFixed(bool val=true) { _mf=val; }
  void SetRegularisation(double lambda, RegularisationType type=BendingEnergy) { _lambda=lambda; _rt=type; }
  void SetSSQLambda(bool val=true) { _ssql=val; }
  void SetHessianPrecision(MISCMATHS::BFMatrixPrecisionType hp) { _hp = hp; }
  void SetInterpolationModel(TopupInterpolationType it) { _sm.SetInterpolationModel(it); }

  // Routines for writing results/information to file.

  void WriteCoefficients(const std::string& fname) const;
  void WriteUnwarped(const std::string& fname) const;
  void WriteUnwarped(const std::string&               fname,
		     const NEWIMAGE::volume4D<float>& hdr,
		     double                           sf) const;
  void WriteJacobiansForDebug(const std::string& fname) const;
  void WriteField(const std::string& fname) const;
  void WriteField(const std::string& fname, const NEWIMAGE::volume4D<float>& hdr) const;
  void WriteDisplacementFields(const std::string& fname) const;
  void WriteMask(const std::string& fname) const;
  void WriteMaskedDiff(const std::string& fname) const;
  void WriteMovementParameters(const std::string& fname) const;
  void WriteRigidBodyMatrices(const std::string& fname) const;
  void WriteJacobians(const std::string& fname) const;

  // Routines for writing debug output to disc.

  void SetDebug(unsigned int level=1) { if (level>3) throw TopupException("TopupCF::SetDebug: Debug level out of range"); else _dl=level; }
  void SetLevel(unsigned int level) const { _level=level; _iter=0; _attempt=0; }

  // Returns relevent field energy or gradient/Hessian thereof.

  double FieldEnergy() const;
  NEWMAT::ReturnMatrix FieldEnergyGrad() const;
  std::shared_ptr<MISCMATHS::BFMatrix> FieldEnergyHess() const;

  // cost-function, gradient and Hessian as specified in NonlinCF class

  virtual double cf(const NEWMAT::ColumnVector& p) const;
  virtual NEWMAT::ReturnMatrix grad(const NEWMAT::ColumnVector& p) const;
  virtual std::shared_ptr<MISCMATHS::BFMatrix> hess(const NEWMAT::ColumnVector& p,
                                                    std::shared_ptr<MISCMATHS::BFMatrix>  iptr=std::shared_ptr<MISCMATHS::BFMatrix>()) const;

private:
  TopupScanManager                  _sm;      // All the scans
  mutable BASISFIELD::splinefield   _field;   // field
  double                            _wr;      // Requested warp resolution (knot-spacing) in mm
  mutable double                    _lambda;  // Regularisation weight
  bool                              _ssql;    // Should regularisation be weighted with SSD?
  RegularisationType                _rt;      // Regularisation type
  bool                              _mf;      // Movements kept fixed if set.
  MISCMATHS::BFMatrixPrecisionType  _hp;      // Precision of Hessian Matrix
  bool                              _verb;    // Verbose
  bool                              _tp;      // Trace-print
  unsigned int                      _dl;      // Debug level
  mutable double                    _lssd;    // Latest sum-of-squared differences
  mutable unsigned int              _level;   // Unwarping level
  mutable unsigned int              _iter;    // Unwarping iteration (within level)
  mutable unsigned int              _attempt; // Unwarping attempt (within iter)

  void set_latest_ssd(double ssd) const { _lssd=ssd; }
  void set_field_params(const NEWMAT::ColumnVector& p) const;
  void set_movement_params(const NEWMAT::ColumnVector& p) const;
  double sum_of_prod(const NEWIMAGE::volume<float>& i1,
                     const NEWIMAGE::volume<float>& i2,
                     const NEWIMAGE::volume<char>&  mask) const;
  NEWMAT::ReturnMatrix movement_hessian(unsigned int s1,
                                        unsigned int s2,
                                        const NEWIMAGE::volume<char>&   mask,
                                        const BASISFIELD::splinefield&  field) const;
  BASISFIELD::splinefield field_factory(const NEWIMAGE::volume4D<float>& scans,
                                        double                           warpres,
                                        unsigned int                     sporder) const;

  // Routines for implementing debug write-outs.
  unsigned int debug_level() const { return(_dl); }
  void set_iter(unsigned int iter=0) const { _iter=iter; _attempt=0; }
  void set_attempt(unsigned int attempt) const { _attempt=attempt; }
  unsigned int level() const { return(_level); }
  unsigned int iter() const { return(_iter); }
  unsigned int attempt() const { return(_attempt); }
  std::string debug_string() const {char c_str[256]; sprintf(c_str,"level%02d_iter%02d_attempt%02d",level(),iter(),attempt()); return(std::string(c_str));}

  // Routines for numerical calculation of parts of
  // the gradient or the hessian
  NEWMAT::ColumnVector numerical_gradient(const NEWMAT::ColumnVector& p, unsigned int fi,
					  unsigned int li, double delta=-1, bool reg=false) const;
  NEWMAT::Matrix numerical_hessian(const NEWMAT::ColumnVector& p, unsigned int fi1, unsigned int li1, unsigned int fi2,
				   unsigned int li2, double delta1=-1, double delta2=-1, bool reg=false) const;
  NEWMAT::Matrix numerical_hessian(const NEWMAT::ColumnVector& p, unsigned int fi, unsigned int li,
				   double delta=-1, bool reg=false) const {
    return(numerical_hessian(p,fi,li,fi,li,delta,delta,reg));
  }
};

// }}} End of fold.

} // End namespace TOPUP

#endif // End #ifndef topup_costfuctions_h
