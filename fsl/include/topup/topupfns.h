// Declarations of utility functions/classes used by topup
//
// topupfns.h
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2012 University of Oxford

/*  CCOPYRIGHT  */

#ifndef topupfns_h
#define topupfns_h

#include <string>
#include <vector>
#include <memory>
#include "armawrap/newmat.h"
#include "newimage/newimage.h"
#include "basisfield/basisfield.h"
#include "basisfield/splinefield.h"
#include "basisfield/dctfield.h"
#include "utils/options.h"
#include "miscmaths/nonlin.h"
#include "topup_costfunctions.h"
#include "topup_file_io.h"

namespace TOPUP {

std::string path(const std::string& fullname);
std::string filename(const std::string& fullname);
std::string extension(const std::string& fullname);

enum PrecisionType         {FloatPrecision, DoublePrecision};

///////////////////////////////////////////////////////////////////////////////////////////////
//
// topup_clp is a glorified struct that holds the command line parameters of topup
//
///////////////////////////////////////////////////////////////////////////////////////////////

class topup_clp
{
public:
  topup_clp(const Utilities::Option<std::string>&     imain,
	    const Utilities::Option<std::string>&         datain,
	    const Utilities::Option<std::string>&         cout,
	    const Utilities::Option<std::string>&         fout,
	    const Utilities::Option<std::string>&         iout,
	    const Utilities::Option<std::string>&         dfout,
	    const Utilities::Option<std::string>&         rbmout,
	    const Utilities::Option<std::string>&         jacout,
	    const Utilities::Option<std::string>&         lout,
	    const Utilities::Option<std::vector<float> >& warpres,
	    const Utilities::Option<std::vector<int> >&   subsamp,
	    const Utilities::Option<std::vector<float> >& fwhm,
	    const Utilities::Option<std::vector<int> >&   miter,
	    const Utilities::Option<std::vector<float> >& lambda,
	    const Utilities::Option<int>&                 ssqlambda,
	    const Utilities::Option<std::vector<int> >&   estmov,
	    const Utilities::Option<std::vector<int> >&   minmet,
	    const Utilities::Option<std::string>&         regmod,
	    const Utilities::Option<int>&                 sporder,
	    const Utilities::Option<std::string>&         numprec,
	    const Utilities::Option<std::string>&         interp,
	    const Utilities::Option<int>&                 indscale,
	    const Utilities::Option<int>&                 regrid,
	    const Utilities::Option<bool>&                verbose,
	    const Utilities::Option<int>&                 debug,
	    const Utilities::Option<bool>&                trace);
  const std::string& ImaFname()  const { return(_imain); }
  std::string CoefFname() const {
    if (!_out.length()) _out = TOPUP::path(_imain) + TOPUP::filename(_imain);
    return(_out + std::string("_fieldcoef"));
  }
  std::string MovParFname() const {
    if (!_out.length()) _out = TOPUP::path(_imain) + TOPUP::filename(_imain);
    return(_out + std::string("_movpar.txt"));
  }
  const std::string& LogFname() const {
    if (!_lout.length()) _lout = TOPUP::path(_imain) + TOPUP::filename(_imain) + std::string(".topup_log");
    return(_lout);
  }
  const std::string& FieldFname() const { return(_fout); }
  const std::string& ImaOutFname() const { return(_iout); }
  const std::string& DisplacementFieldBaseFname() const { return(_dfout); }
  const std::string& RigidBodyBaseFname() const { return(_rbmout); }
  const std::string& JacobianBaseFname() const { return(_jacout); }
  const NEWMAT::Matrix PhaseEncodeVectors() const { return(_datafile.PhaseEncodeVectors()); }
  const NEWMAT::ColumnVector ReadoutTimes() const { return(_datafile.ReadOutTimes()); }
  unsigned int NoOfLevels() const { return(_nlev); }
  TopupInterpolationType InterpolationModel() const { return(_interp); }
  MISCMATHS::BFMatrixPrecisionType HessianPrecision() const { return(_numprec); }
  RegularisationType RegularisationModel() const { return(_regtype); }
  unsigned int SplineOrder() const { return(_sporder); }
  unsigned int DebugLevel() const { return(_debug); }
  bool Verbose() const { return(_verbose); }
  bool Trace() const { return(_trace); }
  bool SSQLambda() const { return(_ssqlambda); }
  bool IndividualScaling() const { return(_indscale); }
  std::vector<unsigned int> Regridding(const NEWIMAGE::volume4D<float>& invols) const;
  unsigned int SubSampling(unsigned int level) const {
    if (level < 1 || level > _nlev) throw TopupException("topup_clp::SubSampling: Out-of-range value of level");
    return(static_cast<unsigned int>(_subsamp[level-1]));
  }
  double WarpRes(unsigned int level) const {
    if (level < 1 || level > _nlev) throw TopupException("topup_clp::WarpRes: Out-of-range value of level");
    return(_warpres[level-1]);
  }
  double FWHM(unsigned int level) const {
    if (level < 1 || level > _nlev) throw TopupException("topup_clp::FWHM: Out-of-range value of level");
    return(_fwhm[level-1]);
  }
  double Lambda(unsigned int level) const {
    if (level < 1 || level > _nlev) throw TopupException("topup_clp::Lambda: Out-of-range value of level");
    return(_lambda[level-1]);
  }
  unsigned int MaxIter(unsigned int level) const {
    if (level < 1 || level > _nlev) throw TopupException("topup_clp::MaxIter: Out-of-range value of level");
    return(static_cast<unsigned int>(_miter[level-1]));
  }
  bool EstimateMovements(unsigned int level) const {
    if (level < 1 || level > _nlev) throw TopupException("topup_clp::EstimateMovements: Out-of-range value of level");
    return(static_cast<bool>(_estmov[level-1] != 0));
  }
  MISCMATHS::NLMethod OptimisationMethod(unsigned int level) const {
    if (level < 1 || level > _nlev) throw TopupException("topup_clp::OptimisationMethod: Out-of-range value of level");
    return(_optmet[level-1]);
  }

private:
  unsigned int                           _nlev;
  std::string                            _imain;
  mutable std::string                    _out;
  std::string                            _fout;
  std::string                            _iout;
  std::string                            _dfout;
  std::string                            _rbmout;
  std::string                            _jacout;
  mutable std::string                    _lout;
  TopupDatafileReader                    _datafile;
  std::vector<float>                     _warpres;
  std::vector<int>                       _subsamp;
  std::vector<float>                     _fwhm;
  std::vector<int>                       _miter;
  std::vector<float>                     _lambda;
  bool                                   _ssqlambda;
  bool                                   _indscale;
  bool                                   _regrid;
  RegularisationType                     _regtype;
  TopupInterpolationType                 _interp;
  MISCMATHS::BFMatrixPrecisionType       _numprec;
  std::vector<MISCMATHS::NLMethod>       _optmet;
  std::vector<int>                       _estmov;
  unsigned int                           _sporder;
  bool                                   _verbose;
  bool                                   _trace;
  unsigned int                           _debug;
};

///////////////////////////////////////////////////////////////////////////////////////////////
//
// Declarations of global functions used by topup
//
///////////////////////////////////////////////////////////////////////////////////////////////

std::shared_ptr<topup_clp> parse_topup_command_line(unsigned int   narg,
                                                    char           *args[]);
bool check_exist(const std::string& fname);
std::string existing_conf_file(const std::string& cfname);

} // End namespace TOPUP

#endif // End #ifndef topupfns_h
