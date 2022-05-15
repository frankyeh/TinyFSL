// Utility for resampling of DTI data using fields/data
// estimated by topup.
//
// applytopup.h
//
// Jesper Andersson, FMRIB Image Analysis Group
//
//Copyright (C) 2012 University of Oxford

/*  CCOPYRIGHT  */

#ifndef applytopup_h
#define applytopup_h

#include <cstring>
#include "armawrap/newmat.h"
#include "miscmaths/miscmaths.h"
#include "newimage/newimageall.h"
#include "warpfns/fnirt_file_reader.h"
#include "topup_file_io.h"

namespace TOPUP {

class ApplyTopupException: public std::exception
{
private:
  std::string m_msg;
public:
  ApplyTopupException(const std::string& msg) throw(): m_msg(msg) { std::cout << what() << std::endl; }

  virtual const char * what() const throw() {
    return std::string("ApplyTopup:: msg=" + m_msg).c_str();
  }

  ~ApplyTopupException() throw() {}
};

// Helper-class for divding a set of scans into collections that can be used for least squares restoration

class TopupCollections
{
public:
  TopupCollections(const std::vector<unsigned int>& indx,
                   const TopupDatafileReader&       dfile);
  ~TopupCollections() {}
  unsigned int NCollections() { return(indxs.size()); }
  unsigned int NIndices(unsigned int c) { if (c>=NCollections()) throw ApplyTopupException("TopupCollections::NIndicies: Index out of range"); else return(indxs[c].size()); }
  unsigned int NScans(unsigned int c) { return(NIndices(c)); }
  unsigned int IndexAt(unsigned int c, unsigned int i);
  unsigned int ScanAt(unsigned int c, unsigned int i);
private:
  std::vector<std::vector<unsigned int> >  indxs;
  std::vector<std::vector<unsigned int> >  scans;
};

// Global functions used by applytopup

// Does the job
int applytopup();
// Helper functions
NEWIMAGE::volume4D<float> vb_resample_4D(const TopupDatafileReader&                       datafile,
                                         const TopupFileReader&                           topupfile,
                                         const std::vector<unsigned int>&                 inindices,
                                         NEWIMAGE::volume<char>&                          mask,
                                         std::vector<NEWIMAGE::volume4D<float> >&         scans);
NEWIMAGE::volume4D<float> vb_resample_3D(const TopupDatafileReader&                       datafile,
                                         const TopupFileReader&                           topupfile,
                                         const std::vector<unsigned int>&                 inindices,
                                         NEWIMAGE::volume<char>&                          mask,
                                         std::vector<NEWIMAGE::volume4D<float> >&         scans);
double tau_update_contrib(const std::vector<std::vector<NEWMAT::ColumnVector> >&        mu,
                          const std::vector<std::vector<MISCMATHS::SpMat<double> > >&   Lambda,
		          const std::vector<std::vector<MISCMATHS::SpMat<double> > >&   Lijk);
double tau_update(const std::vector<std::vector<std::vector<NEWMAT::ColumnVector> > >&  mu,
                  const std::vector<std::vector<MISCMATHS::SpMat<double> > >&           Lambda,
		  const std::vector<std::vector<MISCMATHS::SpMat<double> > >&           Lijk,
                  double                                                                tau_0);
double tau_update(const std::vector<std::vector<NEWMAT::ColumnVector> >&        mu,
                  const std::vector<std::vector<MISCMATHS::SpMat<double> > >&   Lambda,
		  const std::vector<std::vector<MISCMATHS::SpMat<double> > >&   Lijk,
                  double                                                        tau_0);
NEWMAT::ColumnVector mu_update(const std::vector<std::vector<NEWMAT::ColumnVector> >&        mu,
                               const MISCMATHS::SpMat<double>&                               K,
		               const NEWMAT::ColumnVector&                                   y,
                               const NEWMAT::CroutMatrix&                                    Lambda,
                               const std::vector<std::vector<MISCMATHS::SpMat<double> > >&   Lijk,
                               double                                                        phi,
                               double                                                        lambda,
                               int                                                           sl,
                               int                                                           row);
NEWIMAGE::volume4D<float> vb_resample_2D(const TopupDatafileReader&                       datafile,
                                         const TopupFileReader&                           topupfile,
                                         const std::vector<unsigned int>&                 inindices,
                                         NEWIMAGE::volume<char>&                          mask,
                                         std::vector<NEWIMAGE::volume4D<float> >&         scans);
std::vector<std::vector<MISCMATHS::SpMat<double> > > GetLijk(int                        sz,
                                                             const std::vector<double>  sf);
std::vector<MISCMATHS::SpMat<double> > GetLij(int                                       sz,
                                   double                                               sf);
MISCMATHS::SpMat<double> GetLii(int                                                     sz);
NEWIMAGE::volume4D<float> jac_resample(const TopupDatafileReader&                       datafile,
                                       const TopupFileReader&                           topupfile,
                                       const std::vector<unsigned int>&                 inindices,
                                       NEWIMAGE::volume<char>&                          mask,
                                       std::vector<NEWIMAGE::volume4D<float> >&         scans,
                                       NEWIMAGE::interpolation                          interp);
NEWIMAGE::volume4D<float> lsr_resample(const TopupDatafileReader&                       datafile,
                                       const TopupFileReader&                           topupfile,
                                       const std::vector<unsigned int>&                 inindices,
                                       NEWIMAGE::volume<char>&                          mask,
                                       std::vector<NEWIMAGE::volume4D<float> >&         scans);
void resample_using_movement_parameters(const TopupDatafileReader&                      datafile,
                                        const TopupFileReader&                          topupfile,
                                        const std::vector<unsigned int>&                inindices,
                                        NEWIMAGE::volume<char>&                         mask,
                                        std::vector<NEWIMAGE::volume4D<float> >&        scans,
                                        NEWIMAGE::interpolation                         interp);
void volume_deffield_2_jacobian(const NEWIMAGE::volume<float>& pxd, // x-displacements, in mm
				const NEWIMAGE::volume<float>& pyd, // y-displacements, in mm
				const NEWIMAGE::volume<float>& pzd, // z-displacements, in mm
				NEWIMAGE::volume<float>&       jac);// Map of Jacobian determinants
std::vector<unsigned int> parse_commaseparated_numbers(const std::string& list);
std::vector<std::string> parse_commaseparated_list(const std::string&  list);
bool row_col_is_alright(const NEWIMAGE::volume<char>&   mask,
                        unsigned int                    k,
                        unsigned int                    ij,
                        bool                            row);
NEWMAT::ReturnMatrix extract_row_col(const NEWIMAGE::volume<float>&  vol,
                                     unsigned int                    k,
                                     unsigned int                    ij,
                                     bool                            row);
void add_to_slice(NEWIMAGE::volume<float>&                  vol,
                  const std::vector<NEWMAT::ColumnVector>&  mu,
                  unsigned int                              sl,
                  bool                                      row);
void add_to_rows_cols(NEWIMAGE::volume4D<float>&  vol,
                      const NEWMAT::Matrix&       B,
                      unsigned int                k,
                      unsigned int                ij,
                      bool                        row);
void zero_out_rows_cols(NEWIMAGE::volume4D<float>&   vol,
                        const NEWMAT::Matrix&        map,
                        bool                         row);


}      // End namespace TOPUP

#endif // End ifndef applytopup_h
