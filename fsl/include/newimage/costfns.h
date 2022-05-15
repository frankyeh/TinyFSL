/*  costfns.h

    Mark Jenkinson, FMRIB Image Analysis Group

    Copyright (C) 2001 University of Oxford  */

/*  CCOPYRIGHT  */


#if !defined(__costfns_h)
#define __costfns_h

#include "newimageall.h"

namespace NEWIMAGE {

  enum costfns { Woods, CorrRatio, MutualInfo, NormCorr, NormMI, LeastSq, LabelDiff,
		 NormCorrSinc, BBR, Unknown };

  costfns costfn_type(const std::string& cname);

  class Costfn {
  public:
    const volume<float> &refvol;
    const volume<float> &testvol;
    const volume<float> &rweight;
    const volume<float> &tweight;
    volume<float> wmseg;  // WM segmentation (or any useful boundary) for BBR
    volume<float> fmap;  // fieldmap for BBR
    volume<float> fmap_mask;  // fieldmap mask for BBR
    //volume4D<float> nonlin_basis;
    mutable volume4D<float> debugvol;
    NEWMAT::ColumnVector testCog;
  private:
    int *bindex;
    int no_bins;
    NEWMAT::ColumnVector plnp;
    int *jointhist;
    int *marghist1;
    int *marghist2;
    float *fjointhist;
    float *fmarghist1;
    float *fmarghist2;
    mutable int p_count;
    costfns p_costtype;
    bool validweights;
    float bin_a0;
    float bin_a1;
    float bbr_dist;  // in mm
    float bbr_offset;
    float bbr_slope;
    NEWMAT::Matrix bbr_pts;  // in mm coords
    NEWMAT::Matrix bbr_norms;  // in mm coords pointing from wm to other
    float *gm_coord_x;  // in mm coords
    float *gm_coord_y;
    float *gm_coord_z;
    float *wm_coord_x;
    float *wm_coord_y;
    float *wm_coord_z;
    int no_coords;
    int vertex_step;
    int pe_dir;  // for fieldmap application
    std::string bbr_type;
    bool debug_mode;
  public:
    float smoothsize;
    float fuzzyfrac;

  public:
    // Publicly available calls
    Costfn(const volume<float>& refv, const volume<float>& inputv);
    Costfn(const volume<float>& refv, const volume<float>& inputv,
	   const volume<float>& refweight, const volume<float>& inweight);
    ~Costfn();

    void set_debug_mode(bool debug_flag=true);
    void set_costfn(const costfns& costtype) { p_costtype = costtype; }
    costfns get_costfn(void) { return p_costtype; }
    void set_no_bins(int n_bins);
    int set_bbr_seg(const volume<float>& bbr_seg);
    int set_bbr_coords(const NEWMAT::Matrix& coords, const NEWMAT::Matrix& norms);
    int set_bbr_type(const std::string& typenm);
    int set_bbr_step(int step);
    int set_bbr_slope(float slope);
    int set_bbr_fmap(const volume<float>& fieldmap, int phase_encode_direction);
    int set_bbr_fmap(const volume<float>& fieldmap, const volume<float>& fieldmap_mask, int phase_encode_direction);
    int count() const { return p_count; }

    // General cost function call
    float cost(const NEWMAT::Matrix& affmat) const;    // affmat is voxel to voxel
    // affmat is voxel to voxel and non-linear parameters are arbitrary
    float cost(const NEWMAT::Matrix& affmat, const NEWMAT::ColumnVector& nonlin_params) const;
    // in the following, all warps are mm to mm
    float cost(const volume4D<float>& warp) const;
    float cost_gradient(volume4D<float>& gradvec,
			const volume4D<float>& warp, bool nullbc) const;

    // some basic entropy calls
    float ref_entropy(const NEWMAT::Matrix& aff) const;
    float test_entropy(const NEWMAT::Matrix& aff) const;
    float joint_entropy(const NEWMAT::Matrix& aff) const;

    volume<float> image_mapper(const NEWMAT::Matrix& affmat) const;    // affmat is voxel to voxel
    NEWMAT::Matrix mappingfn(const NEWMAT::Matrix& affmat) const;    // affmat is voxel to voxel
    float get_bin_intensity(int bin_number) const;
    float get_bin_number(float intensity) const;
    bool is_bbr_set(void) const;

    // a resampling function (since it is logical to keep it with the general cost processing for bbr)
    float bbr_resamp(const NEWMAT::Matrix& aff, const NEWMAT::ColumnVector& nonlin_params, volume<float>& resampvol) const;

  private:
    // Prevent default behaviours
    Costfn();
    Costfn operator=(const Costfn&);
    Costfn(const Costfn&);

    // Internal functions available
    float normcorr(const NEWMAT::Matrix& aff) const;
    float normcorr_smoothed(const NEWMAT::Matrix& aff) const;
    float normcorr_smoothed_sinc(const NEWMAT::Matrix& aff) const;
    float normcorr_fully_weighted(const NEWMAT::Matrix& aff,
				  const volume<float>& refweight,
				  const volume<float>& testweight) const;

    float leastsquares(const NEWMAT::Matrix& aff) const;
    float leastsquares_smoothed(const NEWMAT::Matrix& aff) const;
    float leastsquares_fully_weighted(const NEWMAT::Matrix& aff,
				      const volume<float>& refweight,
				      const volume<float>& testweight) const;

    float labeldiff(const NEWMAT::Matrix& aff) const;
    float labeldiff_smoothed(const NEWMAT::Matrix& aff) const;
    float labeldiff_fully_weighted(const NEWMAT::Matrix& aff,
				      const volume<float>& refweight,
				      const volume<float>& testweight) const;

    float bbr(const NEWMAT::Matrix& aff) const;
    float bbr(const NEWMAT::Matrix& aff, const NEWMAT::ColumnVector& nonlin_params) const;
    float bbr(const NEWMAT::Matrix& aff, const NEWMAT::ColumnVector& nonlin_params,
	      volume<float>& resampvol, bool resampling_required) const;
    float fmap_extrap(const double& x_vox, const double& y_vox, const double& z_vox, const NEWMAT::ColumnVector& v_pe) const;
    int vox_coord_calc(NEWMAT::ColumnVector& tvc, NEWMAT::ColumnVector& rvc, const NEWMAT::Matrix& aff, const NEWMAT::ColumnVector& nonlin_params,
		       const NEWMAT::Matrix& iaffbig, const NEWMAT::Matrix& mm2vox, const NEWMAT::ColumnVector& pe_dir_vec) const;

    float woods_fn(const NEWMAT::Matrix& aff) const;
    float woods_fn_smoothed(const NEWMAT::Matrix& aff) const;

    float corr_ratio(const NEWMAT::Matrix& aff) const;
    float corr_ratio_smoothed(const NEWMAT::Matrix& aff) const;
    float corr_ratio_fully_weighted(const NEWMAT::Matrix& aff,
				    const volume<float>& refweight,
				    const volume<float>& testweight) const;
    float corr_ratio_fully_weighted(const volume4D<float>& warpvol,
				    const volume<float>& refweight,
				    const volume<float>& testweight) const;
    float corr_ratio_gradient_fully_weighted(volume4D<float>& gradvec,
					     const volume4D<float>& warpvol,
					     const volume<float>& refweight,
					     const volume<float>& testweight,
					     bool nullbc) const;

    float mutual_info(const NEWMAT::Matrix& aff) const;
    float mutual_info_smoothed(const NEWMAT::Matrix& aff) const;
    float mutual_info_fully_weighted(const NEWMAT::Matrix& aff,
				     const volume<float>& refweight,
				     const volume<float>& testweight) const;

    float normalised_mutual_info(const NEWMAT::Matrix& aff) const;
    float normalised_mutual_info_smoothed(const NEWMAT::Matrix& aff) const;
    float normalised_mutual_info_fully_weighted(const NEWMAT::Matrix& aff,
						const volume<float>& refweight,
						const volume<float>& testweight) const;

    float cost(const NEWMAT::Matrix& affmat,
	       const volume<float>& refweight,
	       const volume<float>& testweight) const;

    float cost(const NEWMAT::Matrix& affmat,
	       const NEWMAT::ColumnVector& nonlin_params,
	       const volume<float>& refweight,
	       const volume<float>& testweight) const;

    float cost(const volume4D<float>& warp,
	       const volume<float>& refweight,
	       const volume<float>& testweight) const;

    float cost_gradient(volume4D<float>& gradvec,
			const volume4D<float>& warp,
			const volume<float>& refweight,
			const volume<float>& testweight,
			bool nullbc=false) const;

    int p_corr_ratio_image_mapper(volume<float>& vout,
				  NEWMAT::Matrix& mappingfn,
				  const volume<float>& vref,
				  const volume<float>& vtest,
				  const volume<float>& refweight,
				  const volume<float>& testweight,
				  int *bindex, const NEWMAT::Matrix& aff,
				  const int no_bins, const float smoothsize) const;
    };


   //////////////////////////////////////////////////////////////////////////


}

#endif
