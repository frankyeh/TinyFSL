//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Provide an interface matching BASISFIELD::splinefield, specifically for use with
///        EDDY::MoveBySuscCF class
/// \details Essentially this is a wrapper around a BASISFIELD::splinefield object, which
///          re-implements some of the more expensive calculations on the GPU, whilst simply
///          passing other calculations through to the underlying BASISFIELD::splinefield
///          object. As such, expect limited functionality as compared to the BASISFIELD
///          version - just enough to get EDDY::MoveBySuscCF running faster. Additionally,
///          we are using the pimpl idiom here.
/// \author Frederik Lange
/// \date February 2018
/// \copyright Copyright (C) 2018 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////

#include <vector>
#include <iostream>
#include <algorithm>
#include <functional>
#include <memory>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>

#pragma push
#pragma diag_suppress = code_is_unreachable // Supress warnings from armawrap
#include "newimage/newimageall.h"
#include "armawrap/newmat.h"
#include "miscmaths/bfmatrix.h"
#include "basisfield/basisfield.h"
#include "basisfield/splinefield.h"
#include "basisfield/fsl_splines.h"

#include "CBFMemory.h"
#include "CBFSplineField.cuh"
#include "CBFKernelHelpers.cuh"
#include "CBFSparseDiagonalMatrix.cuh"
#include "CBFKernels.cuh"
#pragma pop

#include "helper_cuda.h"



/// Cuda Basisfield
namespace CBF
{
////////////////////////////////////////////////////////////////////////////////
// PIMPL Class
////////////////////////////////////////////////////////////////////////////////
  /// Implementation class for CBFSplineField
  class CBFSplineField::Impl
  {
    public:
      // Methods to wrap
      /// Standard BASISFIELD::splinefield constructor
      Impl(const std::vector<unsigned int>& psz, const std::vector<double>& pvxs,
          const std::vector<unsigned int>& pksp, int porder=3)
        : spline_field_(psz,pvxs,pksp,porder)
        // WARNING this might be a bad idea - what if pksp is too small?
        , spline_x_(porder,pksp.at(0))
        , spline_y_(porder,pksp.at(1))
        , spline_z_(porder,pksp.at(2))
      {}

      unsigned int CoefSz_x() const
      {
        return spline_field_.CoefSz_x();
      }
      unsigned int CoefSz_y() const
      {
        return spline_field_.CoefSz_y();
      }
      unsigned int CoefSz_z() const
      {
        return spline_field_.CoefSz_z();
      }
      /// From base class BASISFIELD::basisfield
      void AsVolume(NEWIMAGE::volume<float>& vol, BASISFIELD::FieldIndex fi=BASISFIELD::FIELD)
      {
        spline_field_.AsVolume(vol,fi);
      }
      /// From base class BASISFIELD::basisfield
      void SetCoef(const NEWMAT::ColumnVector& pcoef)
      {
        spline_field_.SetCoef(pcoef);
      }
      /// Calculate Bending Energy for regularisation purposes
      double BendEnergy() const
      {
        return spline_field_.BendEnergy();
      }
      /// Calculate gradient of the Bending Energy
      NEWMAT::ReturnMatrix BendEnergyGrad() const
      {
        return spline_field_.BendEnergyGrad();
      }
      /// Calculate hessian of the Bending Energy
      std::shared_ptr<MISCMATHS::BFMatrix> BendEnergyHess(
          MISCMATHS::BFMatrixPrecisionType prec) const
      {
        return spline_field_.BendEnergyHess(prec);
      }
      /// Jte V1
      NEWMAT::ReturnMatrix Jte(const NEWIMAGE::volume<float>&  ima1,
                               const NEWIMAGE::volume<float>&  ima2,
                               const NEWIMAGE::volume<char>    *mask)
                               const
      {
        return spline_field_.Jte(ima1,ima2,mask);
      }
      /// Jte V2
      NEWMAT::ReturnMatrix Jte(const std::vector<unsigned int>&  deriv,
                               const NEWIMAGE::volume<float>&    ima1,
                               const NEWIMAGE::volume<float>&    ima2,
                               const NEWIMAGE::volume<char>      *mask)
                               const
      {
        return spline_field_.Jte(deriv,ima1,ima2,mask);
      }
      /// Jte V3
      NEWMAT::ReturnMatrix Jte(const NEWIMAGE::volume<float>&    ima,
                               const NEWIMAGE::volume<char>      *mask)
                               const
      {
        return spline_field_.Jte(ima,mask);
      }
      /// Jte V4
      NEWMAT::ReturnMatrix Jte(const std::vector<unsigned int>&  deriv,
                               const NEWIMAGE::volume<float>&    ima,
                               const NEWIMAGE::volume<char>      *mask)
                               const
      {
        return spline_field_.Jte(deriv,ima,mask);
      }
      /// JtJ V1
      std::shared_ptr<MISCMATHS::BFMatrix> JtJ(const NEWIMAGE::volume<float>& ima,
                                               const NEWIMAGE::volume<char> *mask,
                                               MISCMATHS::BFMatrixPrecisionType prec)
                                               const
      {
        return spline_field_.JtJ(ima,mask,prec);
      }
      /// JtJ V2
      std::shared_ptr<MISCMATHS::BFMatrix> JtJ(const NEWIMAGE::volume<float>& ima1,
                                               const NEWIMAGE::volume<float>& ima2,
                                               const NEWIMAGE::volume<char> *mask,
                                               MISCMATHS::BFMatrixPrecisionType prec)
                                               const
      {
        auto deriv = std::vector<unsigned int>(3,0);
        return JtJ(deriv,ima1,ima2,mask,prec);
      }
      /// JtJ V3
      std::shared_ptr<MISCMATHS::BFMatrix> JtJ(const std::vector<unsigned int>& deriv,
                                               const NEWIMAGE::volume<float>& ima,
                                               const NEWIMAGE::volume<char> *mask,
                                               MISCMATHS::BFMatrixPrecisionType prec)
                                               const
      {
        return spline_field_.JtJ(deriv,ima,mask,prec);
      }
      /// JtJ V4
      std::shared_ptr<MISCMATHS::BFMatrix> JtJ(const std::vector<unsigned int>& deriv,
                                               const NEWIMAGE::volume<float>& ima1,
                                               const NEWIMAGE::volume<float>& ima2,
                                               const NEWIMAGE::volume<char> *mask,
                                               MISCMATHS::BFMatrixPrecisionType prec)
                                               const
      {
        // Check dimensions match
        if (deriv.size() != 3){
          throw BASISFIELD::BasisfieldException("splinefield::JtJ: Wrong size derivative vector");
        }
        if (!samesize(ima1,ima2,3,true)){
          throw BASISFIELD::BasisfieldException("splinefield::JtJ: Image dimension mismatch");
        }
        if (mask && !samesize(ima1,*mask,3)){
          throw BASISFIELD::BasisfieldException("splinefield::JtJ: Mismatch between image and mask");
        }
        if (spline_field_.FieldSz_x() != static_cast<unsigned int>(ima1.xsize()) ||
            spline_field_.FieldSz_y() != static_cast<unsigned int>(ima1.ysize()) ||
            spline_field_.FieldSz_z() != static_cast<unsigned int>(ima1.zsize())){
          throw BASISFIELD::BasisfieldException("splinefield::JtJ: Mismatch between image and field");
        }

        // Calculate Hadamard product of images and mask
        auto product_image = std::vector<float>(spline_field_.FieldSz());
        hadamard(ima1,ima2,*mask,product_image.data());

        // Get 1D spline kernels
        auto spline_x = BASISFIELD::Spline1D<float>(
            spline_x_.Order(),
            spline_x_.KnotSpacing(),
            deriv.at(0));
        auto spline_x_vals = spline_as_vec(spline_x);
        auto spline_y = BASISFIELD::Spline1D<float>(
            spline_y_.Order(),
            spline_y_.KnotSpacing(),
            deriv.at(1));
        auto spline_y_vals = spline_as_vec(spline_y);
        auto spline_z = BASISFIELD::Spline1D<float>(
            spline_z_.Order(),
            spline_z_.KnotSpacing(),
            deriv.at(2));
        auto spline_z_vals = spline_as_vec(spline_z);

        // Image and coefficient sizes
        std::vector<int> image_size{
            static_cast<int>(spline_field_.FieldSz_x()),
            static_cast<int>(spline_field_.FieldSz_y()),
            static_cast<int>(spline_field_.FieldSz_z())};
        std::vector<int> coef_size{
            static_cast<int>(spline_field_.CoefSz_x()),
            static_cast<int>(spline_field_.CoefSz_y()),
            static_cast<int>(spline_field_.CoefSz_z())};
        std::vector<int> knot_spacing{
            static_cast<int>(spline_x_.KnotSpacing()),
            static_cast<int>(spline_y_.KnotSpacing()),
            static_cast<int>(spline_z_.KnotSpacing())};

        auto r_JtJ = make_symmetric_JtJ(
            product_image,
            image_size,
            spline_x_vals,
            spline_y_vals,
            spline_z_vals,
            coef_size,
            knot_spacing,
            prec);

        return r_JtJ;
      }

      /// JtJ V5
      std::shared_ptr<MISCMATHS::BFMatrix> JtJ(const std::vector<unsigned int>& deriv1,
                                               const NEWIMAGE::volume<float>& ima1,
                                               const std::vector<unsigned int>& deriv2,
                                               const NEWIMAGE::volume<float>& ima2,
                                               const NEWIMAGE::volume<char> *mask,
                                               MISCMATHS::BFMatrixPrecisionType prec)
                                               const
      {
        // Check dimensions match
        if (deriv1.size() != 3 || deriv2.size() != 3){
          throw BASISFIELD::BasisfieldException("splinefield::JtJ: Wrong size derivative vector");
        }
        if (!samesize(ima1,ima2,3,true)){
          throw BASISFIELD::BasisfieldException("splinefield::JtJ: Image dimension mismatch");
        }
        if (mask && !samesize(ima1,*mask,3)){
          throw BASISFIELD::BasisfieldException("splinefield::JtJ: Mismatch between image and mask");
        }
        if (spline_field_.FieldSz_x() != static_cast<unsigned int>(ima1.xsize()) ||
            spline_field_.FieldSz_y() != static_cast<unsigned int>(ima1.ysize()) ||
            spline_field_.FieldSz_z() != static_cast<unsigned int>(ima1.zsize())){
          throw BASISFIELD::BasisfieldException("splinefield::JtJ: Mismatch between image and field");
        }

        // Calculate Hadamard product of images and mask
        auto product_image = std::vector<float>(spline_field_.FieldSz());
        hadamard(ima1,ima2,*mask,product_image.data());

        // Get 1D spline kernels
        auto spline_x_1 = BASISFIELD::Spline1D<float>(
            spline_x_.Order(),
            spline_x_.KnotSpacing(),
            deriv1.at(0));
        auto spline_x_vals_1 = spline_as_vec(spline_x_1);
        auto spline_y_1 = BASISFIELD::Spline1D<float>(
            spline_y_.Order(),
            spline_y_.KnotSpacing(),
            deriv1.at(1));
        auto spline_y_vals_1 = spline_as_vec(spline_y_1);
        auto spline_z_1 = BASISFIELD::Spline1D<float>(
            spline_z_.Order(),
            spline_z_.KnotSpacing(),
            deriv1.at(2));
        auto spline_z_vals_1 = spline_as_vec(spline_z_1);
        auto spline_x_2 = BASISFIELD::Spline1D<float>(
            spline_x_.Order(),
            spline_x_.KnotSpacing(),
            deriv2.at(0));
        auto spline_x_vals_2 = spline_as_vec(spline_x_2);
        auto spline_y_2 = BASISFIELD::Spline1D<float>(
            spline_y_.Order(),
            spline_y_.KnotSpacing(),
            deriv2.at(1));
        auto spline_y_vals_2 = spline_as_vec(spline_y_2);
        auto spline_z_2 = BASISFIELD::Spline1D<float>(
            spline_z_.Order(),
            spline_z_.KnotSpacing(),
            deriv2.at(2));
        auto spline_z_vals_2 = spline_as_vec(spline_z_2);

        // Image and coefficient sizes
        std::vector<int> image_size{
            static_cast<int>(spline_field_.FieldSz_x()),
            static_cast<int>(spline_field_.FieldSz_y()),
            static_cast<int>(spline_field_.FieldSz_z())};
        std::vector<int> coef_size_1{
            static_cast<int>(spline_field_.CoefSz_x()),
            static_cast<int>(spline_field_.CoefSz_y()),
            static_cast<int>(spline_field_.CoefSz_z())};
        std::vector<int> coef_size_2{
            static_cast<int>(spline_field_.CoefSz_x()),
            static_cast<int>(spline_field_.CoefSz_y()),
            static_cast<int>(spline_field_.CoefSz_z())};
        std::vector<int> knot_spacing_1{
            static_cast<int>(spline_x_.KnotSpacing()),
            static_cast<int>(spline_y_.KnotSpacing()),
            static_cast<int>(spline_z_.KnotSpacing())};
        std::vector<int> knot_spacing_2{
            static_cast<int>(spline_x_.KnotSpacing()),
            static_cast<int>(spline_y_.KnotSpacing()),
            static_cast<int>(spline_z_.KnotSpacing())};

        auto r_JtJ = make_non_symmetric_JtJ(
            product_image,
            image_size,
            spline_x_vals_1,
            spline_y_vals_1,
            spline_z_vals_1,
            spline_x_vals_2,
            spline_y_vals_2,
            spline_z_vals_2,
            coef_size_1,
            coef_size_2,
            knot_spacing_1,
            knot_spacing_2,
            prec);

        return r_JtJ;

      }

      /// JtJ V6
      std::shared_ptr<MISCMATHS::BFMatrix> JtJ(const NEWIMAGE::volume<float>& ima1,
                                               const BASISFIELD::basisfield& bf2,
                                               const NEWIMAGE::volume<float>& ima2,
                                               const NEWIMAGE::volume<char> *mask,
                                               MISCMATHS::BFMatrixPrecisionType prec)
                                               const
      {
        return spline_field_.JtJ(ima1,bf2,ima2,mask,prec);
      }

    private:
      /// Wrap texture object creation to conform to RAII and ensure the texture is
      /// always deleted
      class TextureHandle
      {
        public:
          TextureHandle(
              thrust::device_vector<float>& ima,
              const std::vector<int>& ima_sz)
            : texture_(0)
          {
            // Create resource descriptor
            cudaResourceDesc res_desc;
            memset(&res_desc,0,sizeof(res_desc));
            res_desc.resType = cudaResourceTypeLinear;
            res_desc.res.linear.devPtr = thrust::raw_pointer_cast(ima.data());
            res_desc.res.linear.desc.f = cudaChannelFormatKindFloat;
            res_desc.res.linear.desc.x = 32;
            res_desc.res.linear.sizeInBytes = ima_sz[0]*ima_sz[1]*ima_sz[2]*sizeof(float);
            // Create texture descriptor
            cudaTextureDesc tex_desc;
            memset(&tex_desc,0,sizeof(tex_desc));
            tex_desc.readMode = cudaReadModeElementType;
            // Create texture object based on channel and resource descriptors
            cudaCreateTextureObject(&texture_,&res_desc,&tex_desc,NULL);
          }
          ~TextureHandle()
          {
            // Release memory
            cudaDestroyTextureObject(texture_);
          }
          cudaTextureObject_t get_texture()
          {
            return texture_;
          }
        private:
          cudaTextureObject_t texture_;
      };

      // Stolen directly from BASISFIELD::splinefield
      inline void hadamard(
          const NEWIMAGE::volume<float>& ima1,
          const NEWIMAGE::volume<float>& ima2,
          const NEWIMAGE::volume<char>& mask,
          float *prod) const
      {
        // Check dimensions match
        if (!NEWIMAGE::samesize(ima1,ima2,3,true) || !NEWIMAGE::samesize(ima1,mask,3)){
          throw BASISFIELD::BasisfieldException("hadamard: Image dimension mismatch");
        }
        NEWIMAGE::volume<char>::fast_const_iterator itm = mask.fbegin();
        for (NEWIMAGE::volume<float>::fast_const_iterator it1=ima1.fbegin(),
            it2=ima2.fbegin(), it1_end=ima1.fend()
            ; it1 != it1_end
            ; ++it1, ++it2, ++itm, ++prod)
        {
          *prod = static_cast<float>(*itm) * (*it1) * (*it2);
        }
      }

      // Convert 1D spline to vector of floats
      std::vector<float> spline_as_vec(
          const BASISFIELD::Spline1D<float>& spline) const
      {
        auto spline_vals = std::vector<float>(spline.KernelSize());
        for (auto i = 0; i < spline.KernelSize(); ++i)
        {
          spline_vals.at(i) = spline(static_cast<unsigned int>(i+1));
        }
        return spline_vals;
      }

      // Create an empty JtJ matrix with the correct sparsity pattern
      CBF::SparseDiagonalMatrix create_empty_JtJ(
          const std::vector<int>& coef_sz) const
      {
        std::vector<int> offsets(343);
        for (unsigned int i = 0; i < 343; ++i)
        {
            unsigned int first_row, first_col, last_row, last_col;
            CBF::identify_diagonal(i,7,7,7,coef_sz[0],coef_sz[1],coef_sz[2],
                                       &first_row,&last_row,&first_col,&last_col);
            if (first_col == 0){
              offsets.at(i) = -static_cast<int>(first_row);
            }
            else{
              offsets.at(i) = static_cast<int>(first_col);
            }
        }
        unsigned int max_diagonal = coef_sz[0]*coef_sz[1]*coef_sz[2];
        CBF::SparseDiagonalMatrix r_matrix(max_diagonal, max_diagonal, offsets);
        return r_matrix;
      }

      // Calculate the fully symmetrical JtJ, i.e. both splines kernels are identical
      std::shared_ptr<MISCMATHS::BFMatrix> make_symmetric_JtJ(
          const std::vector<float>& prod_ima,
          const std::vector<int>& ima_sz,
          const std::vector<float>& spline_x,
          const std::vector<float>& spline_y,
          const std::vector<float>& spline_z,
          const std::vector<int>& coef_sz,
          const std::vector<int>& ksp,
          MISCMATHS::BFMatrixPrecisionType prec) const
      {
        // Create internal sparse matrix
        auto sparse_JtJ = create_empty_JtJ(coef_sz);
        // Make everything a device vector using thrust::
        auto sparse_JtJ_offsets_dev = thrust::device_vector<int>(sparse_JtJ.get_offsets());
        auto prod_ima_dev = thrust::device_vector<float>(prod_ima);
        auto spline_x_dev = thrust::device_vector<float>(spline_x);
        auto spline_y_dev = thrust::device_vector<float>(spline_y);
        auto spline_z_dev = thrust::device_vector<float>(spline_z);
        // Create texture handle
        auto texture_handle = TextureHandle(prod_ima_dev,ima_sz);
        auto tex = texture_handle.get_texture();
        // Get all the raw pointers ready for the Kernel
        float *spline_x_raw = thrust::raw_pointer_cast(spline_x_dev.data());
        float *spline_y_raw = thrust::raw_pointer_cast(spline_y_dev.data());
        float *spline_z_raw = thrust::raw_pointer_cast(spline_z_dev.data());
        int *sparse_JtJ_offsets_raw = thrust::raw_pointer_cast(
            sparse_JtJ_offsets_dev.data());
        float *sparse_JtJ_raw = sparse_JtJ.get_raw_pointer();
        // Calculate parameters for running kernel
        int min_grid_size;
        int threads;
        cudaOccupancyMaxPotentialBlockSize(
            &min_grid_size,
            &threads,
            CBF::kernel_make_jtj_symmetrical,
            0,
            0);
        unsigned int blocks = 172;
        unsigned int chunks = static_cast<unsigned int>(
            std::ceil(float(coef_sz.at(0)*coef_sz.at(1)*coef_sz.at(2))/float(threads)));
        dim3 blocks_2d(blocks,chunks);
        unsigned int smem = (spline_x.size()+spline_y.size()+spline_z.size())*sizeof(float);
        // Call CUDA Kernel
        CBF::kernel_make_jtj_symmetrical<<<blocks_2d,threads,smem>>>(
            // Input
            ima_sz.at(0),
            ima_sz.at(1),
            ima_sz.at(2),
            tex,
            spline_x_raw,
            spline_y_raw,
            spline_z_raw,
            ksp.at(0),
            ksp.at(1),
            ksp.at(2),
            coef_sz.at(0),
            coef_sz.at(1),
            coef_sz.at(2),
            sparse_JtJ_offsets_raw,
            // Output
            sparse_JtJ_raw);
        checkCudaErrors(cudaDeviceSynchronize());
        // Save sparse_JtJ as matrix market file
        auto r_matrix_ptr = sparse_JtJ.convert_to_sparse_bf_matrix(prec);
        return r_matrix_ptr;
      }

      // Calculate the non-symmetrical JtJ, i.e. different kernels
      std::shared_ptr<MISCMATHS::BFMatrix> make_non_symmetric_JtJ(
          const std::vector<float>& prod_ima,
          const std::vector<int>& ima_sz,
          const std::vector<float>& spline_x_1,
          const std::vector<float>& spline_y_1,
          const std::vector<float>& spline_z_1,
          const std::vector<float>& spline_x_2,
          const std::vector<float>& spline_y_2,
          const std::vector<float>& spline_z_2,
          const std::vector<int>& coef_sz_1,
          const std::vector<int>& coef_sz_2,
          const std::vector<int>& ksp_1,
          const std::vector<int>& ksp_2,
          MISCMATHS::BFMatrixPrecisionType prec) const
      {
        // Create internal sparse matrix
        auto sparse_JtJ = create_empty_JtJ(coef_sz_1);
        // Make everything a device vector using thrust::
        auto sparse_JtJ_offsets_dev = thrust::device_vector<int>(sparse_JtJ.get_offsets());
        auto prod_ima_dev = thrust::device_vector<float>(prod_ima);
        auto spline_x_1_dev = thrust::device_vector<float>(spline_x_1);
        auto spline_y_1_dev = thrust::device_vector<float>(spline_y_1);
        auto spline_z_1_dev = thrust::device_vector<float>(spline_z_1);
        auto spline_x_2_dev = thrust::device_vector<float>(spline_x_2);
        auto spline_y_2_dev = thrust::device_vector<float>(spline_y_2);
        auto spline_z_2_dev = thrust::device_vector<float>(spline_z_2);
        // Create texture handle
        auto texture_handle = TextureHandle(prod_ima_dev,ima_sz);
        auto tex = texture_handle.get_texture();
        // Get all the raw pointers ready for the Kernel
        float *spline_x_1_raw = thrust::raw_pointer_cast(spline_x_1_dev.data());
        float *spline_y_1_raw = thrust::raw_pointer_cast(spline_y_1_dev.data());
        float *spline_z_1_raw = thrust::raw_pointer_cast(spline_z_1_dev.data());
        float *spline_x_2_raw = thrust::raw_pointer_cast(spline_x_2_dev.data());
        float *spline_y_2_raw = thrust::raw_pointer_cast(spline_y_2_dev.data());
        float *spline_z_2_raw = thrust::raw_pointer_cast(spline_z_2_dev.data());
        int *sparse_JtJ_offsets_raw = thrust::raw_pointer_cast(
            sparse_JtJ_offsets_dev.data());
        float *sparse_JtJ_raw = sparse_JtJ.get_raw_pointer();
        // Calculate parameters for running kernel
        int min_grid_size;
        int threads;
        cudaOccupancyMaxPotentialBlockSize(
            &min_grid_size,
            &threads,
            CBF::kernel_make_jtj_non_symmetrical,
            0,
            0);
        unsigned int blocks = 343;
        unsigned int chunks = static_cast<unsigned int>(
            std::ceil(float(coef_sz_1.at(0)*coef_sz_1.at(1)*coef_sz_1.at(2))/float(threads)));
        dim3 blocks_2d(blocks,chunks);
        unsigned int smem =
          (spline_x_1.size()+spline_y_1.size()+spline_z_1.size()
           + spline_x_2.size()+spline_y_2.size()+spline_z_2.size())
          * sizeof(float);
        // Call CUDA Kernel
        CBF::kernel_make_jtj_non_symmetrical<<<blocks_2d,threads,smem>>>(
            // Input
            ima_sz.at(0),
            ima_sz.at(1),
            ima_sz.at(2),
            tex,
            spline_x_1_raw,
            spline_y_1_raw,
            spline_z_1_raw,
            spline_x_2_raw,
            spline_y_2_raw,
            spline_z_2_raw,
            ksp_1.at(0),
            ksp_1.at(1),
            ksp_1.at(2),
            ksp_2.at(0),
            ksp_2.at(1),
            ksp_2.at(2),
            coef_sz_1.at(0),
            coef_sz_1.at(1),
            coef_sz_1.at(2),
            coef_sz_2.at(0),
            coef_sz_2.at(1),
            coef_sz_2.at(2),
            sparse_JtJ_offsets_raw,
            // Output
            sparse_JtJ_raw);
        checkCudaErrors(cudaDeviceSynchronize());
        // Save sparse_JtJ as matrix market file
        auto r_matrix_ptr = sparse_JtJ.convert_to_sparse_bf_matrix(prec);
        return r_matrix_ptr;
      }
      //////////////////////////////////////////////////////////////////////////
      // Pimpl Private Datamembers
      //////////////////////////////////////////////////////////////////////////
      BASISFIELD::splinefield spline_field_;
      BASISFIELD::Spline1D<float> spline_x_;
      BASISFIELD::Spline1D<float> spline_y_;
      BASISFIELD::Spline1D<float> spline_z_;
  };

////////////////////////////////////////////////////////////////////////////////
// Main Class
////////////////////////////////////////////////////////////////////////////////
  /// Default dtor
  CBFSplineField::~CBFSplineField() = default;
  /// Move ctor
  CBFSplineField::CBFSplineField(CBFSplineField&& rhs) = default;
  /// Move assignment operator
  CBFSplineField& CBFSplineField::operator=(CBFSplineField&& rhs) = default;
  /// Copy ctor
  CBFSplineField::CBFSplineField(const CBFSplineField& rhs)
  : pimpl_(nullptr)
  {
    if (rhs.pimpl_){
      pimpl_ = CBF::make_unique<Impl>(*rhs.pimpl_);
    }
  }
  /// Copy assignment operator
  CBFSplineField& CBFSplineField::operator=(const CBFSplineField& rhs)
  {
    if (!rhs.pimpl_){
      pimpl_.reset();
    }
    else if (!pimpl_){
      pimpl_ = CBF::make_unique<Impl>(*rhs.pimpl_);
    }
    else{
      *pimpl_ = *rhs.pimpl_;
    }
    return *this;
  }

  // Methods to wrap
  /// Standard BASISFIELD::splinefield constructor
  CBFSplineField::CBFSplineField(const std::vector<unsigned int>& psz,
                                   const std::vector<double>& pvxs,
                                   const std::vector<unsigned int>& pksp,
                                   int porder)
    : pimpl_(CBF::make_unique<Impl>(psz,pvxs,pksp,porder))
  {}

  unsigned int CBFSplineField::CoefSz_x() const
  {
    return pimpl_->CoefSz_x();
  }
  unsigned int CBFSplineField::CoefSz_y() const
  {
    return pimpl_->CoefSz_y();
  }
  unsigned int CBFSplineField::CoefSz_z() const
  {
    return pimpl_->CoefSz_z();
  }
  /// From base class BASISFIELD::basisfield
  void CBFSplineField::AsVolume(NEWIMAGE::volume<float>& vol,
      BASISFIELD::FieldIndex fi)
  {
    return pimpl_->AsVolume(vol,fi);
  }
  /// From base class BASISFIELD::basisfield
  void CBFSplineField::SetCoef(const NEWMAT::ColumnVector& pcoef)
  {
    return pimpl_->SetCoef(pcoef);
  }
  /// Calculate Bending Energy for regularisation purposes
  double CBFSplineField::BendEnergy() const
  {
    return pimpl_->BendEnergy();
  }
  /// Calculate gradient of the Bending Energy
  NEWMAT::ReturnMatrix CBFSplineField::BendEnergyGrad() const
  {
    return pimpl_->BendEnergyGrad();
  }
  /// Calculate hessian of the Bending Energy
  std::shared_ptr<MISCMATHS::BFMatrix> CBFSplineField::BendEnergyHess(
      MISCMATHS::BFMatrixPrecisionType prec) const
  {
    return pimpl_->BendEnergyHess(prec);
  }
  /// Jte V1
  NEWMAT::ReturnMatrix CBFSplineField::Jte(const NEWIMAGE::volume<float>&  ima1,
                           const NEWIMAGE::volume<float>&  ima2,
                           const NEWIMAGE::volume<char>    *mask)
                           const
  {
    return pimpl_->Jte(ima1,ima2,mask);
  }
  /// Jte V2
  NEWMAT::ReturnMatrix CBFSplineField::Jte(const std::vector<unsigned int>&  deriv,
                           const NEWIMAGE::volume<float>&    ima1,
                           const NEWIMAGE::volume<float>&    ima2,
                           const NEWIMAGE::volume<char>      *mask)
                           const
  {
    return pimpl_->Jte(deriv,ima1,ima2,mask);
  }
  /// Jte V3
  NEWMAT::ReturnMatrix CBFSplineField::Jte(const NEWIMAGE::volume<float>&    ima,
                           const NEWIMAGE::volume<char>      *mask)
                           const
  {
    return pimpl_->Jte(ima,mask);
  }
  /// Jte V4
  NEWMAT::ReturnMatrix CBFSplineField::Jte(const std::vector<unsigned int>&  deriv,
                           const NEWIMAGE::volume<float>&    ima,
                           const NEWIMAGE::volume<char>      *mask)
                           const
  {
    return pimpl_->Jte(deriv,ima,mask);
  }
  /// JtJ V1
  std::shared_ptr<MISCMATHS::BFMatrix> CBFSplineField::JtJ(
                                             const NEWIMAGE::volume<float>& ima,
                                             const NEWIMAGE::volume<char> *mask,
                                             MISCMATHS::BFMatrixPrecisionType prec)
                                             const
  {
    return pimpl_->JtJ(ima,mask,prec);
  }
  /// JtJ V2
  std::shared_ptr<MISCMATHS::BFMatrix> CBFSplineField::JtJ(
                                             const NEWIMAGE::volume<float>& ima1,
                                             const NEWIMAGE::volume<float>& ima2,
                                             const NEWIMAGE::volume<char> *mask,
                                             MISCMATHS::BFMatrixPrecisionType prec)
                                             const
  {
    return pimpl_->JtJ(ima1,ima2,mask,prec);
  }
  /// JtJ V3
  std::shared_ptr<MISCMATHS::BFMatrix> CBFSplineField::JtJ(
                                             const std::vector<unsigned int>& deriv,
                                             const NEWIMAGE::volume<float>& ima,
                                             const NEWIMAGE::volume<char> *mask,
                                             MISCMATHS::BFMatrixPrecisionType prec)
                                             const
  {
    return pimpl_->JtJ(deriv,ima,mask,prec);
  }
  /// JtJ V4
  std::shared_ptr<MISCMATHS::BFMatrix> CBFSplineField::JtJ(
                                             const std::vector<unsigned int>& deriv,
                                             const NEWIMAGE::volume<float>& ima1,
                                             const NEWIMAGE::volume<float>& ima2,
                                             const NEWIMAGE::volume<char> *mask,
                                             MISCMATHS::BFMatrixPrecisionType prec)
                                             const
  {
    return pimpl_->JtJ(deriv,ima1,ima2,mask,prec);
  }
  /// JtJ V5
  std::shared_ptr<MISCMATHS::BFMatrix> CBFSplineField::JtJ(
                                             const std::vector<unsigned int>& deriv1,
                                             const NEWIMAGE::volume<float>& ima1,
                                             const std::vector<unsigned int>& deriv2,
                                             const NEWIMAGE::volume<float>& ima2,
                                             const NEWIMAGE::volume<char> *mask,
                                             MISCMATHS::BFMatrixPrecisionType prec)
                                             const
  {
    return pimpl_->JtJ(deriv1,ima1,deriv2,ima2,mask,prec);
  }
  /// JtJ V6
  std::shared_ptr<MISCMATHS::BFMatrix> CBFSplineField::JtJ(
                                             const NEWIMAGE::volume<float>& ima1,
                                             const BASISFIELD::basisfield& bf2,
                                             const NEWIMAGE::volume<float>& ima2,
                                             const NEWIMAGE::volume<char> *mask,
                                             MISCMATHS::BFMatrixPrecisionType prec)
                                             const
  {
    return pimpl_->JtJ(ima1,bf2,ima2,mask,prec);
  }
} // namespace CBF
