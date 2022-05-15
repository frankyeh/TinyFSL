
/*  BET - Brain Extraction Tool

    BETv1 Steve Smith
    BETv2 Mickael Pechaud, Mark Jenkinson, Steve Smith
    FMRIB Image Analysis Group

    Copyright (C) 1999-2003 University of Oxford  */

/*  CCOPYRIGHT  */

#ifndef _t1only
#define _t1only

#include "meshclass/meshclass.h"
#include "newimage/newimageall.h"
#include "utils/options.h"

struct trMatrix
{
  double m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34, m41, m42, m43, m44;
};

NEWIMAGE::volume<short> default_volume;

void draw_segment(NEWIMAGE::volume<short>& image, const mesh::Pt& p1, const mesh::Pt& p2);

void draw_mesh(NEWIMAGE::volume<short>& image, const mesh::Mesh &m);

NEWIMAGE::volume<short> make_mask_from_mesh(const NEWIMAGE::volume<short> & image, const mesh::Mesh& m);

double standard_step_of_computation(const NEWIMAGE::volume<float> & image, mesh::Mesh & m, const int iteration_number, const double E,const double F, const float addsmooth, const float speed, const int nb_iter, const int id=5, const int od=15, const bool vol=false, const NEWIMAGE::volume<short> & mask=default_volume);

std::vector<double> t1only_special_extract(const NEWIMAGE::volume<float> & t1, const mesh::Pt & point, const mesh::Vec & n) ;

std::vector<double> t1only_co_ext(const NEWIMAGE::volume<float> & t1, const mesh::Pt & point, const mesh::Vec & n) ;

void t1only_write_ext_skull(NEWIMAGE::volume<float> & output_inskull, NEWIMAGE::volume<float> & output_outskull, NEWIMAGE::volume<float> & output_outskin, const NEWIMAGE::volume<float> & t1, const mesh::Mesh & m, const trMatrix & M) ;

int t1only_main(int argc, char *argv[], int nb_pars, Utilities::OptionParser & options);

std::vector<double> special_extract(const NEWIMAGE::volume<float> & t1, const NEWIMAGE::volume<float> & t2, const mesh::Pt & point, const mesh::Vec & n, NEWIMAGE::volume<short> & csfvolume);

std::vector<double> co_ext(const NEWIMAGE::volume<float> & t1, const NEWIMAGE::volume<float> & t2, const mesh::Pt & point, const mesh::Vec & n, NEWIMAGE::volume<short> & csfvolume);

bool special_case(const mesh::Pt & point, const mesh::Vec & n, const trMatrix & M);

void write_ext_skull(NEWIMAGE::volume<float> & output_inskull, NEWIMAGE::volume<float> & output_outskull, NEWIMAGE::volume<float> & output_outskin, const NEWIMAGE::volume<float> & t1, const NEWIMAGE::volume<float> & t2, const mesh::Mesh & m, const trMatrix & M, NEWIMAGE::volume<short> & csfvolume);


#endif
