// Topup - FMRIB's Tool for correction of susceptibility induced distortions
//
// topup.cpp
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2012 University of Oxford

/*  CCOPYRIGHT  */

#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <algorithm>
#include <memory>
#include "armawrap/newmat.h"
#ifndef EXPOSE_TREACHEROUS
#define EXPOSE_TREACHEROUS           // To allow us to use .set_sform etc
#endif
#include "newimage/newimageall.h"
#include "miscmaths/miscmaths.h"
#include "miscmaths/nonlin.h"
#include "warpfns/warpfns.h"
#include "basisfield/basisfield.h"
#include "basisfield/splinefield.h"
#include "basisfield/dctfield.h"
#include "topup_costfunctions.h"
#include "topupfns.h"

using namespace std;
using namespace NEWMAT;
using namespace MISCMATHS;
using namespace NEWIMAGE;
using namespace BASISFIELD;
using namespace TOPUP;

int main(int   argc,
         char  *argv[])
{

  // Read command line input
  std::shared_ptr<topup_clp> clp;
  try {
    clp = parse_topup_command_line(argc,argv);
  }
  catch (const std::exception& error) {
    cerr << "Topup: Error occured when parsing the command line" << endl;
    cerr << "Exception thrown with message: " << error.what() << endl;
    exit(EXIT_FAILURE);
  }

  // Read input images
  NEWIMAGE::volume4D<float>  in;
  try {
    read_volume4D(in,clp->ImaFname());
  }
  catch (const std::exception& error) {
    cerr << "Error occurred when reading --imain file " << clp->ImaFname() << endl;
    cerr << "Exception thrown with message: " << error.what() << endl;
    exit(EXIT_FAILURE);
  }

  // Prepare for running

  // SCale images to obtain a consistent weighting against regularisation,
  // and possibly between scans if the scale has not been preserved.

  std::vector<double> means(in.tsize());
  double gmean = 0.0;
  for (int i=0; i<in.tsize(); i++) {
    means[i] = in[i].mean();
    gmean += means[i];
  }
  gmean /= in.tsize();
  if (clp->IndividualScaling()) {
    for (int i=0; i<in.tsize(); i++) in[i] *= 100.0/means[i];
  }
  else in *= 100.0/gmean;

  std::shared_ptr<TopupCF>      cf;
  std::shared_ptr<NonlinParam>  nlpar;
  try {
    // Create cost-function object and
    // set properties for first level

    cf = std::shared_ptr<TopupCF>(new TopupCF(in,clp->PhaseEncodeVectors(),clp->ReadoutTimes(),clp->WarpRes(1),clp->SplineOrder()));
    cf->SetTracePrint(clp->Trace());
    cf->SetVerbose(clp->Verbose());
    cf->SetDebug(clp->DebugLevel());
    cf->SetLevel(1);
    cf->SetInterpolationModel(clp->InterpolationModel());
    cf->SetRegridding(clp->Regridding(in));
    cf->SubSample(clp->SubSampling(1));
    cf->Smooth(clp->FWHM(1));
    cf->SetMovementsFixed(!clp->EstimateMovements(1));
    cf->SetRegularisation(clp->Lambda(1),clp->RegularisationModel());
    cf->SetSSQLambda(clp->SSQLambda());
    cf->SetHessianPrecision(clp->HessianPrecision());

    // Create non-linear parameters object
    ColumnVector spar(cf->NPar());
    spar = 0;
    nlpar = std::shared_ptr<NonlinParam>(new NonlinParam(cf->NPar(),clp->OptimisationMethod(1),spar));
    if (nlpar->Method() == MISCMATHS::NL_LM) {
      nlpar->SetEquationSolverMaxIter(500);
      nlpar->SetEquationSolverTol(1.0e-3);
    }
    nlpar->SetMaxIter(clp->MaxIter(1));
  }
  catch (const std::exception& error) {
    cerr << "Error occurred when preparing to run topup" << endl;
    cerr << "Exception thrown with message: " << error.what() << endl;
    exit(EXIT_FAILURE);
  }

  // Run minimisation at first level

  try {
    nonlin(*nlpar,*cf);
  }
  catch (const std::exception& error) {
    cerr << "Error occurred when running first level of topup" << endl;
    cerr << "Exception thrown with message: " << error.what() << endl;
    exit(EXIT_FAILURE);
  }

  // Run remaining levels to refine solution.

  unsigned int l=2;
  try {
    for ( ; l<=clp->NoOfLevels(); l++) {
      if (clp->Verbose()) cout << "***Going to next resolution level***" << endl;
      // Change settings for cost-function object
      cf->SetLevel(l);
      cf->SubSample(clp->SubSampling(l));
      cf->Smooth(clp->FWHM(l));
      cf->SetWarpResolution(clp->WarpRes(l));
      cf->SetMovementsFixed(!clp->EstimateMovements(l));
      cf->SetRegularisation(clp->Lambda(l),clp->RegularisationModel());
      // Make a new nonlinear object
      nlpar = std::shared_ptr<NonlinParam>(new NonlinParam(cf->NPar(),clp->OptimisationMethod(l),cf->Par()));
      if (nlpar->Method() == MISCMATHS::NL_LM) {
        nlpar->SetEquationSolverMaxIter(500);
        nlpar->SetEquationSolverTol(1.0e-3);
      }
      nlpar->SetMaxIter(clp->MaxIter(l));
      try {
        nonlin(*nlpar,*cf);
      }
      catch (const std::exception& error) {
        cerr << "Error occurred when running level " << l << " of topup" << endl;
        cerr << "Exception thrown with message: " << error.what() << endl;
        exit(EXIT_FAILURE);
      }
    }
  }
  catch (const std::exception& error) {
    cerr << "Error occurred when preparing to run level " << l << " of topup" << endl;
    cerr << "Exception thrown with message: " << error.what() << endl;
    exit(EXIT_FAILURE);
  }

  // Save Everything we have been asked so save
  try {
    if (clp->SubSampling(clp->NoOfLevels()) > 1) {
      cf->Smooth(0.0);
      cf->SubSample(1);
    }
    cf->WriteCoefficients(clp->CoefFname());
    cf->WriteMovementParameters(clp->MovParFname());
    if (clp->FieldFname().size()) cf->WriteField(clp->FieldFname(),in);
    if (clp->ImaOutFname().size()) cf->WriteUnwarped(clp->ImaOutFname(),in,gmean/100.0);
    if (clp->DisplacementFieldBaseFname().size()) {
      cf->WriteDisplacementFields(clp->DisplacementFieldBaseFname());
    }
    if (clp->RigidBodyBaseFname().size()) {
      cf->WriteRigidBodyMatrices(clp->RigidBodyBaseFname());
    }
    if (clp->JacobianBaseFname().size()) {
      cf->WriteJacobians(clp->JacobianBaseFname());
    }
  }
  catch(const std::exception& error) {
    cerr << "Error occured while writing output from topup" << endl;
    cerr << "Exception thrown with message: " << error.what() << endl;
    exit(EXIT_FAILURE);
  }

  exit(EXIT_SUCCESS);
}
