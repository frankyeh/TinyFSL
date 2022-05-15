/*  histogram.h

    Mark Woolrich, Matthew Webster and Emma Robinson, FMRIB Image Analysis Group

    Copyright (C) 1999-2000 University of Oxford  */

/*  CCOPYRIGHT  */

#if !defined(_histogram_h)
#define _histogram_h

#include <iostream>
#include <fstream>
#define WANT_STREAM
#define WANT_MATH

#include "armawrap/newmatap.h"
#include "armawrap/newmatio.h"
#include "miscmaths.h"

namespace MISCMATHS {

  class Histogram
    {
    public:
      Histogram(){};
      const Histogram& operator=(const Histogram& in){
	sourceData=in.sourceData; calcRange=in.calcRange; histMin=in.histMin; histMax=in.histMax; bins=in.bins; histogram=in.histogram; CDF=in.CDF; datapoints=in.datapoints; exclusion=in.exclusion;
	return *this;
      }

      Histogram(const Histogram& in){ *this=in;}

      Histogram(const NEWMAT::ColumnVector& psourceData, int numBins)
	: sourceData(psourceData), calcRange(true), bins(numBins){}

      Histogram(const NEWMAT::ColumnVector& psourceData, float phistMin, float phistMax, int numBins)
	: sourceData(psourceData), calcRange(false), histMin(phistMin), histMax(phistMax), bins(numBins){}

      void set(const NEWMAT::ColumnVector& psourceData, int numBins) {
	sourceData=psourceData; calcRange=true; bins=numBins;
      }

      void set(const NEWMAT::ColumnVector& psourceData, float phistMin, float phistMax, int numBins) {
	sourceData=psourceData; calcRange=false; histMin=phistMin; histMax=phistMax; bins=numBins;
      }

      void generate();
      void generate(NEWMAT::ColumnVector exclusion_values);
      NEWMAT::ColumnVector generateCDF();

      float getHistMin() const {return histMin;}
      float getHistMax() const {return histMax;}
      void setHistMax(float phistMax) {histMax = phistMax;}
      void setHistMin(float phistMin) {histMin = phistMin;}
      void setexclusion(NEWMAT::ColumnVector exclusion_values) {exclusion =exclusion_values;}
      void smooth();

      int integrateAll() {return sourceData.Nrows();}

      const NEWMAT::ColumnVector& getData() {return histogram;}
      void setData(const NEWMAT::ColumnVector& phist) { histogram=phist;}

      int integrateToInf(float value) const { return integrate(value, histMax); }
      int integrateFromInf(float value) const { return integrate(histMin, value); }
      int integrate(float value1, float value2) const;

      void match(Histogram &);

      float mode() const;

      int getBin(float value) const;
      float getValue(int bin) const;
      float getPercentile(float perc);

      inline int getNumBins() const {return bins;}
      inline NEWMAT::ColumnVector getCDF() const {return CDF;}
      inline NEWMAT::ColumnVector getsourceData()const {return sourceData;}
    protected:

    private:

      NEWMAT::ColumnVector sourceData;
      NEWMAT::ColumnVector histogram;
      NEWMAT::ColumnVector exclusion;
      NEWMAT::ColumnVector CDF;

      bool calcRange;

      float histMin;
      float histMax;

      int bins; // number of bins in histogram
      int datapoints;
    };

  inline int Histogram::getBin(float value) const
    {
      float binwidth=(histMax-histMin)/bins;
      return Max(1, Min((int)((((float)bins)*((float)(value-(histMin-binwidth))))/((float)(histMax-histMin))),bins));
    }

  inline float Histogram::getValue(int bin) const
    {
      return (bin*(histMax-histMin))/(float)bins + histMin;
    }

  inline NEWMAT::ColumnVector Histogram::generateCDF()
    {


      CDF.ReSize(bins);


      CDF(1)=histogram(1)/datapoints;

      for (int i=2;i<=bins;i++)
	CDF(i)=CDF(i-1)+ histogram(i)/datapoints;


      return CDF;
    }
}

#endif
