/*  Copyright (C) 1999-2004 University of Oxford  */

/*  CCOPYRIGHT */

#ifndef _point
#define _point

#include <iostream>
#include <cmath>

namespace mesh{

class Vec {
 public:

  Vec() : X(0), Y(0), Z(0){};
  Vec(double x, double y, double z) : X(x), Y(y), Z(z){};
  Vec(const Vec& v) : X(v.X), Y(v.Y), Z(v.Z){};
  Vec operator = (const Vec& v)
    {
      X = v.X;
      Y = v.Y;
      Z = v.Z;
      return *this;
    }

  //newmat conversions
  /*
    RowVector RV() {
    RowVector result(3);
    result(0) = X;
    result(1) = Y;
    result(2) = Z;
    return result;
    };

    Vec(RowVector r) {
    X = r(0);
    Y = r(1);
    Z = r(2);
    }

    Vec operator=(const RowVector & r){
    X = r(0);
    Y = r(1);
    Z = r(2);
    return *this
    }

  */

  double X;
  double Y;
  double Z;

  const inline double norm() const
    {
      return (sqrt(X*X + Y*Y + Z*Z));
    }

  inline void operator+=(const Vec v)
    {
      X+=v.X;
      Y+=v.Y;
      Z+=v.Z;
    }

  void normalize(){
    double n = norm();
    if (n!=0){
    X/=n;
    Y/=n;
    Z/=n;}
  }

};

const double operator|(const Vec &v1, const Vec &v2);
const Vec operator*(const double &d, const Vec &v);
const Vec operator+(const Vec &v1, const Vec &v2);
const Vec operator-(const Vec &v1, const Vec &v2);
const Vec operator*(const Vec &v1, const Vec &v2);
const Vec operator/(const Vec &v, const double &d);
const Vec operator*(const Vec &v, const double &d);

class Pt {
 public:
  Pt() : X(0), Y(0), Z(0){};
  Pt(double x, double y, double z) : X(x), Y(y), Z(z){};
  Pt (const Pt& p) : X(p.X), Y(p.Y), Z(p.Z){};
  Pt operator =(const Pt& p)
    {
      X = p.X;
      Y = p.Y;
      Z = p.Z;
      return *this;
    }

  double X;
  double Y;
  double Z;

  inline void operator+=(const Pt p)
    {
      X=X+p.X;
      Y=Y+p.Y;
      Z=Z+p.Z;
    }
  inline void operator*=(const double d)
    {
      X*=d;
      Y*=d;
      Z*=d;
    }

  inline void operator/=(const double d)
    {
      if (d!=0)
	{
	  X/=d;
	  Y/=d;
	  Z/=d;
	}
      else std::cerr << "division by zero" << std::endl;
    }


  inline bool operator==(const Pt &p) const
    {
      return((std::fabs(X-p.X)<1e-8) && (std::fabs(Y-p.Y)<1e-8) && (std::fabs(Z-p.Z)<1e-8));
    }

};

const Pt operator + (const Pt &p, const Vec &v);
const Vec operator-(const Pt &p1, const Pt &p2);

}

#endif
