/*  Copyright (C) 1999-2004 University of Oxford  */

/*  CCOPYRIGHT */

#include "point.h"

using namespace std;

namespace mesh {

const Vec operator*(const double &d, const Vec &v)
{
  return Vec(d*v.X, d*v.Y, d*v.Z);
}

const double operator|(const Vec &v1, const Vec &v2)
{
  return v1.X*v2.X+ v1.Y*v2.Y+ v1.Z*v2.Z;
}

const Vec operator+(const Vec &v1, const Vec &v2)
{
  return Vec(v1.X+v2.X, v1.Y+v2.Y, v1.Z+v2.Z);
}

const Vec operator-(const Vec &v1, const Vec &v2)
{
  return Vec(v1.X-v2.X, v1.Y-v2.Y, v1.Z-v2.Z);
}

const Vec operator*(const Vec &v1, const Vec &v2)
{
  return(Vec(v1.Y * v2.Z - v1.Z * v2.Y,
	     v2.X * v1.Z - v2.Z * v1.X,
	     v1.X * v2.Y - v2.X * v1.Y));
}

const Vec operator/(const Vec &v, const double &d)
{
  if (d!=0)
    {
      return(Vec(v.X/d, v.Y/d, v.Z/d));
    }
  else {cerr<<"division by zero"<<endl; return v;}
}

const Vec operator*(const Vec &v, const double &d)
{
  return(Vec(v.X*d, v.Y*d, v.Z*d));
}

const Pt operator + (const Pt &p, const Vec &v)
{
  return Pt(p.X+v.X, p.Y+v.Y, p.Z+v.Z);
}

const Vec operator-(const Pt &p1, const Pt &p2)
{
  return Vec(p1.X-p2.X, p1.Y-p2.Y, p1.Z-p2.Z);
}


}
