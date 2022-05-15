/*  Copyright (C) 1999-2004 University of Oxford  */

/*  CCOPYRIGHT */

#ifndef _triangle
#define _triangle
#include <iostream>
#include <vector>

#include"point.h"

namespace mesh{

class Mpoint;

class Triangle
{
 private:
  Mpoint * _vertice[3];
  float _value;

 public:
  Triangle(Mpoint* const p1,Mpoint* const p2,Mpoint* const p3, float val=0);
  ~Triangle(void);
  Triangle(Triangle & t);          //prevents from using copy constructor
  Triangle operator=(Triangle &t); //prevents from using affectation operator

  std::vector<double> data; //can be used to store extra-data attached to the triangle

  const Pt centroid() const;
  const Vec normal() const;
  const Vec area(const Mpoint* p) const;
  Mpoint * const get_vertice(const int i) const;
  const float get_value() const {return _value;};
  void set_value(const float val){_value=val;};
  void swap();             //changes triangle orientation
  const int operator <(const Triangle *const t) const;   //checks if two triangles are adjacents and well-oriented
  const bool operator ==(const Triangle & t) const;
  const bool intersect(const Triangle & t) const; //checks if two triangles intersect
  const bool intersect(const std::vector<Pt> & p)const; // checks if a segment intersects the triangle
  const bool intersect(const std::vector<Pt> & p,int& ind)const; // checks if a segment intersects the triangle
  bool oriented;           //has the triangle been well oriented ?


};

}
#endif
