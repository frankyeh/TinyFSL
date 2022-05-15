/*  Copyright (C) 1999-2004 University of Oxford  */

/*  CCOPYRIGHT */

#ifndef _mpoint
#define _mpoint

#include <list>
#include <iostream>

#include "point.h"

namespace mesh{

class Triangle;

class Mpoint {
 public:
  ~Mpoint(void);
  Mpoint(double x, double y, double z, int counter,float val=0);
  Mpoint(const Pt p, int counter,float val=0);

  Mpoint (Mpoint & m); //prevents from using copy constructor
  Mpoint operator=(Mpoint & m); //prevents from using affectation operator

  void translation(const double x,const double y,const double z);
  void rescale(const double t, const double x, const double y, const double z);
  void rescale(const double tx, const double ty, const double tz, const Pt p);
  void rotation(const double r11, const double r12, const double r13, const double r21, const double r22, const double r23, const double r31, const double r32, const double r33,const double x, const double y, const double z);

  void update();

  Pt _update_coord;
  std::list<Triangle*> _triangles;

  const Vec local_normal() const;
  const Pt medium_neighbours() const;
  const Vec difference_vector() const;
  const Vec orthogonal() const;
  const Vec tangential() const;
  const Vec max_triangle() const;
  const double medium_distance_of_neighbours() const;

  const Pt get_coord() const {return _coord;};
  void set_coord(const Pt& coord){_coord=coord;}
  const int get_no() const {return _no;};
  const float get_value() const {return _value;};
  void set_value(const float val){_value=val;};
  std::list<Mpoint *> _neighbours;

  std::list<double> data; //can be used to store extra-data attached to the point

 private:
  Pt _coord;
  const int _no;
  float _value;

};

const bool operator ==(const Mpoint &p2, const Mpoint &p1);
const bool operator ==(const Mpoint &p2, const Pt &p1);
const Vec operator -(const Mpoint&p1, const Mpoint &p2);
const bool operator <(const Mpoint &p1, const Mpoint &p2); //checks if p1 and p2 are adjacent
const Vec operator -(const Pt&p1, const Mpoint &p2);
const Vec operator -(const Mpoint&p1, const Pt &p2);

}

#endif
