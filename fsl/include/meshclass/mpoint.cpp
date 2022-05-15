/*  Copyright (C) 1999-2004 University of Oxford  */

/*  CCOPYRIGHT */

#include "mpoint.h"
#include "triangle.h"
#include "point.h"

using namespace std;

namespace mesh{

Mpoint::~Mpoint(void){
}

Mpoint::Mpoint(double x, double y, double z, int counter, float val): _no(counter), _value(val) {
  _coord=Pt(x, y, z);
  _update_coord=Pt(0, 0, 0);
}

Mpoint::Mpoint(const Pt p, int counter,float val):_no(counter),_value(val) {
  _coord = p;
  _update_coord=Pt(0, 0, 0);
}


void Mpoint::translation(const double x,const double y,const double z)
{
  _coord+= Pt(x, y, z);
}
void Mpoint::rotation(const double r11, const double r12, const double r13, const double r21, const double r22, const double r23, const double r31, const double r32, const double r33,const double x, const double y, const double z)
{
	Vec cen=_coord - Pt(x, y, z);

  _coord = Pt(x, y, z) + Vec( (cen.X*r11+cen.Y*r12+cen.Z*r13) , (cen.X*r21+cen.Y*r22+cen.Z*r23) , (cen.X*r31+cen.Y*r32+cen.Z*r33));



}
void Mpoint::rescale(const double t, const double x, const double y, const double z)
{
  _coord = Pt(x, y, z) + t*(_coord - Pt(x, y, z));
}

void Mpoint::rescale(const double tx, const double ty, const double tz, const Pt p)
{
  _coord.X = p.X + tx * (_coord.X - p.X);
  _coord.Y = p.Y + ty * (_coord.Y - p.Y);
  _coord.Z = p.Z + tz * (_coord.Z - p.Z);
}

void Mpoint::update() {
  _coord = _update_coord;
}

const Vec Mpoint::local_normal() const
{
  Vec v(0, 0, 0);
  for (list<Triangle*>::const_iterator i = _triangles.begin(); i!=_triangles.end(); i++)
    {
      v+=(*i)->normal();
    }
  v.normalize();
  return v;
}

const Pt Mpoint::medium_neighbours() const
{
  Pt resul(0, 0, 0);
  int counter=_neighbours.size();
  for (list<Mpoint*>::const_iterator i = _neighbours.begin(); i!=_neighbours.end(); i++)
    {
      resul+=(*i)->_coord;
    }
  resul=Pt(resul.X/counter, resul.Y/counter, resul.Z/counter);
  return resul;
}

const Vec Mpoint::difference_vector() const
{
  return medium_neighbours() - _coord;
}

const Vec Mpoint::orthogonal() const
{
  Vec n = local_normal();
  return n * (difference_vector()| n);
}

const Vec Mpoint::tangential() const
{
  return (difference_vector() - orthogonal());
}

const double Mpoint::medium_distance_of_neighbours() const
{
  double l = 0;
  for (list<Mpoint*>::const_iterator i=_neighbours.begin(); i!=_neighbours.end(); i++)
    {
      l+=(*(*i)-*this).norm();
    }
  l/=_neighbours.size();
  return l;
}
const Vec Mpoint::max_triangle() const
{
  //returns a vector pointing from the vertex to the triangle centroid, scaled by the triangle area

  vector<float> Areas;
  int ind=0;
  Vec vA,temp;
  for (list<Triangle*>::const_iterator i=_triangles.begin(); i!=_triangles.end(); i++)
    {
      temp=(*i)->area(this);
      Areas.push_back(temp.norm());
		//don't need to store in vector anymore
      if (Areas.back() >= Areas.at(ind)){
		ind=Areas.size()-1;
		vA = temp;
      }
    }

  return vA;
}


const bool operator ==(const Mpoint &p2, const Mpoint &p1){
  return (fabs(p1.get_coord().X- p2.get_coord().X)<1e-8 && fabs(p1.get_coord().Y - p2.get_coord().Y)<1e-8 && fabs(p1.get_coord().Z - p2.get_coord().Z)<1e-8);
}
const bool operator ==(const Mpoint &p1, const Pt &p2){
  return (fabs(p1.get_coord().X- p2.X)<1e-2 && fabs(p1.get_coord().Y - p2.Y)<1e-2 && fabs(p1.get_coord().Z - p2.Z)<1e-2);
}

const Vec operator -(const Mpoint&p1, const Mpoint &p2){
  return Vec (p1.get_coord().X - p2.get_coord().X,p1.get_coord().Y - p2.get_coord().Y,p1.get_coord().Z - p2.get_coord().Z );
}

const Vec operator -(const Pt&p1, const Mpoint &p2){
  return Vec (p1.X - p2.get_coord().X,p1.Y - p2.get_coord().Y,p1.Z - p2.get_coord().Z );
}

const Vec operator -(const Mpoint&p1, const Pt &p2){
  return Vec (p1.get_coord().X - p2.X,p1.get_coord().Y - p2.Y,p1.get_coord().Z - p2.Z );
}

const bool operator <(const Mpoint &p1,const Mpoint &p2){
  bool result = false;
  for (list<Mpoint *>::const_iterator i= p1._neighbours.begin(); i!=p1._neighbours.end();i++){
    if (*(*i)==p2) result = true;
  }
  return result;
}




}
