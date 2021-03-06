/*  Copyright (C) 1999-2004 University of Oxford  */

/*  CCOPYRIGHT */

#include "triangle.h"
#include "mpoint.h"

using namespace std;

namespace mesh{

//this constructor also puts the connexions between the points.
Triangle::Triangle(Mpoint* const p1, Mpoint* const p2, Mpoint* const p3,float val):_value(val) {
  oriented = false;
  _vertice[0]=p1;
  _vertice[1]=p2;
  _vertice[2]=p3;

  p1->_triangles.push_back(this);
  p2->_triangles.push_back(this);
  p3->_triangles.push_back(this);

  p1->_neighbours.remove(p2);
  p1->_neighbours.remove(p3);
  p2->_neighbours.remove(p3);
  p2->_neighbours.remove(p1);
  p3->_neighbours.remove(p1);
  p3->_neighbours.remove(p2);

  p1->_neighbours.push_back(p2);
  p1->_neighbours.push_back(p3);
  p2->_neighbours.push_back(p3);
  p2->_neighbours.push_back(p1);
  p3->_neighbours.push_back(p1);
  p3->_neighbours.push_back(p2);

}


//warning, you should remove neighbourhood relations between points by hand
Triangle::~Triangle() {
  _vertice[0]->_triangles.remove(this);
  _vertice[1]->_triangles.remove(this);
  _vertice[2]->_triangles.remove(this);
}


const Pt Triangle::centroid() const{
return Pt((_vertice[0]->get_coord().X +_vertice[1]->get_coord().X +_vertice[2]->get_coord().X)/3,
(_vertice[0]->get_coord().Y +_vertice[1]->get_coord().Y +_vertice[2]->get_coord().Y)/3,
(_vertice[0]->get_coord().Z +_vertice[1]->get_coord().Z +_vertice[2]->get_coord().Z)/3
);
}

const Vec Triangle::normal() const{
  Vec result = (_vertice[2]->get_coord() - _vertice[0]->get_coord()) * (_vertice[1]->get_coord() - _vertice[0]->get_coord());
  return result;
}

const Vec Triangle::area(const Mpoint* const p) const{
  Vec v1,v2,vA;
  float Tarea;

  //calculate
  v1=*_vertice[1]-*_vertice[0];
  v2=*_vertice[2]-*_vertice[0];
  Tarea=0.5*((v1*v2).norm());
  //find appriopriate vector
  for (int i = 0; i<3; i++){
    if (p==_vertice[i]){
      vA=(this->centroid())-*_vertice[i];
    }
  }
  vA=vA/vA.norm()*Tarea;

  return vA;
}



Mpoint * const Triangle::get_vertice(const int i) const
{return _vertice[i];}

void Triangle::swap() {
  Mpoint * p = _vertice[1];
  _vertice[1] = _vertice[2];
  _vertice[2] = p;
}

//check if two triangles are adjacents
//0 if not
//1 if yes and good orientation
//2 if yes and bad orientation
const int Triangle::operator <(const Triangle * const t) const{
  int c = 0;
  int a11=-1, a12=-1, a21=-1, a22=-1;
  for (int i=0; i<3; i++)
    for (int j=0; j<3; j++)
      if (_vertice[i]==t->_vertice[j])
	{
	  if (a11 == -1) {a11=i; a21=j;}
	  else {a12=i; a22=j;};
	  c++;//cout<<i<<"++"<<j<<endl;
	};
  if (c == 2) {if ((a12-a11 + a22-a21) % 3 == 0) return 1;
  else return 2;}
  else return 0;
}

const bool Triangle::operator ==(const Triangle & t) const
{
  return ((*get_vertice(0)) == *(t.get_vertice(0)) && (*get_vertice(1)) == *(t.get_vertice(1)) && (*get_vertice(2)) == *(t.get_vertice(2)));
}

const bool Triangle::intersect(const Triangle & t) const
{
  bool result = false;
  Vec normal = (this->get_vertice(0)->get_coord() - this->get_vertice(1)->get_coord()) * (this->get_vertice(0)->get_coord() - this->get_vertice(2)->get_coord());

  if ((normal|(t.get_vertice(0)->get_coord() - this->get_vertice(0)->get_coord())) * (normal|(t.get_vertice(1)->get_coord() - this->get_vertice(0)->get_coord())) < 0)
    {
      //possible intersection -> make the full test
      //test from *this
      for (int i = 0; i < 3; i++)
	if ((normal|(t.get_vertice(i)->get_coord() - this->get_vertice(0)->get_coord())) * (normal|(t.get_vertice((i+1)%3)->get_coord() - this->get_vertice(0)->get_coord())) < 0)
	  {
	    Vec v1 = this->get_vertice(1)->get_coord() - this->get_vertice(0)->get_coord();
	    Vec v2 = this->get_vertice(2)->get_coord() - this->get_vertice(0)->get_coord();
	    Vec v3 = this->get_vertice(2)->get_coord() - this->get_vertice(1)->get_coord();
	    Vec v = v1 * v2;

	    Vec p1 = t.get_vertice(i)->get_coord() - this->get_vertice(0)->get_coord();
	    Vec d1 = t.get_vertice((i+1)%3)->get_coord() - t.get_vertice(i)->get_coord();
	    double denom = (d1.X * v.X + d1.Y * v.Y + d1.Z * v.Z);
	    if (denom != 0)
	      {
		double lambda1 = - (p1.X * v.X + p1.Y * v.Y + p1.Z * v.Z)/denom;
		Vec proj1 = p1 + (d1 * lambda1);

		//checks if proj is inside the triangle ...
		bool inside = false;
		Vec n1 = v1 * proj1;
		Vec n2 = proj1 * v2;
		Vec n3 = v3 * (proj1 + (v1 * -1));
		if (((n1 | n3) > 0 & (n2 | n3) > 0 & (n1 | n2) > 0) | ((n1 | n3) < 0 & (n2 | n3) < 0 & (n1 | n2) < 0) )
		  inside = true;

		result = result | inside;
	      }
	  }

      //test from t

      Vec normalt = (t.get_vertice(0)->get_coord() - t.get_vertice(1)->get_coord()) * (t.get_vertice(0)->get_coord() - t.get_vertice(2)->get_coord());
      for (int i = 0; i < 3; i++)
	if ((normalt|(this->get_vertice(i)->get_coord() - t.get_vertice(0)->get_coord())) * (normalt|(this->get_vertice((i+1)%3)->get_coord() - t.get_vertice(0)->get_coord())) < 0)
	  {
	    Vec v1 = t.get_vertice(1)->get_coord() - t.get_vertice(0)->get_coord();
	    Vec v2 = t.get_vertice(2)->get_coord() - t.get_vertice(0)->get_coord();
	    Vec v3 = t.get_vertice(2)->get_coord() - t.get_vertice(1)->get_coord();
	    Vec v = v1 * v2;

	    Vec p1 = this->get_vertice(i)->get_coord() - t.get_vertice(0)->get_coord();
	    Vec d1 = this->get_vertice((i+1)%3)->get_coord() - this->get_vertice(i)->get_coord();

	    double denom = (d1.X * v.X + d1.Y * v.Y + d1.Z * v.Z);
	    if (denom != 0)
	      {
		double lambda1 = - (p1.X * v.X + p1.Y * v.Y + p1.Z * v.Z)/denom;
		Vec proj1 = p1 + (d1 * lambda1);

		//checks if proj is inside the triangle ...
		bool inside = false;
		Vec n1 = v1 * proj1;
		Vec n2 = proj1 * v2;
		Vec n3 = v3 * (proj1 + (v1 * -1));
		if (((n1 | n3) > 0 & (n2 | n3) > 0 & (n1 | n2) > 0) | ((n1 | n3) < 0 & (n2 | n3) < 0 & (n1 | n2) < 0) )
		  inside = true;

		result = result | inside;
	      }
	  }



    }
  else if ((normal|(t.get_vertice(0)->get_coord() - this->get_vertice(0)->get_coord())) * (normal|(t.get_vertice(2)->get_coord() - this->get_vertice(0)->get_coord())) < 0)
    {
      //possible intersection -> make the full test
      //test from *this
      for (int i = 0; i < 3; i++)
	if ((normal|(t.get_vertice(i)->get_coord() - this->get_vertice(0)->get_coord())) * (normal|(t.get_vertice((i+1)%3)->get_coord() - this->get_vertice(0)->get_coord())) < 0)
	  {
	    Vec v1 = this->get_vertice(1)->get_coord() - this->get_vertice(0)->get_coord();
	    Vec v2 = this->get_vertice(2)->get_coord() - this->get_vertice(0)->get_coord();
	    Vec v3 = this->get_vertice(2)->get_coord() - this->get_vertice(1)->get_coord();
	    Vec v = v1 * v2;

	    Vec p1 = t.get_vertice(i)->get_coord() - this->get_vertice(0)->get_coord();
	    Vec d1 = t.get_vertice((i+1)%3)->get_coord() - t.get_vertice(i)->get_coord();
	    double denom = (d1.X * v.X + d1.Y * v.Y + d1.Z * v.Z);
	    if (denom != 0)
	      {
		double lambda1 = - (p1.X * v.X + p1.Y * v.Y + p1.Z * v.Z)/denom;
		Vec proj1 = p1 + (d1 * lambda1);
		//checks if proj is inside the triangle ...
		bool inside = false;
		Vec n1 = v1 * proj1;
		Vec n2 = proj1 * v2;
		Vec n3 = v3 * (proj1 + (v1 * -1));
		if (((n1 | n3) > 0 & (n2 | n3) > 0 & (n1 | n2) > 0) | ((n1 | n3) < 0 & (n2 | n3) < 0 & (n1 | n2) < 0) )
		  {
		    inside = true;
		  }
		result = result | inside;
	      }
	  }

      //test from t
      Vec normalt = (t.get_vertice(0)->get_coord() - t.get_vertice(1)->get_coord()) * (t.get_vertice(0)->get_coord() - t.get_vertice(2)->get_coord());
      for (int i = 0; i < 3; i++)
	if ((normalt|(this->get_vertice(i)->get_coord() - t.get_vertice(0)->get_coord())) * (normalt|(this->get_vertice((i+1)%3)->get_coord() - t.get_vertice(0)->get_coord())) < 0)
	  {
	    Vec v1 = t.get_vertice(1)->get_coord() - t.get_vertice(0)->get_coord();
	    Vec v2 = t.get_vertice(2)->get_coord() - t.get_vertice(0)->get_coord();
	    Vec v3 = t.get_vertice(2)->get_coord() - t.get_vertice(1)->get_coord();
	    Vec v = v1 * v2;

	    Vec p1 = this->get_vertice(i)->get_coord() - t.get_vertice(0)->get_coord();
	    Vec d1 = this->get_vertice((i+1)%3)->get_coord() - this->get_vertice(i)->get_coord();
	    double denom = (d1.X * v.X + d1.Y * v.Y + d1.Z * v.Z);
	    if (denom != 0)
	      {
		double lambda1 = - (p1.X * v.X + p1.Y * v.Y + p1.Z * v.Z)/denom;
		Vec proj1 = p1 + (d1 * lambda1);

		//checks if proj is inside the triangle ...
		bool inside = false;
		Vec n1 = v1 * proj1;
		Vec n2 = proj1 * v2;
		Vec n3 = v3 * (proj1 + (v1 * -1));
		if (((n1 | n3) > 0 & (n2 | n3) > 0 & (n1 | n2) > 0) | ((n1 | n3) < 0 & (n2 | n3) < 0 & (n1 | n2) < 0) )
		  {
		    inside = true;
		  }

		result = result | inside;
	      }
	  }




    }
  else {return (false);}
  return result;



}

  // Saad
  // algorithm from:
  // http://softsurfer.com/Archive/algorithm_0105/algorithm_0105.htm#intersect_RayTriangle()

  const bool Triangle::intersect(const vector<Pt> & p) const {
    Vec    u,v,n;   // triangle vectors
    Vec    dir,w0,w; // ray vectors
    double r, a, b;              // params to calc ray-plane intersect

    // check if point is one the vertices
    for(int ii=0;ii<=2;ii++){
      if((*_vertice[ii])==p[0])return true;
      if((*_vertice[ii])==p[1])return true;
    }

    // get triangle edge vectors and plane normal
    u = *_vertice[1]-*_vertice[0];
    v = *_vertice[2]-*_vertice[0];
    n = u*v;             // cross product
    if (n.norm()==0) // triangle is degenerate
      return false;


    dir = p[1]-p[0];             // ray direction vector
    w0 = p[0]-*_vertice[0];
    a = -(n|w0);
    b = (n|dir);
    if (fabs(b) < 0.001) { // ray is parallel to triangle plane
      if (fabs(a) < 0.001)                 // ray lies in triangle plane
	return true;
      else return false;             // ray disjoint from plane
    }

    // get intersect point of ray with triangle plane
    r = a / b;
    if (r < 0.0)                   // ray goes away from triangle
      return false;                  // => no intersect
    if(r > 1.0)
      return false;
    // for a segment, also test if (r > 1.0) => no intersect
    Pt I;
    I = p[0] + r * dir;           // intersect point of ray and plane

    // is I inside T?
    double    uu, uv, vv, wu, wv, D;
    uu = (u|u);
    uv = (u|v);
    vv = (v|v);
    w = I - *_vertice[0];
    wu = (w|u);
    wv = (w|v);
    D = uv * uv - uu * vv;

    // get and test parametric coords
    double s, t;
    s = (uv * wv - vv * wu) / D;
    if (s < 0.0 || s > 1.0)        // I is outside T
      return false;
    t = (uv * wu - uu * wv) / D;
    if (t < 0.0 || (s + t) > 1.0)  // I is outside T
      return false;

    return true;                      // I is in T

  }


  const bool Triangle::intersect(const vector<Pt> & p,int& ind) const {
    Vec    u,v,n;   // triangle vectors
    Vec    dir,w0,w; // ray vectors
    double r, a, b;              // params to calc ray-plane intersect

    // check if point is one the vertices
    for(int ii=0;ii<=2;ii++){
      if((*_vertice[ii])==p[0]){ind=ii;return true;}
      if((*_vertice[ii])==p[1]){ind=ii;return true;}
    }

    // get triangle edge vectors and plane normal
    u = *_vertice[1]-*_vertice[0];
    v = *_vertice[2]-*_vertice[0];
    n = u*v;             // cross product
    if (n.norm()==0) // triangle is degenerate
      return false;


    dir = p[1]-p[0];             // ray direction vector
    w0 = p[0]-*_vertice[0];
    a = -(n|w0);
    b = (n|dir);
    if (fabs(b) < 0.0000000001) { // ray is parallel to triangle plane
      if (fabs(a) < 0.0000000001)                 // ray lies in triangle plane
	return true;
      else return false;             // ray disjoint from plane
    }

    // get intersect point of ray with triangle plane
    r = a / b;
    if (r < 0.0)                   // ray goes away from triangle
      return false;                  // => no intersect
    if(r > 1.0)
      return false;
    // for a segment, also test if (r > 1.0) => no intersect
    Pt I;
    I = p[0] + r * dir;           // intersect point of ray and plane

    // is I inside T?
    double    uu, uv, vv, wu, wv, D;
    uu = (u|u);
    uv = (u|v);
    vv = (v|v);
    w = I - *_vertice[0];
    wu = (w|u);
    wv = (w|v);
    D = uv * uv - uu * vv;

    // get and test parametric coords
    double s, t;
    s = (uv * wv - vv * wu) / D;
    if (s < 0.0 || s > 1.0)        // I is outside T
      return false;
    t = (uv * wu - uu * wv) / D;
    if (t < 0.0 || (s + t) > 1.0)  // I is outside T
      return false;

    // which vertex is closest to where the segment intersects?
    float x=uu-2*wu,y=vv-2*wv;
    if( x<0 ){
      if( x<y ) ind=1;
      else ind=2;
    }
    else{
      if( y<0 ) ind=2;
      else ind=0;
    }

    return true;                      // I is in T

  }


}
