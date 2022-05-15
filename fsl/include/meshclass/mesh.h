/*  Copyright (C) 1999-2004 University of Oxford  */

/*  CCOPYRIGHT */

#ifndef _mesh
#define _mesh

#include <list>
#include <iostream>
#include <string>
#include <fstream>
#include <stdio.h>
#include <algorithm>

#include "point.h"
#include "newimage/newimageall.h"

namespace mesh{

class Mpoint;
class Triangle;

class Mesh {
 public:
  std::vector<Mpoint *> _points;
  std::list<Triangle *> _triangles;
  std::vector<Triangle*> loc_triangles; // SJ: for index access to triangles

  Mesh();
  ~Mesh();
  Mesh(const Mesh&m);
  Mesh operator=(const Mesh&m);

  void display() const;
  void clear();                 //clear the mesh and delete its components
  const int nvertices() const;
  Mpoint * get_point(int n){return _points[n];};
  Triangle * get_triangle(int n)const{
    return loc_triangles[n];
  }

  void init_loc_triangles();
  double distance(const Pt& p) const; //signed distance of the point to the mesh
  void reorientate();     //puts the triangles in a coherent orientation
  void addvertex(Triangle *const t,const Pt p);
  void retessellate(); //global retesselation
  void update();       //puts _update_coord into _coords for each point
  void translation(const double x,const double y,const double z);
  void translation(const Vec v);
  void rotation(const double r11, const double r12, const double r13,const double r21, const double r22, const double r23,const double r31, const double r32, const double r33, const double x, const double y, const double z);

  void rescale(const double t, const double x=0,const double y=0,const double z=0);
  void rescale(const double t , const Pt p);
  void rescale(const double tx, const double ty, const double tz, const Pt p);

  int load(std::string s="manual_input"); //  returns: -1 if load fails, 0 if load is cancelled
  //  1 if load succeeds and file is a .off file, 2 if load succeeds and file is a freesurfer file
  void load_off(std::string s="manual_input");
  void load_vtk_ASCII(std::string s="manual_input");
  void load_fs(std::string s="manual_input");
  void load_fs_label(std::string s="manual_input",const int& value=1);
  void save(std::string s="manual_input",int type=1) const;
  void save_fs_label(std::string s,bool saveall=false) const;//save an fs label of all points with non-zero value
  void save_fs(std::string s) const;//save whole surface with values


  const double self_intersection(const Mesh& original) const;
  const bool real_self_intersection();
  void stream_mesh(std::ostream& flot, int type=1) const; //type=1 -> .off style stream. type=2 -> freesurfer style stream
};

void make_mesh_from_tetra(int, Mesh&);
void make_mesh_from_icosa(int, Mesh&);
void make_mesh_from_octa(int, Mesh&);
std::ostream& operator <<(std::ostream& flot,const Mesh & m);

}

#endif
