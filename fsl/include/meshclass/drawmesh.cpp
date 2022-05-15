/*  Copyright (C) 1999-2004 University of Oxford  */

/*  CCOPYRIGHT */

#include <iostream>
#include <string>
#include <fstream>
#include <stdio.h>
#include <cmath>
#include <algorithm>

#include "utils/options.h"
#include "newimage/newimageall.h"
#include "meshclass.h"

using namespace std;
using namespace NEWIMAGE;
using namespace Utilities;
using namespace mesh;

void noMoreMemory()
{
  cerr<<"Unable to satisfy request for memory"<<endl;
  abort();
}


vector<float> empty_vector(0, 0);



void draw_segment(volume<short>& image, const Pt& p1, const Pt& p2)
{
  double xdim = (double) image.xdim();
  double ydim = (double) image.ydim();
  double zdim = (double) image.zdim();
  double mininc = min(xdim,min(ydim,zdim)) * .5;

  Vec n = p1 - p2;
  double d = n.norm();
  n.normalize();

  for (double i=0; i<=d; i+=mininc)
    {
      Pt p = p2 + i* n;
      image((int) floor((p.X)/xdim +.5),(int) floor((p.Y)/ydim +.5),(int) floor((p.Z)/zdim +.5)) = 1;
    }
}


volume<short> draw_mesh(const volume<short>& image, const Mesh &m)
{
  double xdim = (double) image.xdim();
  double ydim = (double) image.ydim();
  double zdim = (double) image.zdim();
  double mininc = min(xdim,min(ydim,zdim)) * .5;
  volume<short> res = image;
  for (list<Triangle*>::const_iterator i = m._triangles.begin(); i!=m._triangles.end(); i++)
    {
      Vec n = (*(*i)->get_vertice(0) - *(*i)->get_vertice(1));
      double d = n.norm();
      n.normalize();

      for (double j=0; j<=d; j+=mininc)
	{
	  Pt p = (*i)->get_vertice(1)->get_coord()  + j* n;
	  draw_segment(res, p, (*i)->get_vertice(2)->get_coord());
	} 
    }
  return res;
}

volume<short> make_mask_from_mesh(const volume<short> & image, const Mesh& m)
{
  double xdim = (double) image.xdim();
  double ydim = (double) image.ydim();
  double zdim = (double) image.zdim();

  volume<short> mask = image;
  
  int xsize = mask.xsize();
  int ysize = mask.ysize();
  int zsize = mask.zsize();
  
  vector<Pt> current;
  current.clear();
  Pt c(0., 0., 0.);
  for (vector<Mpoint *>::const_iterator it=m._points.begin(); it!=m._points.end(); it++)
    c+=(*it)->get_coord();

  c*=(1./m._points.size());
  c.X/=xdim; c.Y/=ydim; c.Z/=zdim;

  current.push_back(c);

  while (!current.empty())
    {
      Pt pc = current.back();
      int x, y, z;
      x=(int) pc.X; y=(int) pc.Y; z=(int) pc.Z;
      current.pop_back();
      mask.value(x, y, z) = 1;
      if (0<=x-1 && mask.value(x-1, y, z)==0) current.push_back(Pt(x-1, y, z));
      if (0<=y-1 && mask.value(x, y-1, z)==0) current.push_back(Pt(x, y-1, z));
      if (0<=z-1 && mask.value(x, y, z-1)==0) current.push_back(Pt(x, y, z-1));
      if (xsize>x+1 && mask.value(x+1, y, z)==0) current.push_back(Pt(x+1, y, z));
      if (ysize>y+1 && mask.value(x, y+1, z)==0) current.push_back(Pt(x, y+1, z));
      if (zsize>z+1 && mask.value(x, y, z+1)==0) current.push_back(Pt(x, y, z+1)); 
    }
  return mask;
}

int main(int argc, char *argv[]) {

  
  if (argc != 3 && argc != 4) {
    cerr<<"Usage : drawmesh  [volume.hdr] [mesh.off] (-m for the mask)"<<endl;
    exit (-1);
  }
  
  bool mesh = false;
  if (argc == 4)
    {
      string ismesh=argv[3];
      if (ismesh.compare("-m") == 0)
	mesh = true;
      else 
	{
	  cerr<<"Usage : drawmesh  [volume.hdr] [mesh.off] (-m for the mask)"<<endl;
	  exit (-1);
	}
    }

  string volumename=argv[1];
  string meshname=argv[2];
  

  string out = meshname;
  if (out.find(".off")!=string::npos) out.erase(out.find(".off"), 4);

  string in = volumename;
  if (in.find(".hdr")!=string::npos) in.erase(in.find(".hdr"), 4);
  if (in.find(".img")!=string::npos) in.erase(in.find(".hdr"), 4);
  if (out == "default__default") {out=in+"_brain";}
  

  //set a memory hanlder that displays an error message
  set_new_handler(noMoreMemory);


  //the real program
  volume<short> testvol;
  volume<double> testvol2;
  
  if (read_volume(testvol2,in.c_str())<0)  return -1;

  copyconvert(testvol2, testvol);

  testvol = 0;

  Mesh m;
  m.load(meshname);  
  
  cout<<"saving volume"<<endl;
  string outlinestr = in+"_outline";
  volume<short> outline = draw_mesh(testvol, m);
  if (save_volume(outline, outlinestr.c_str())<0)  return -1;

  if (mesh) 
    {
      outline = make_mask_from_mesh(outline, m);
      if (save_volume(outline, (in+"_mask"))<0)  return -1;
    }

  return 0;
  
}





