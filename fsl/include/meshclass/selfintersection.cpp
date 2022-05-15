/*  Copyright (C) 1999-2004 University of Oxford  */

/*  CCOPYRIGHT */

#include <iostream>
#include <string>
#include <fstream>
#include <stdio.h>
#include <cmath>
#include <algorithm>

#include "meshclass.h"

using namespace std;
using namespace mesh;

void noMoreMemory()
{
  cerr<<"Unable to satisfy request for memory"<<endl;
  abort();
}


int main(int argc, char *argv[]) {
  
  if (argc != 2)
    {
      cerr<<"Usage : selfintersection [mesh.off]"<<endl;
      exit (-1);
    }
  string s = argv[1];
  Mesh m;
  m.load(s);
  if (!m.real_self_intersection()) cout<<"lucky you, the mesh is NOT self-intersecting"<<endl; else cout<<"oups, the mesh IS self-intersecting"<<endl; 
  return 0;
}





