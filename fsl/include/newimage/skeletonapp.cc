/*  skeletonapp.cc

    Mark Jenkinson, FMRIB Image Analysis Group

    Copyright (C) 2003 University of Oxford  */

/*  CCOPYRIGHT  */

// Skeleton application framework for using newimage


#include "newimage/newimageall.h"
#include "miscmaths/miscmaths.h"
#include "utils/options.h"
#include <stdlib.h>
#include <time.h>

using namespace MISCMATHS;
using namespace NEWIMAGE;
using namespace Utilities;

// The two strings below specify the title and example usage that is
//  printed out as the help or usage message

string title="skeletonapp (Version 1.0)\nCopyright(c) 2003, University of Oxford (Mark Jenkinson)";
string examples="skeletonapp [options] --in1=<image1> --in2=<image2>";

// Each (global) object below specificies as option and can be accessed
//  anywhere in this file (since they are global).  The order of the
//  arguments needed is: name(s) of option, default value, help message,
//       whether it is compulsory, whether it requires arguments
// Note that they must also be included in the main() function or they
//  will not be active.

Option<bool> verbose(string("-v,--verbose"), false,
		     string("switch on diagnostic messages"),
		     false, no_argument);
Option<bool> help(string("-h,--help"), false,
		  string("display this message"),
		  false, no_argument);
Option<string> inname(string("-i"), string(""),
		  string("input filename"),
		  true, requires_argument);
Option<string> outname(string("-o"), string(""),
		  string("output filename"),
		  true, requires_argument);
int nonoptarg;

////////////////////////////////////////////////////////////////////////////

// Local functions
void food(ShadowVolume<float>& finput) {
  volume<float> v2(1,2,3);
  ShadowVolume<float> foo=v2[0];
  cerr << finput.zsize() << endl;
  finput=v2[0];
  cerr << finput.zsize() << endl;
}
// for example ... print difference of COGs between 2 images ...
int do_work(int argc, char* argv[])
{
  volume4D<float> v1(1,2,3,4);
  v1=1;
  ShadowVolume<float> foo(v1[2]);
  foo=0.0;
  food(foo);
  ShadowVolume<float> foo2(foo);
  foo=foo2;
  v1=foo;
  return 0;
}

////////////////////////////////////////////////////////////////////////////

int main(int argc,char *argv[])
{

  Tracer tr("main");
  OptionParser options(title, examples);

  try {
    // must include all wanted options here (the order determines how
    //  the help message is printed)
    options.add(inname);
    options.add(outname);
    options.add(verbose);
    options.add(help);

    nonoptarg = options.parse_command_line(argc, argv);

    // line below stops the program if the help was requested or
    //  a compulsory option was not set
    if ( (help.value()) || (!options.check_compulsory_arguments(true)) )
      {
	options.usage();
	exit(EXIT_FAILURE);
      }

  }  catch(X_OptionError& e) {
    options.usage();
    cerr << endl << e.what() << endl;
    exit(EXIT_FAILURE);
  } catch(std::exception &e) {
    cerr << e.what() << endl;
  }

  // Call the local functions

  return do_work(argc,argv);
}
