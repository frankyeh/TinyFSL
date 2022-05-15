/*  Matthew Webster, FMRIB Image Analysis Group
    Copyright (C) 2007-2010 University of Oxford  */

/*  CCOPYRIGHT */

#if !defined(BUILDNO_H)
#define BUILDNO_H

#include <string>
#define XSTR(s) STR(s)
#define STR(s) #s
namespace Utilities {

  std::string build(XSTR(BUILDSTRING));

}

#endif
