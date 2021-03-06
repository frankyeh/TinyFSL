/*  Tracer_Plus.h

    Mark Woolrich, FMRIB Image Analysis Group

    Copyright (C) 1999-2000 University of Oxford  */

/*  CCOPYRIGHT  */

#if !defined(Tracer_Plus_h)
#define Tracer_Plus_h

#include <iostream>
#include <fstream>
#include <string>
#include <time.h>
#include <set>

#include "armawrap/newmatap.h"
#include "armawrap/newmatio.h"
#include "time_tracer.h"

namespace Utilities {

  // Newmat version:
  class Tracer_Plus : public RBD_COMMON::Tracer, public Time_Tracer
    {
    public:
      Tracer_Plus(const char* str) :
	RBD_COMMON::Tracer(const_cast<char*>(str)),
	Time_Tracer(str)
	{
	}

      Tracer_Plus(char* str) :
	RBD_COMMON::Tracer(str),
	Time_Tracer(str)
	{
	}

      virtual ~Tracer_Plus()
	{
	}

    private:
      Tracer_Plus();
      const Tracer_Plus& operator=(Tracer_Plus&);
      Tracer_Plus(Tracer_Plus&);
    };

}

#endif
