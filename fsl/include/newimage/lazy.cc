/*  Lazy evaluation support

    Mark Jenkinson, FMRIB Image Analysis Group

    Copyright (C) 2000 University of Oxford  */

/*  CCOPYRIGHT  */

#include "lazy.h"
#include <algorithm>

#ifndef NO_NAMESPACE
 namespace LAZY {
#endif

  lazymanager::lazymanager() 
    {
      tagnum = 1;
      validflag = false;  
      validcache.erase(validcache.begin(),validcache.end()); 
    }


  void lazymanager::copylazymanager(const lazymanager &source)
    {
      validflag = source.validflag;
      tagnum = source.tagnum;
      validcache = source.validcache;
    }

  void lazymanager::invalidate_whole_cache() const 
    {	
      for (mapiterator p=validcache.begin(); p!=validcache.end(); ++p) 
	{
	  p->second = false;
	}
    }

#ifndef NO_NAMESPACE
 }
#endif












