/*  Lazy evaluation support

    Mark Jenkinson, FMRIB Image Analysis Group

    Copyright (C) 2000 University of Oxford  */

/*  CCOPYRIGHT  */

#if !defined(__lazy_h)
#define __lazy_h

#include <iostream>
#include <map>
#include <cstdlib>

namespace LAZY {


typedef std::map<unsigned int,bool,std::less<unsigned int> > mapclass;
typedef std::map<unsigned int,bool,std::less<unsigned int> >::iterator mapiterator;


//-------------------------------------------------------------------------//

template <class T, class S>
class lazy {
private:
  mutable T storedval;
  unsigned int tag;
  const S *iptr;
  T (*calc_fn)(const S &);

private:
  const T& value() const;
  T calculate_val() const { return (*calc_fn)(*iptr); }

public:
  lazy() { tag = 0; }
  void init(const S *, T (*fnptr)(const S &));
  void copy(const lazy &, const S *);

  const T& force_recalculation() const;

  const T& operator() () const { return this->value(); }
};


//-------------------------------------------------------------------------//

class lazymanager {
  template <class T, class S>  friend class lazy;
private:
  mutable bool validflag;
  mutable mapclass validcache;
  mutable unsigned int tagnum;

private:
  unsigned int getnewtag() const { return tagnum++; }

  bool is_whole_cache_valid() const
    { return validflag; }

  bool is_cache_entry_valid(const unsigned int tag) const
    { return validcache[tag]; }
  void set_cache_entry_validity(const unsigned int tag, const bool newflag) const
    { validcache[tag] = newflag; }

  void invalidate_whole_cache() const;

public:
  lazymanager();
  void copylazymanager(const lazymanager &);
  void set_whole_cache_validity(const bool newflag) const
    { validflag = newflag; }
};


//-------------------------------------------------------------------------//

// Body of lazy member functions (put here as cannot simply separate
//   templated definitions into seperate source files if building a library)



template <class T, class S>
const T& lazy<T,S>::value() const
  {
    if ( (iptr == 0) || (tag==0) ) {
      std::cerr << "Error: uninitialized lazy evaluation class" << std::endl;
      exit(-1);
    }
    if (! iptr->is_whole_cache_valid() ) {
      iptr->invalidate_whole_cache();
      iptr->set_whole_cache_validity(true);
    }
    if (! iptr->is_cache_entry_valid(tag)) {
      //cerr << "Calculating value" << endl;
      storedval = calculate_val();
      iptr->set_cache_entry_validity(tag,true);
    }
    return storedval;
  }


template <class T, class S>
const T& lazy<T,S>::force_recalculation() const
  {
    if ( (iptr == 0) || (tag==0) ) {
      std::cerr << "Error: uninitialized lazy evaluation class" << std::endl;
      exit(-1);
    }
    // still process the whole cache vailidity so that this calculation
    //  can get cached correctly
    if (! iptr->is_whole_cache_valid() ) {
      iptr->invalidate_whole_cache();
      iptr->set_whole_cache_validity(true);
    }

    storedval = calculate_val();
    iptr->set_cache_entry_validity(tag,true);

    return storedval;
  }


template <class T, class S>
void lazy<T,S>::init(const S *ip, T (*fnptr)(const S &))
  {
    iptr = ip;
    calc_fn = fnptr;
    tag = iptr->getnewtag();
    iptr->set_cache_entry_validity(tag,false);
  }


template <class T, class S>
void lazy<T,S>::copy(const lazy &source, const S *ip) {
  storedval = source.storedval;
  tag = source.tag;
  calc_fn = source.calc_fn;
  // Do NOT copy the same parent class pointer
  //   (allows parent class to be copied correctly)
  iptr = ip;
}

 }

#endif
