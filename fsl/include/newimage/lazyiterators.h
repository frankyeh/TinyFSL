/*  Templated iterators for any storage class that uses lazymanager

    Mark Jenkinson, FMRIB Image Analysis Group

    Copyright (C) 2000 University of Oxford  */

/*  CCOPYRIGHT  */

#if !defined(__lazyiterators_h)
#define __lazyiterators_h

#include <cstdlib>
#include <iostream>
#include "lazy.h"

namespace LAZY {

  // Mutable, Random Access Iterator

  template <class IT, class T>
  class rlazyiterator {
  private:
    IT iter;
    lazymanager* lazyptr;
  public:
    rlazyiterator() : lazyptr(0) { }
    rlazyiterator(const rlazyiterator<IT,T>& source) :
      iter(source.iter) , lazyptr(source.lazyptr) { }
    rlazyiterator(const IT& sourceiter, lazymanager* lazyp) :
      iter(sourceiter) , lazyptr(lazyp) { }
    ~rlazyiterator() { }  // do nothing

    inline const rlazyiterator<IT,T> operator++(int)
      { rlazyiterator<IT,T> tmp=*this; iter++; return tmp; }
    inline const rlazyiterator<IT,T>& operator++() // prefix
      { ++iter; return *this; }
    inline const rlazyiterator<IT,T> operator--(int)
      { rlazyiterator<IT,T> tmp=*this; iter--; return tmp; }
    inline const rlazyiterator<IT,T>& operator--() // prefix
      { --iter; return *this; }

    inline const rlazyiterator<IT,T>& operator+=(int n)
      { iter+=n; return *this; }
    inline const rlazyiterator<IT,T>& operator-=(int n)
      { iter-=n; return *this; }

//      template <class ITF, class TF> friend const rlazyiterator<ITF,TF>
//      operator+(const rlazyiterator<ITF,TF>& it, int n)
//        { return rlazyiterator<ITF,TF>(it.iter + n,it.lazyptr); }
//      template <class ITF, class TF> friend const rlazyiterator<ITF,TF>
//      operator+(int n, const rlazyiterator<ITF,TF>& it)
//        { return rlazyiterator<ITF,TF>(n + it.iter,it.lazyptr); }
//      template <class ITF, class TF> friend const rlazyiterator<ITF,TF>
//      operator-(const rlazyiterator<ITF,TF>& it, int n)
//        { return rlazyiterator<ITF,TF>(it.iter - n,it.lazyptr); }
//      template <class ITF, class TF> friend const rlazyiterator<ITF,TF>
//      operator-(int n, const rlazyiterator<ITF,TF>& it)
//        { return rlazyiterator<ITF,TF>(n - it.iter,it.lazyptr); }


    inline bool operator==(const rlazyiterator<IT,T>& it) const
       { return iter == it.iter; }
    inline bool operator!=(const rlazyiterator<IT,T>& it) const
       { return iter != it.iter; }
    inline bool operator<(const rlazyiterator<IT,T>& it) const
       { return iter < it.iter; }
    inline bool operator>(const rlazyiterator<IT,T>& it) const
       { return iter > it.iter; }
    inline bool operator<=(const rlazyiterator<IT,T>& it) const
       { return iter <= it.iter; }
    inline bool operator>=(const rlazyiterator<IT,T>& it) const
       { return iter >= it.iter; }

    inline const rlazyiterator<IT,T>&
      operator=(const rlazyiterator<IT,T>& source)
      { iter=source.iter; lazyptr = source.lazyptr; return *this; }

    inline T& operator*() const
        { lazyptr->set_whole_cache_validity(false); return *iter;}
    inline T& operator[](int n) const { return *(this + n); }
  };


  //---------------------------------------------------------------------//

  // Constant, Random Access Iterator

  // Use normal constant iterator


  //---------------------------------------------------------------------//

  // Mutable, Bidirectional Iterator

  template <class IT, class T>
  class bilazyiterator {
   private:
    IT iter;
    lazymanager* lazyptr;
  public:
    bilazyiterator() : lazyptr(0) { }
    bilazyiterator(const bilazyiterator<IT,T>& source) :
      iter(source.iter) , lazyptr(source.lazyptr) { }
    bilazyiterator(const IT& sourceiter, lazymanager* lazyp) :
      iter(sourceiter) , lazyptr(lazyp) { }
    ~bilazyiterator() { }  // do nothing

    inline const bilazyiterator<IT,T> operator++(int)
      { bilazyiterator<IT,T> tmp=*this; iter++; return tmp; }
    inline const bilazyiterator<IT,T>& operator++() // prefix
      { ++iter; return *this; }
    inline const bilazyiterator<IT,T> operator--(int)
      { bilazyiterator<IT,T> tmp=*this; iter--; return tmp; }
    inline const bilazyiterator<IT,T>& operator--() // prefix
      { --iter; return *this; }

    inline bool operator==(const bilazyiterator<IT,T>& it) const
       { return iter == it.iter; }
    inline bool operator!=(const bilazyiterator<IT,T>& it) const
       { return iter != it.iter; }

    inline const bilazyiterator<IT,T>&
      operator=(const bilazyiterator<IT,T>& source)
      { iter=source.iter; lazyptr = source.lazyptr; return *this; }

    inline T& operator*() const
        { lazyptr->set_whole_cache_validity(false); return *iter;}
  };


  //---------------------------------------------------------------------//

  // Constant, Bidirectional Iterator

  // Use normal constant iterator

  //---------------------------------------------------------------------//

  // Mutable, Forward Iterator

  template <class IT, class T>
  class flazyiterator {
  private:
    IT iter;
    lazymanager* lazyptr;
  public:
    flazyiterator() : lazyptr(0) { }
    flazyiterator(const flazyiterator<IT,T>& source) :
      iter(source.iter) , lazyptr(source.lazyptr) { }
    flazyiterator(const IT& sourceiter, lazymanager* lazyp) :
      iter(sourceiter) , lazyptr(lazyp) { }
    ~flazyiterator() { }  // do nothing

    inline const flazyiterator<IT,T> operator++(int)
      { flazyiterator<IT,T> tmp=*this; iter++; return tmp; }
    inline const flazyiterator<IT,T>& operator++() // prefix
      { ++iter; return *this; }

    inline bool operator==(const flazyiterator<IT,T>& it) const
       { return iter == it.iter; }
    inline bool operator!=(const flazyiterator<IT,T>& it) const
       { return iter != it.iter; }

    inline const flazyiterator<IT,T>&
      operator=(const flazyiterator<IT,T>& source)
      { iter=source.iter; lazyptr = source.lazyptr; return *this; }

    inline T& operator*() const
        { lazyptr->set_whole_cache_validity(false); return *iter;}
  };


  //---------------------------------------------------------------------//

  // Constant, Forward Iterator

  // Use normal constant iterator

  //---------------------------------------------------------------------//

}  // end namespace

#endif
