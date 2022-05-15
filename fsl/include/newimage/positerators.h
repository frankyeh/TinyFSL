/*  Templated iterators for the image storage class (which uses lazymanager)

    Mark Jenkinson, FMRIB Image Analysis Group

    Copyright (C) 2000 University of Oxford  */

/*  CCOPYRIGHT  */

#if !defined(__positerators_h)
#define __positerators_h

#include <cstdlib>
#include "lazy.h"
#include <iostream>

namespace NEWIMAGE {

 //---------------------------------------------------------------------//

  // Mutable, Forward Iterator

  template <class T>
  class poslazyiterator {
  private:
    T* iter;
    LAZY::lazymanager* lazyptr;
    int x;
    int y;
    int z;
    int lx0;
    int ly0;
    int lz0;
    int lx1;
    int ly1;
    int lz1;
    int RowAdjust;
    int SliceAdjust;

    void calc_offsets(int rowoff, int sliceoff) {
      RowAdjust = rowoff + (lx0 - lx1);
      SliceAdjust = sliceoff + rowoff*(ly0 - ly1) + (lx0 - lx1);
    }

  public:
    poslazyiterator() : lazyptr(0) { }
    poslazyiterator(const poslazyiterator<T>& source)
      { this->operator=(source); }
    poslazyiterator(T* sourceptr, LAZY::lazymanager* lazyp,
		    int xinit, int yinit, int zinit,
		    int x0, int y0, int z0, int x1, int y1, int z1,
		    int rowoff, int sliceoff) :
      iter(sourceptr), lazyptr(lazyp), x(xinit), y(yinit), z(zinit),
      lx0(x0), ly0(y0), lz0(z0), lx1(x1), ly1(y1), lz1(z1)
      { calc_offsets(rowoff,sliceoff); }
    ~poslazyiterator() { }  // do nothing

    inline const poslazyiterator<T> operator++(int)
      { poslazyiterator<T> tmp=*this; ++(*this); return tmp; }
    inline const poslazyiterator<T>& operator++() // prefix
      { x++; if (x>lx1) { x=lx0; y++;
               if (y>ly1) { y=ly0; z++; if (z>lz1) { ++iter; } // end condition
                   else { iter+=SliceAdjust; } }
               else { iter+=RowAdjust; } }
             else { ++iter;}   return *this; }

    inline bool operator==(const poslazyiterator<T>& it) const
       { return iter == it.iter; }
    inline bool operator!=(const poslazyiterator<T>& it) const
       { return iter != it.iter; }

    inline void getposition(int &rx, int &ry, int &rz) const
      { rx= x; ry = y; rz = z; }
    inline const int& getx() const { return x; }
    inline const int& gety() const { return y; }
    inline const int& getz() const { return z; }

    inline const poslazyiterator<T>&
      operator=(const poslazyiterator<T>& source)
      { iter = source.iter; lazyptr = source.lazyptr;
        lx0=source.lx0; ly0=source.ly0; lz0=source.lz0;
	lx1=source.lx1; ly1=source.ly1; lz1=source.lz1;
	x=source.x; y=source.y; z=source.z;
	RowAdjust = source.RowAdjust;  SliceAdjust = source.SliceAdjust;
	return *this; }

    inline T& operator*() const
      { lazyptr->set_whole_cache_validity(false); return *iter;}
  };


  //---------------------------------------------------------------------//

  // Constant, Forward Iterator

  template <class T>
  class posconstiterator {
  private:
    T* iter;
    int x;
    int y;
    int z;
    int lx0;
    int ly0;
    int lz0;
    int lx1;
    int ly1;
    int lz1;
    int RowAdjust;
    int SliceAdjust;

    void calc_offsets(int rowoff, int sliceoff) {
      RowAdjust = rowoff + (lx0 - lx1);
      SliceAdjust = sliceoff + rowoff*(ly0 - ly1) + (lx0 - lx1);
    }

  public:
    posconstiterator() { }
    posconstiterator(const posconstiterator<T>& source)
      { this->operator=(source); }
    posconstiterator(T* sourceptr,
		    int xinit, int yinit, int zinit,
		    int x0, int y0, int z0, int x1, int y1, int z1,
		    int rowoff, int sliceoff) :
      iter(sourceptr), x(xinit), y(yinit), z(zinit),
      lx0(x0), ly0(y0), lz0(z0), lx1(x1), ly1(y1), lz1(z1)
      { calc_offsets(rowoff,sliceoff); }
    ~posconstiterator() { }  // do nothing

    inline const posconstiterator<T> operator++(int)
      { posconstiterator<T> tmp=*this; ++(*this); return tmp; }
    inline const posconstiterator<T>& operator++() // prefix
      { x++; if (x>lx1) { x=lx0; y++;
               if (y>ly1) { y=ly0; z++; if (z>lz1) { ++iter; } // end condition
                   else { iter+=SliceAdjust; } }
               else { iter+=RowAdjust; } }
             else { ++iter;}   return *this; }

    inline bool operator==(const posconstiterator<T>& it) const
       { return iter == it.iter; }
    inline bool operator!=(const posconstiterator<T>& it) const
       { return iter != it.iter; }

    inline void getposition(int &rx, int &ry, int &rz) const
      { rx= x; ry = y; rz = z; }
    inline const int& getx() const { return x; }
    inline const int& gety() const { return y; }
    inline const int& getz() const { return z; }

    inline const posconstiterator<T>&
      operator=(const posconstiterator<T>& source)
      { iter = source.iter;
        lx0=source.lx0; ly0=source.ly0; lz0=source.lz0;
	lx1=source.lx1; ly1=source.ly1; lz1=source.lz1;
	x=source.x; y=source.y; z=source.z;
	RowAdjust = source.RowAdjust;  SliceAdjust = source.SliceAdjust;
	return *this; }

    inline const T& operator*() const { return *iter;}
  };



  //---------------------------------------------------------------------//

}  // end namespace

#endif
