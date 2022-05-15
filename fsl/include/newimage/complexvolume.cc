/*  Copyright (C) 2000 University of Oxford  */

/*  CCOPYRIGHT  */

#include "newimagefns.h"
#include "complexvolume.h"
#include <cstdlib>

namespace NEWIMAGE {

// COMPLEX REF
const complexpoint& complexref::operator=(const complexpoint& val){
  *m_real = val.re();
  *m_imag = val.im();
  return(val);
}
  
// COMPLEX POINT
float complexpoint::operator=(const float val)
{
  m_real=val;
  m_imag=0;
  return(val);
}
complexpoint& complexpoint::operator=(const complexpoint& source)
{
  m_real=source.re();
  m_imag=source.im();
  return *this;
}
complexpoint& complexpoint::operator=(const complexref& source){
  m_real = *(source.re_pointer());
  m_imag = *(source.im_pointer());
  return *this;
}
const complexpoint& complexpoint::operator+=(const complexpoint& val)
{
  m_real+=val.re();
  m_imag+=val.im();
  return *this;
}
const complexpoint& complexpoint::operator-=(const complexpoint& val)
{
  m_real-=val.re();
  m_imag-=val.im();
  return *this;
}
const complexpoint& complexpoint::operator*=(const complexpoint& val)
{
  float r2 = (m_real*val.re()) - (m_imag*val.im());
  float i2 = (m_real*val.im()) + (m_imag*val.re());
  m_real = r2;
  m_imag = i2;
  return *this;
}
const complexpoint& complexpoint::operator/=(const complexpoint& val)
{
  float r2 = ((m_real*val.re())+(m_imag*val.im()))/((val.re()*val.re())+(val.im()*val.im()));
  float i2 = ((m_imag*val.re())-(m_real*val.im()))/((val.re()*val.re())+(val.im()*val.im()));
  m_real = r2;
  m_imag = i2;
  return *this;
}
complexpoint complexpoint::operator+(const complexpoint& val) const
{
  complexpoint tmp = *this;
  tmp += val;
  return(tmp);
}
complexpoint complexpoint::operator-(const complexpoint& val) const
{
  complexpoint tmp = *this;
  tmp -= val;
  return(tmp);
}
complexpoint complexpoint::operator*(const complexpoint& val) const
{
  complexpoint tmp = *this;
  tmp *= val;
  return(tmp);
}
complexpoint complexpoint::operator/(const complexpoint& val) const
{
  complexpoint tmp = *this;
  tmp /= val;
  return(tmp);
}
float complexpoint::abs(void) const
{
  return(sqrt((m_real)*(m_real)+(m_imag)*(m_imag)));
}
float complexpoint::phase(void) const
{
  return(atan2(m_imag,m_real));
}

//ostream& complexpoint::operator<<(ostream& s, const complexpoint& val)
//{
//  if(im()>=0.0){
//    return s << re() << " + " << im() << "i";
//  } else {
//    return s << re() << " - " << fabs(im()) << "i";
//  }
//}

// COMPLEX VOLUME
complexvolume::complexvolume(int xsize, int ysize, int zsize)
{
  volume<float> dummy(xsize,ysize,zsize);
  dummy=0.0;
  real=dummy;
  imag=dummy;
}

complexvolume::complexvolume(const complexvolume& source)
{
  real=source.real;
  imag=source.imag;
}
complexvolume::complexvolume(const volume<float>& r, const volume<float>& i)
{
  real=r;
  imag=i;
  if(!samesize(r,i))
    imthrow("Attempted to create complex volume with non-matching sizes",2);
}
complexvolume::complexvolume(const volume<float>& r)
{
  real=r;
  imag=0;
}
complexvolume::~complexvolume()
{
  this->destroy();
}
void complexvolume::destroy()
{
  real.destroy();
  imag.destroy();
}
float complexvolume::operator=(const float val)
{
  real=val;
  imag=0;
  return(val);
}
complexvolume& complexvolume::operator=(const complexvolume& source)
{
  real=source.real;
  imag=source.imag;
  return *this;
}
int complexvolume::copyproperties(const complexvolume& source)
{
  real.copyproperties(source.real);
  imag.copyproperties(source.real);
  return 0;
}
int complexvolume::copydata(const complexvolume& source)
{
  real.copydata(source.real);
  imag.copydata(source.real);
  return 0;
}
const complexvolume& complexvolume::operator+=(const complexpoint& val)
{
  real+=val.re();
  imag+=val.im();
  return *this;
}
const complexvolume& complexvolume::operator-=(const complexpoint& val)
{
  real-=val.re();
  imag-=val.im();
  return *this;
}
const complexvolume& complexvolume::operator*=(const complexpoint& val)
{
  volume<float> r2 = (real*val.re())-(imag*val.im());
  volume<float> i2 = (real*val.im())+(imag*val.re());
  real=r2;
  imag=i2;
  return *this;
}
const complexvolume& complexvolume::operator/=(const complexpoint& val)
{
  volume<float> r2 = ((real*val.re())+(imag*val.im()))/((val.re()*val.re())+(val.im()*val.im()));
  volume<float> i2 = ((imag*val.re())-(real*val.im()))/((val.re()*val.re())+(val.im()*val.im()));
  real = r2;
  imag = i2;
  return *this;
}
const complexvolume& complexvolume::operator+=(const complexvolume& val)
{
  real+=val.real;
  imag+=val.imag;
  return *this;
}
const complexvolume& complexvolume::operator-=(const complexvolume& val)
{
  real-=val.real;
  imag-=val.imag;
  return *this;
}
const complexvolume& complexvolume::operator*=(const complexvolume& val)
{
  volume<float> r2 = (real*val.real)-(imag*val.imag);
  volume<float> i2 = (real*val.imag)+(imag*val.real);
  real=r2;
  imag=i2;
  return *this;
}
const complexvolume& complexvolume::operator/=(const complexvolume& val)
{
  volume<float> r2 = ((real*val.real)+(imag*val.imag))/((val.real*val.real)+(val.imag*val.imag));
  volume<float> i2 = ((imag*val.real)-(real*val.imag))/((val.real*val.real)+(val.imag*val.imag));
  real = r2;
  imag = i2;
  return *this;
}
complexvolume complexvolume::operator+(const complexpoint& val) const
{
  complexvolume tmp=*this;
  tmp += val;
  return(tmp);
}
complexvolume complexvolume::operator-(const complexpoint& val) const
{
  complexvolume tmp=*this;
  tmp -= val;
  return(tmp);
}
complexvolume complexvolume::operator*(const complexpoint& val) const
{
  complexvolume tmp=*this;
  tmp *= val;
  return(tmp);
}
complexvolume complexvolume::operator/(const complexpoint& val) const
{
  complexvolume tmp=*this;
  tmp /= val;
  return(tmp);
}
complexvolume complexvolume::operator+(const complexvolume& val) const
{
  complexvolume tmp=*this;
  tmp += val;
  return(tmp);
}
complexvolume complexvolume::operator-(const complexvolume& val) const
{
  complexvolume tmp=*this;
  tmp -= val;
  return(tmp);
}
complexvolume complexvolume::operator*(const complexvolume& val) const
{
  complexvolume tmp=*this;
  tmp *= val;
  return(tmp);
}
complexvolume complexvolume::operator/(const complexvolume& val) const
{
  complexvolume tmp=*this;
  tmp /= val;
  return(tmp);
}

volume<float> complexvolume::abs(void) const
{
  return(NEWIMAGE::abs(real,imag));
}

volume<float> complexvolume::phase(void) const
{
  return(NEWIMAGE::phase(real,imag));
}

volume<float>& complexvolume::re(void)
{
  return(real);
}

volume<float>& complexvolume::im(void)
{
  return(imag);
}

const volume<float>& complexvolume::re(void) const
{
  return(real);
}

const volume<float>& complexvolume::im(void) const
{
  return(imag);
}

complexvolume complexvolume::extract_slice(int slice) const
{
  volume<float> tempr(real.xsize(),real.ysize(),1);
  volume<float> tempi(real.xsize(),real.ysize(),1);
  
  for(int x=0;x<real.xsize();x++){
    for(int y=0;y<real.ysize();y++){
      tempr(x,y,0) = real(x,y,slice);
      tempi(x,y,0) = imag(x,y,slice);
    }
  }
  complexvolume out(tempr,tempi);
  return(out);
}
void complexvolume::overwrite_slice(const complexvolume& data,int slice)
{
  for(int x=0;x<real.xsize();x++){
    for(int y=0;y<real.ysize();y++){
      real(x,y,slice) = data.re(x,y,0);
      imag(x,y,slice) = data.im(x,y,0);
    }
  }
}

}
