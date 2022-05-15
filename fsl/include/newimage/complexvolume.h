/*  Copyright (C) 2000 University of Oxford  */

/*  CCOPYRIGHT  */

#if !defined(__complexvolume_h)
#define __complexvolume_h

#include "newimage.h" 


namespace NEWIMAGE {

  class complexpoint;

  class complexref {
  private:
    float *m_real;
    float *m_imag;
  public:
    complexref(float* r, float* i) : m_real(r), m_imag(i){}
    ~complexref(){}
    inline float* re_pointer() const { return m_real; }
    inline float* im_pointer() const { return m_imag; }
    const complexpoint& operator=(const complexpoint& val);
  };

  class complexpoint {

  private:
    float m_real;
    float m_imag;

  public:

    complexpoint(){}
    complexpoint(float r, float i){m_real=r; m_imag=i;}
    complexpoint(float r){m_real=r; m_imag=0;}
    complexpoint(const complexref& source){
      m_real = *(source.re_pointer());
      m_imag = *(source.im_pointer());
    }
    ~complexpoint(){}
    float operator=(const float val);
    complexpoint& operator=(const complexpoint& source);
    complexpoint& operator=(const complexref& source);
    inline float re() const { return m_real; }
    inline float im() const { return m_imag; }

    const complexpoint& operator+=(const complexpoint& val); 
    const complexpoint& operator-=(const complexpoint& val); 
    const complexpoint& operator*=(const complexpoint& val); 
    const complexpoint& operator/=(const complexpoint& val);
    complexpoint operator+(const complexpoint& val) const;
    complexpoint operator-(const complexpoint& val) const;
    complexpoint operator*(const complexpoint& val) const;
    complexpoint operator/(const complexpoint& val) const;

    //ostream& operator<<(ostream& s, const complexpoint& val);

    float abs() const;
    float phase() const;
  };

  class complexvolume {
    
  private:
    volume<float> real;
    volume<float> imag;

  public:
    complexvolume(){}
    complexvolume(int xsize, int ysize, int zsize);
    complexvolume(const complexvolume& source);
    complexvolume(const volume<float>& r, const volume<float>& i);
    complexvolume(const volume<float>& r);
    ~complexvolume();
    float operator=(const float val);
    complexvolume& operator=(const complexvolume& source); 
    void destroy();
    int copyproperties(const complexvolume& source);
    int copydata(const complexvolume& source);
 
    const float& re(int x,int y, int z) const { return real(x,y,z); }
    const float& im(int x,int y, int z) const { return imag(x,y,z); }
    float& re(int x,int y, int z) { return real(x,y,z); }
    float& im(int x,int y, int z) { return imag(x,y,z); }

    inline int xsize() const { return real.xsize(); }
    inline int ysize() const { return real.ysize(); }
    inline int zsize() const { return real.zsize(); }
    inline float xdim() const { return real.xdim(); }
    inline float ydim() const { return real.ydim(); }
    inline float zdim() const { return real.zdim(); }
    void setxdim(float x) { real.setxdim(x); imag.setxdim(x); }
    void setydim(float y) { real.setydim(y); imag.setydim(y); }
    void setzdim(float z) { real.setzdim(z); imag.setzdim(z); }
    void setdims(float x, float y, float z){ setxdim(x); setydim(y); setzdim(z); }
    int nvoxels() const { return real.nvoxels(); }
  
    const complexvolume& operator+=(const complexpoint& val);
    const complexvolume& operator-=(const complexpoint& val);
    const complexvolume& operator*=(const complexpoint& val);
    const complexvolume& operator/=(const complexpoint& val);
    const complexvolume& operator+=(const complexvolume& source); 
    const complexvolume& operator-=(const complexvolume& source); 
    const complexvolume& operator*=(const complexvolume& source); 
    const complexvolume& operator/=(const complexvolume& source); 
    complexvolume operator+(const complexpoint& val) const;
    complexvolume operator-(const complexpoint& val) const;
    complexvolume operator*(const complexpoint& val) const;
    complexvolume operator/(const complexpoint& val) const;
    complexvolume operator+(const complexvolume& vol) const;
    complexvolume operator-(const complexvolume& vol) const;
    complexvolume operator*(const complexvolume& vol) const;
    complexvolume operator/(const complexvolume& vol) const;

    volume<float> abs() const;
    volume<float> phase() const;
    volume<float>& re();
    volume<float>& im();
    const volume<float>& re() const;
    const volume<float>& im() const;

    complexref operator()(int x,int y, int z)
      { return(complexref(&real(x,y,z),&imag(x,y,z))); }

    
    complexvolume extract_slice(int slice) const;
    void overwrite_slice(const complexvolume& data,int slice);

  };
}
#endif
