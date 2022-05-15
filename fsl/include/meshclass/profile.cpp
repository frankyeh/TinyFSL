/*  Copyright (C) 1999-2004 University of Oxford  */

/*  CCOPYRIGHT */

#include "profile.h"

using namespace std;


Profile::Profile()
{
  v.clear();
  lroi = 0;
  rroi = 1;
  maxdef = false;
  mindef = false;
}


Profile::~Profile()
{
}


int Profile::size() const
{
  int counter = 0;
  for (vector<pro_pair>::const_iterator i = v.begin(); i!=v.end(); i++)
    counter++;
  return counter;
}

void Profile::print() const
{
  vector<pro_pair>::const_iterator i;
  for (i = v.begin(); i!=v.end(); i++)
    cout<<(*i).abs<<" : "<<(*i).val<<endl;

}


void Profile::add (const double d, const double t)
{
  //excuse our french.... leaky-leaky!
  //pro_pair * p = new(pro_pair);
  //p->abs = d;
  //p->val = t;
  //v.push_back(*p);

  //corrected code:
  pro_pair p;
  p.abs=d;
  p.val=t;
  v.push_back(p);

  rroi=v.size();
  maxdef = false;
  mindef = false;
}


void Profile::init_roi()
{
  lroi = 0;
  rroi = v.size();
  maxdef = false;
  mindef = false;
}


void Profile::set_lroi(const double abs)
{
  vector<pro_pair >::const_iterator i = v.begin();
  int counter = 0;
  while ((*i).abs < abs && (i++)!=v.end()) counter ++;
  lroi = counter;
  maxdef = false;
  mindef = false;
  if (rroi < lroi) rroi = lroi;
}


void Profile::set_rroi(const double abs)
{
  vector<pro_pair >::const_iterator i = v.end();
  i--;
  int counter = v.size();
  while ((*i).abs > abs && i!=v.begin()) {counter --; i--;}
  rroi = counter;
  maxdef = false;
  mindef = false;
  if (rroi < lroi) lroi = rroi;
}


const double Profile::value(const double d) const
{
  vector<pro_pair>::const_iterator i = v.begin();
  while ((*i).abs < d && i!=v.end())
    i++ ;
  if (i == v.end())
    {
      cerr<<"out of range"<<endl;
      exit (-1);
    }
  return (*i).val;
}


const double Profile::min()
{
  if (mindef) return v[amin].val;
  double result = v[lroi].val;
  int abs = lroi;
  for (int i = lroi; i < rroi; i++)
    {
      if (v[i].val < result) {result = v[i].val; abs = i;}
    }
  mindef = true;
  amin = abs;
  return result;
}


const double Profile::max()
{
  if (maxdef) {return v[amax - 1].val;};
  double result = v[lroi].val;
  int abs = lroi;
  for (int i = lroi; i < rroi; i++)
    {
      if (v[i].val > result) {result = v[i].val; abs = i;}
    }
  maxdef = true;
  amax = abs + 1;
  return result;
}



const double Profile::minabs()
{
  if (mindef) return v[amin].abs;
  else {
    min();
    return v[amin].abs;
  }
}


const double Profile::maxabs()
{
  if (maxdef) return v[amax - 1].abs;
  max();
  return v[amax - 1].abs;
}


const double Profile::threshold(const double d)
{
  return min() + d* (max() - min());
}


const double Profile::begin()
{
  return v[lroi].abs;
}


const double Profile::end()
{
  return v[rroi - 1].abs;
}


const double Profile::next_point_over (const double abs, const double thr)
{
  double t = threshold(thr);
  vector<pro_pair >::const_iterator i = v.begin();
  int counter = 0;
  while ((*i).abs < abs && (i++)!=v.end()) counter ++;

  if (i == v.end()) return -500;

  while ((*i).val < t && counter < rroi) {counter ++; i++; if(i == v.end()) return -500;};

  if (counter == rroi) return -500;
  else
    return (v[counter].abs);
}


const double Profile::next_point_under (const double abs, const double thr)
{
  double t = threshold(thr);

  vector<pro_pair >::const_iterator i = v.begin();
  int counter = 0;
  while ((*i).abs < abs && (i++)!=v.end()) counter ++;

  while ((*i).val > t && counter < rroi) {counter ++; i++; if(i == v.end()) return -500;};

  if (counter == rroi) return -500;
  else
    return (v[counter].abs);
}



const double Profile::last_point_under (const double abs, const double thr)
{
  double t = threshold(thr);

  vector<pro_pair >::const_iterator i = v.end();
  i--;
  int counter = v.size();
  while ((*i).abs > abs && i!=v.begin()) {counter --; i--;}

  while (counter > lroi && (*i).val > t && i!=v.begin()) {counter --; i--;};

  if ((counter == lroi) | (i==v.begin())) return -500;
  else
    return (v[counter - 1].abs);
}



const double Profile::last_point_over (const double abs, const double thr)
{
  double t = threshold(thr);

  vector<pro_pair >::const_iterator i = v.end();
  i--;
  int counter = v.size();
  while ((*i).abs > abs && i!=v.begin()) {counter --; i--;}

  while ((*i).val < t && counter > lroi && i!=v.begin()) {counter --; i--;};

  if ((counter == lroi) | (i==v.begin())) return -500;
  else
    return (v[counter - 1].abs);
}
