/*! \file FSLProfiler.h
    \brief Contains declaration of class used for profiling

    \author Jesper Andersson
    \version 1.0b, Feb., 2020.
*/
// 
// EddyHelperClasses.h
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2020 University of Oxford 
//

#ifndef FSLProfiler_h
#define FSLProfiler_h

#include <cstdlib>
#include <cstring>
#include <string>
#include <fstream>
#include <vector>
#include <map>
#include <cmath>
#include <iomanip>
#include <sys/time.h>

namespace Utilities {

/****************************************************************//**
*
* \brief This is the exception that is being thrown by FSLProfiler
*
********************************************************************/ 
class FSLProfilerException: public std::exception
{
public:
  FSLProfilerException(const std::string& msg) noexcept : message(msg) {}
  ~FSLProfilerException() noexcept {}
  virtual const char * what() const noexcept { return std::string("FSLProfiler:::  " + message).c_str(); }
private:
  std::string message;
};

/****************************************************************//**
*
*  \class FSLProfiler
*
*  \brief Used for profiling
*
*  This class is used to provide profiling information for eddy.
*  It is intended to be very low overhead when profiling is not
*  turned on, so that the "profiling commands" can be left in the
*  code and only incur a performance penalty when profiling is
*  actually turned on.
*
*  The suggested usage is to static declare objects inside individual
*  functions and then start and stop log entries inside that function.
*  Wether profiling actually takes place or not depends on if it has
*  been turned on with a call to the static function SetProfilingOn.
*
*  \verbatim
main()
{
  FSLProfiler::SetProfilingOn("my_prof_name");

  f1();
}

f1()
{
  static FSLProfiler prof("_"+string(__FILE__)+"_"+string(__func__));
  double key = prof.StartEntry("First bit");
  // Do stuff
  prof.EndEntry(key);
  key = prof.StartEntry("Second bit");
  // Do more stuff
  prof.EndEntry(key);
} 
*  \endverbatim
*
********************************************************************/ 
enum class FSLProfilerStatus { Off, On };
class FSLProfiler
{
public:
  FSLProfiler(const std::string& fname);
  ~FSLProfiler();
  FSLProfilerStatus GetStatus() const { return(_status); }
  double StartEntry(const std::string& descrip);
  void EndEntry(double key);

  static void SetProfilingOn(const std::string& bfname);
  static void SetProfilingOff();
  static FSLProfilerStatus GetProfilingStatus() { return(_status); }
  static unsigned int NumberOfProfilers() { return(_profs.size()); }
private:
  static FSLProfilerStatus           _status;   /// Profiling turned off by default;
  static std::string                  _bfname;   /// Total file name is _bfname + _fname
  static std::vector<FSLProfiler*>   _profs;    /// Vector of pointers to all instances of FSLProfiler

  std::string                         _fname;    /// Internal copy of filename
  std::ofstream                       _out;      /// Output streams (files) at muliple "levels"
  std::map<double,std::string>        _ent;      /// Associative array that binds each entry to a time and a descriptive string
  void set_status_on();
  void flush_and_close();
};

} // End namespace Utilities

#endif // End #ifndef FSLProfiler_h
