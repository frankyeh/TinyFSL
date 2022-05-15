/*! \file FSLProfiler.cpp
    \brief Contains definitions of class used for profiling

    \author Jesper Andersson
    \version 1.0b, Feb., 2020.
*/

#include <algorithm>
#include <chrono>
#include "FSLProfiler.h"

using namespace Utilities;

// Initialisation of non-literal static variables in FSLProfiler
FSLProfilerStatus          FSLProfiler::_status = FSLProfilerStatus::Off;
std::string                FSLProfiler::_bfname; 
std::vector<FSLProfiler*>  FSLProfiler::_profs;

FSLProfiler::FSLProfiler(const std::string& fname) 
{
  _fname = fname;
  std::replace_if(_fname.begin(),_fname.end(),[](char x){return(x=='/' || x==' ');},'_'); // Remove crap
  if (_status==FSLProfilerStatus::On) {
    _out.open(_bfname + _fname);
    if (!_out.is_open()) throw FSLProfilerException("FSLProfiler::FSLProfiler: Unable to open logfile " + _bfname + _fname); 
  }
  _profs.push_back(this);
}

FSLProfiler::~FSLProfiler() 
{
  if (_status==FSLProfilerStatus::On && _out.is_open()) flush_and_close();
  for (auto it=_profs.begin(); it!=_profs.end(); ++it) {
    if (*it == this) {
      _profs.erase(it);
      break;
    }
  }
}

double FSLProfiler::StartEntry(const std::string& descrip) 
{
  if (_status==FSLProfilerStatus::Off) return(0.0);
  else {
    std::chrono::duration<double> tse = std::chrono::system_clock::now().time_since_epoch();
    auto rval = _ent.emplace(tse.count(),descrip);
    if (!rval.second) { // If a duplicate was found
      _out << "Invalid entry: StartEntry with descrip \"" << descrip << "\" resulted in duplicate key " << tse.count() << std::endl;
      return(0.0);
    }
    else return(tse.count());
  }
}

void FSLProfiler::EndEntry(double key) 
{
  if (_status==FSLProfilerStatus::Off || key==0.0) return;
  else {
    std::chrono::duration<double> tse = std::chrono::system_clock::now().time_since_epoch();
    double dur = tse.count() - key;
    auto it = _ent.find(key);
    if (it != _ent.end()) {
      const std::string& descrip = _ent.at(key); 
      _out << "Entry: " << descrip << " Duration: " << dur << std::endl;
      _ent.erase(key);
    }
    else _out << "Invalid entry: EndEntry called with invalid key " << key << std::endl;
  }
  return;
}

void FSLProfiler::SetProfilingOn(const std::string& bfname) 
{ 
  if (_status!=FSLProfilerStatus::On) { // If we turn it from off to on
    _bfname = bfname;
    _status = FSLProfilerStatus::On;
    for (unsigned int i=0; i<_profs.size(); i++) _profs[i]->set_status_on();
  }
  else if (bfname != _bfname) { // If we go from on to on, but with a new basename
    for (unsigned int i=0; i<_profs.size(); i++) _profs[i]->flush_and_close();
    _bfname = bfname;
    for (unsigned int i=0; i<_profs.size(); i++) _profs[i]->set_status_on();
  }
}

void FSLProfiler::SetProfilingOff() 
{
  if (_status!=FSLProfilerStatus::Off) { // If we actually change something
    for (unsigned int i=0; i<_profs.size(); i++) _profs[i]->flush_and_close();
    _status = FSLProfilerStatus::Off;
  }
}

void FSLProfiler::set_status_on() 
{
  if (!_out.is_open()) {
    _out.open(_bfname + _fname);
    if (!_out.is_open()) throw FSLProfilerException("FSLProfiler::set_status_on: Unable to open logfile " + _bfname + _fname); 
  }
}

void FSLProfiler::flush_and_close() 
{
  if (_ent.size() && _out.is_open()) {
    for (auto it=_ent.begin(); it!=_ent.end(); ++it) {
      _out << "Invalid entry: Uncompleted entry: description: " << it->second << std::endl;
    }
  }
  if (!_ent.empty()) _ent.clear();
  if (_out.is_open()) _out.close();
}


