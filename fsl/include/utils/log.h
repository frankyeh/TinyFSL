/*  log.h

    Mark Woolrich, FMRIB Image Analysis Group

    Copyright (C) 1999-2000 University of Oxford  */

/* The Log class allows for instantiation of more than one Log
either sharing directories or not. However, Logs can not share log files.
Or you can work with the LogSIngleton class.

A Log can open new logfiles in the same log directory or start on an
entirely new directory. You can stream directly to a Log with flags
determining streaming to the Logfile and/or cout. */

/*  CCOPYRIGHT  */

#if !defined(log_h)
#define log_h

#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <sstream>

#include "armawrap/newmatap.h"
#include "armawrap/newmatio.h"

namespace Utilities{

  template<class t> std::string tostring(const t obj)
  {
    std::ostringstream str;
    str << obj;
    return str.str();
  }

  class Log
    {
    public:
      Log():logEstablished(false) {}

      Log(const std::string& pdirname, const std::string& plogfilename = "logfile", bool pstream_to_logfile = true, bool pstream_to_cout = false, bool makedir = true):logEstablished(false)
	{
	  if(makedir)
	    makeDir(pdirname, plogfilename, pstream_to_logfile, pstream_to_cout);
	  else
	    setDir(pdirname, plogfilename, pstream_to_logfile, pstream_to_cout);
	}

      ~Log() { logfileout.close(); }

      /** Need to call makeDir or setDir before Log can be used */

      /** Makes a directory to place results into:
	  keeps adding "+" to pdirname until unique directory is made. */
      /** The stream_to* variables define the streaming behaviour */
      void makeDir(const std::string& pdirname, const std::string& plogfilename = "logfile", bool pstream_to_logfile = true, bool pstream_to_cout = false);

      /** Sets an existing directory to place results into. */
      /** The stream_to* variables define the streaming behaviour */
      void setDir(const std::string& pdirname, const std::string& plogfilename = "logfile", bool pstream_to_logfile = true, bool pstream_to_cout = false, std::ios_base::openmode mode=std::ios::app);

      /** Sets an existing directory to place results into. */
      /** If does not exist then makes it. */
      /** The stream_to* variables define the streaming behaviour */
      void setthenmakeDir(const std::string& pdirname, const std::string& plogfilename = "logfile", bool pstream_to_logfile = true, bool pstream_to_cout = false);

      /** Closes old logfile buffer and attempts to open new one with name specified and sets streaming to logfile on */
      void setLogFile(const std::string& plogfilename, std::ios_base::openmode mode=std::ios::app);

      const std::string& getDir() const { if(!logEstablished)throw std::runtime_error("Log not setup");return dir; }

      const std::string& getLogFileName() const { if(!logEstablished)throw std::runtime_error("Log not setup");return logfilename; }

      /** returns passed in filename appended onto the end of the dir name */
      const std::string appendDir(const std::string& filename) const;

      std::ofstream& get_logfile_ofstream() { return  logfileout;}

      inline void flush() {
	if(stream_to_logfile)
	  logfileout.flush();

	if(stream_to_cout)
	  std::cout.flush();
      }

      /** allows streaming into cout and/or logfile depending upon the */
      /** stream_to_cout and stream_to_logfile respectively */
      /** use like a normal ostream, e.g. log.str() << "hello" << endl */
      /** NOTE: can simply stream straight to Log instead, e.g. log << "hello" << endl */
      Log& str();

      /** sets whether or not you stream to cout */
      void set_stream_to_cout(bool in = true) { stream_to_cout = in; }

      /** sets whether or not you stream to logfile */
      void set_stream_to_logfile(bool in = true) {
	if(!stream_to_logfile && in)
	  {
	    if(logfileout.bad())
	      {
		std::cerr << "Warning: Unable to stream to logfile " << logfilename << ". Need to have called log.setLogFile. Therefore, no streaming to logfile will be performed" << std::endl;
	      }
	  }
	else stream_to_logfile = in;
      }

    private:

      const Log& operator=(Log&);
      Log(Log&);

      std::string dir;
      std::ofstream logfileout;
      std::string logfilename;

      bool logEstablished;

      bool stream_to_logfile;
      bool stream_to_cout;

      friend Log& operator<<(Log& log, std::ostream& (*obj) (std::ostream &));

      template<class t>
	friend Log& operator<<(Log& log, const t& obj);

      template<class t>
	friend Log& operator<<(Log& log, t& obj);
    };

  template<class t> Log& operator<<(Log& log, const t& obj)
    {
      if(log.stream_to_logfile)
	log.logfileout << obj;

      if(log.stream_to_cout)
	std::cout << obj;

      return log;
    }

  template<class t> Log& operator<<(Log& log, t& obj)
    {
      if(log.stream_to_logfile)
	log.logfileout << obj;

      if(log.stream_to_cout)
	std::cout << obj;

      return log;
    }

  class LogSingleton
    {
    public:

      static Log& getInstance();

      ~LogSingleton() { delete logger; }

      /** hacked in utility provides a global counter for general use: */
      static int counter() { return count++; }

    private:
      LogSingleton() {}

      const LogSingleton& operator=(LogSingleton&);
      LogSingleton(LogSingleton&);

      static Log* logger;

      static int count;

    };

  inline Log& LogSingleton::getInstance(){
    if(logger == NULL)
      logger = new Log();

    return *logger;
  }

  inline void Log::setLogFile(const std::string& plogfilename, std::ios_base::openmode mode) {

    if(!logEstablished)
      {
	throw std::runtime_error("Log not setup");
      }

    logfileout.close();

    logfilename = plogfilename;

    // setup logfile
    logfileout.open((dir + "/" + logfilename).c_str(), mode);
    if(logfileout.bad())
      {
        throw std::runtime_error(std::string(std::string("Unable to setup logfile ")+logfilename+std::string(" in directory ")+dir).c_str());
      }

    stream_to_logfile = true;

    logEstablished = true;
  }

  inline Log& Log::str() {

    if(!logEstablished)
      {
      throw std::runtime_error("Log not setup");
      }

    return *this;
  }

  inline const std::string Log::appendDir(const std::string& filename) const {

    if(!logEstablished)
      {
	throw std::runtime_error("Log not setup");
      }

    return dir + "/" + filename;
  }


  inline Log& operator<<(Log& log, std::ostream& (*obj)(std::ostream &))
    {
      if(log.stream_to_logfile)
 	log.logfileout << obj;

      if(log.stream_to_cout)
  	std::cout << obj;

      return log;
    }

}

#endif
