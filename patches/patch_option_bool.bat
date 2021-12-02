powershell -Command "(gc include\utils\functions.cc) -replace 'template<> string Option<bool>::config_key', '/*template<>  string Option<bool>::config_key' | Out-File -encoding ASCII include\utils\functions.cc
powershell -Command "(gc include\utils\functions.cc) -replace 'ostream& operator<<', '*/ostream&  operator<<' | Out-File -encoding ASCII include\utils\functions.cc


powershell -Command "(gc include\utils\options.h) -replace 'template<> std::ostream& Option<bool>::print\(std::ostream& s\) const;', 'template<> std::string Option<bool>::config_key() const{return set()? (long_form().empty()?short_form():long_form()):std::string();}template<> bool Option<bool>::set_value(const std::string& s) {value_=s.empty()?(!default_):(s==\"true\"?true:(s==\"false\"?false:value_));return !(unset_=!s.empty()&&s!=\"true\"&&s!=\"false\");}template<> bool Option<bool>::set_value(const std::string& vs,char*[],int,int) {return set_value(vs);}template<> std::ostream& Option<bool>::print(std::ostream& os) const {os << \"# \" << help_text() << std::endl; if (set()) os << config_key().substr(0, config_key().find(\"=\"));return os;}' | Out-File -encoding ASCII include\utils\options.h



