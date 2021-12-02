powershell -Command "(gc include\armawrap\base.hpp) -replace 'inline AWCallManager<elem_type, wT, is_simple_armawrap_type<wT>::value>', 'inline elem_type&' | Out-File -encoding ASCII include\armawrap\base.hpp"
powershell -Command "(gc include\armawrap\base.hpp) -replace 'return AWCallManager<elem_type, wT, is_simple_armawrap_type<wT>::value>', 'return *&AWCallManager<elem_type, wT, is_simple_armawrap_type<wT>::value>' | Out-File -encoding ASCII include\armawrap\base.hpp"


powershell -Command "(gc include\armawrap\operator_call.hpp) -replace 'inline void operator\*=', 'elem_type* operator&(){if (i != -1) return const_cast<elem_type *>(&lookup1D(target, i));else return const_cast<elem_type *>(&lookup2D(target, row, col));} inline void operator*=' | Out-File -encoding ASCII include\armawrap\operator_call.hpp"

