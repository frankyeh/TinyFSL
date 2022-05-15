//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Provides the "make_unique" function missing from C++11
/// \details This is the same as the implementation included as standard in C++14, and is
///          included in order to maintain consistency when using the pimpl idion as is common
///          in much of my code.
/// \author Frederik Lange
/// \date February 2018
/// \copyright Copyright (C) 2018 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////
#ifndef CBF_MEMORY_H
#define CBF_MEMORY_H

#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

namespace CBF {
    template<class T> struct _Unique_if {
        typedef std::unique_ptr<T> _Single_object;
    };

    template<class T> struct _Unique_if<T[]> {
        typedef std::unique_ptr<T[]> _Unknown_bound;
    };

    template<class T, size_t N> struct _Unique_if<T[N]> {
        typedef void _Known_bound;
    };

    template<class T, class... Args>
        typename _Unique_if<T>::_Single_object
        make_unique(Args&&... args) {
            return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
        }

    template<class T>
        typename _Unique_if<T>::_Unknown_bound
        make_unique(size_t n) {
            typedef typename std::remove_extent<T>::type U;
            return std::unique_ptr<T>(new U[n]());
        }

    template<class T, class... Args>
        typename _Unique_if<T>::_Known_bound
        make_unique(Args&&...) = delete;
} // namespace CBF
#endif // CBF_MEMORY_H
