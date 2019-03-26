#ifndef UTIL_H
#define UTIL_H

#include <chrono>
// this file define a set of macros for logging and timing
#ifdef DEBUG_PRINT
#undef DEBUG_PRINT
#endif

#ifdef LOGGING
#define DEBUG_PRINT(x) x
#else
#define DEBUG_PRINT(x) 
#endif


inline std::chrono::high_resolution_clock::time_point  NOW() { 
    return std::chrono::high_resolution_clock::now();
}

inline float DIFF_T(std::chrono::high_resolution_clock::time_point &t1,  std::chrono::high_resolution_clock::time_point &t2) {
  return std::chrono::duration_cast<std::chrono::microseconds >(t2 - t1).count() / 1000.0;
}


#endif // UTIL_H
