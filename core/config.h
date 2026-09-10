// core/config.h — compile-time capacity ceilings (non-CMake default copy).
//
// CMake generates a build-local copy from core/config.h.in at configure
// time; this committed file holds the same desktop defaults so the Makefile
// build path (which does not run CMake) compiles unchanged. Keep the values
// here in sync with the defaults in CMakeLists.txt. Never edit this for a
// device build — set FERRITE_MAX_DIMS / FE_MAX_NODES / FE_MAX_TENSORS /
// FE_MAX_ALLOCS at configure time instead.
#ifndef FERRITE_CONFIG_H
#define FERRITE_CONFIG_H

#define FERRITE_MAX_DIMS 8      /* tensor rank ceiling     */
#define FE_MAX_NODES     512    /* graph node ceiling      */
#define FE_MAX_TENSORS   1024   /* tensor registry ceiling */
#define FE_MAX_ALLOCS    1024   /* planner allocation ceiling */
#define FERRITE_PLANNER_ALIGN 64 /* activation-buffer align (SIMD) */

#endif // FERRITE_CONFIG_H