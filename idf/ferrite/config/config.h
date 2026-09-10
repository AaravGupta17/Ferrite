// idf/ferrite/config/config.h — ESP32 per-target capacity ceilings (Section 4.4).
//
// This header is the IDF component's override for core/config.h: types.h does
// `#include <config.h>`, and this directory is earlier on the include path,
// so the ESP32 build gets device-sized ceilings without touching the shared
// source. Keep the values in sync with cmake/esp32.toolchain.cmake.
#ifndef FERRITE_CONFIG_H
#define FERRITE_CONFIG_H

#define FERRITE_MAX_DIMS 4         /* tensor rank ceiling     */
#define FE_MAX_NODES     48        /* graph node ceiling      */
#define FE_MAX_TENSORS   96        /* tensor registry ceiling */
#define FE_MAX_ALLOCS    96        /* planner allocation ceiling */
#define FERRITE_PLANNER_ALIGN 4    /* scalar-only: natural alignment */

#endif // FERRITE_CONFIG_H