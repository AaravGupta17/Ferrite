// core/platform_esp32.c — ESP32 platform seam (Section 4.2).
//
// FreeRTOS + ESP-IDF implementations of the Section 2.6 seam. Used by the
// idf/ferrite component; a plain-CMake ESP32 build (cmake/esp32.toolchain.
// cmake) can compile this file too, as long as esp_timer and esp_rom headers
// are on the include path (they are in the ESP-IDF toolchain).
//
// The profiler lives on esp_timer's microsecond counter (monotonic, 64-bit)
// — this is the ESP32 analogue of CLOCK_MONOTONIC. Logging routes through the
// ROM printk so it works before the driver console is up.

#include "platform.h"

#include <stdint.h>
#include <string.h>

#if defined(__XTENSA__) || defined(ESP_PLATFORM)

#include "esp_timer.h"
#include "esp_rom_sys.h"

uint64_t fe_platform_now_ns(void) {
    return (uint64_t)esp_timer_get_time() * 1000ull;  /* us -> ns */
}

void fe_platform_log_write(const char *line) {
    esp_rom_printf("%s", line);  /* low-level UART log, no stdio alloc */
}

/*
 * Parallel subsystem is compiled out on this target (FERRITE_NO_PARALLEL).
 * Keep the seam honest: threads are unsupported rather than silently
 * present-yet-broken. Single core, no POSIX threads.
 */
int  fe_platform_threads_supported(void) { return 0; }
int  fe_platform_thread_create(void **handle, void *(*fn)(void *), void *arg) {
    (void)handle; (void)fn; (void)arg; return 0;
}
void fe_platform_thread_join(void *handle) { (void)handle; }

int fe_platform_cpu_count(void) { return 1; }  /* Xtensa core count */

#else /* !__XTENSA__ */

/*
 * Compiling core platform_esp32.c on a non-ESP32 host is a configuration
 * error, not a silent host stub: the IDF component's source list is what
 * decides which platform file ships, so a desktop build should never pull
 * this in (Section 4.2, fail-loudly rule).
 */
#error "core/platform_esp32.c requires an ESP32/XTensa toolchain"

#endif