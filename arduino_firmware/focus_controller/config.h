// ============================================================================
// Motorized Focus Controller - Configuration
// ============================================================================
// Hardware: CRD5103PB (Oriental Motor 5-phase microstep driver) + PK513PB motor
//   - 2-pulse (CW/CCW) mode, negative logic (motor steps on falling edge)
//   - Wiring:  D11 -> CW+ (PLS+),  D12 -> CCW+ (DIR+),  GND -> CW-/CCW- common
//   - 100-220R series resistor per signal line recommended
//   - CN2 TIMING (pins 11-12) is a 24V open-collector OUTPUT - NEVER wire to AVR
// ============================================================================
#ifndef FOCUS_CONFIG_H
#define FOCUS_CONFIG_H

#include <Arduino.h>
#include <avr/wdt.h>

// ---- Identity ----
#define FW_NAME_STR           "FOCUSCTRL"
#define FW_VERSION_STR        "1.0"

// ---- Pins ----
#define PIN_CW                11          // D11 -> driver CW+ (PLS+)
#define PIN_CCW               12          // D12 -> driver CCW+ (DIR+)

// Optional CN2 signals - disabled until physically wired and verified:
#define HAS_AWOFF             0           // CN2 5-6: windings off (free shaft)
#define PIN_AWOFF             5
#define AWOFF_ACTIVE_LOW      1           // verify polarity at wiring time!
#define HAS_CDINH             0           // CN2 7-8: release current cutback
#define PIN_CDINH             7
#define CDINH_ACTIVE_LOW      0           // verify polarity at wiring time!

// ---- Serial ----
#define SERIAL_BAUD           115200UL
#define MAX_CMD_LEN           64
#define MAX_CHARS_PER_LOOP    8           // bound RX processing per loop pass (jitter control)
#define STALE_LINE_MS         50UL        // drop a partial line silent this long (junk/truncation guard)

// ---- Motion ----
#define MIN_SPEED_SPS         10UL
#define HARD_MAX_SPEED_SPS    5000UL
#define DEFAULT_MAX_SPEED_SPS 2000UL
#define DEFAULT_ACCEL_SPS2    20000UL
#define HARD_MAX_ACCEL_SPS2   100000UL
#define DEFAULT_MVSPD_SPS     1000UL
#define DEFAULT_SERIAL_TMO_MS 5000UL      // stop motion after this long without serial RX (0 = off)
#define HALF_PERIOD_MIN_US    100UL       // = 5000 sps cap
#define HALF_PERIOD_MAX_US    50000UL     // = 10 sps floor

// ---- Software limits ----
#define DEFAULT_SLIM_ON       0
#define DEFAULT_SLIM_MIN      (-2000000L)
#define DEFAULT_SLIM_MAX      (2000000L)

// ---- Watchdog ----
#define WDT_TIMEOUT           WDTO_2S
#define WDT_FEED_MS           500UL

// ---- EEPROM ----
#define EEPROM_MAGIC          0x46D5
#define EEPROM_VERSION        1
#define EEPROM_ADDR           0

// ---- Axis count (scalability: instantiate more StepAxis objects) ----
#define AXIS_COUNT            1

#endif // FOCUS_CONFIG_H
