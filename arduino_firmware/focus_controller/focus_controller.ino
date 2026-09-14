// ============================================================================
// Motorized Focus Controller - main sketch
// ============================================================================
// Single-axis Z-focus stepper control for a CRD5103PB 5-phase microstep
// driver (2-pulse CW/CCW mode, Arduino Uno/Nano).
//
// Non-blocking micros()-based pulse engine, line-based ASCII protocol at
// 115200 8N1, hardware watchdog + serial inactivity stop. See
// docs/protocol.md for the command reference and config.h for wiring/pins.
// ============================================================================
#include <avr/wdt.h>

#include "config.h"
#include "axis_engine.h"
#include "settings.h"
#include "protocol.h"

StepAxis g_axis;
Settings g_settings;
Protocol g_proto(&g_axis, &g_settings);
uint32_t g_lastWdtFeed = 0;

void setup() {
  wdt_enable(WDT_TIMEOUT);
  Serial.begin(SERIAL_BAUD);

  g_settings.load();
  AxisConfig cfg;
  cfg.cwPin = PIN_CW;
  cfg.ccwPin = PIN_CCW;
  cfg.hasAwoff = HAS_AWOFF;
  cfg.awoffPin = PIN_AWOFF;
  cfg.awoffActiveLow = AWOFF_ACTIVE_LOW;
  cfg.hasCdin = HAS_CDINH;
  cfg.cdinPin = PIN_CDINH;
  cfg.cdinActiveLow = CDINH_ACTIVE_LOW;
  cfg.mvSpeed = DEFAULT_MVSPD_SPS;
  g_settings.applyTo(cfg);   // maxSpeed, accel, slim*
  g_axis.begin(cfg);

  g_proto.begin();
  g_lastWdtFeed = millis();
}

void loop() {
  g_proto.poll();
  g_axis.update();
  g_proto.drainEvents();
  g_proto.checkSerialTmo();

  if ((uint32_t)(millis() - g_lastWdtFeed) >= WDT_FEED_MS) {
    wdt_reset();
    g_lastWdtFeed = millis();
  }
}
