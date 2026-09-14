// ============================================================================
// Settings - EEPROM persistence of user-configurable runtime parameters
// ============================================================================
// Persisted: slim bounds/state, accel, max speed, serial TMO.
// Not persisted: position, mode, speed, mvSpeed (runtime-only).
// Written only on explicit CFG:/SLIM: commands -> EEPROM wear is negligible.
#ifndef FOCUS_SETTINGS_H
#define FOCUS_SETTINGS_H

#include <Arduino.h>
#include "config.h"
#include "axis_engine.h"

class Settings {
public:
  uint8_t slimOn = DEFAULT_SLIM_ON;
  int32_t slimMin = DEFAULT_SLIM_MIN;
  int32_t slimMax = DEFAULT_SLIM_MAX;
  uint32_t accel = DEFAULT_ACCEL_SPS2;
  uint32_t maxSpeed = DEFAULT_MAX_SPEED_SPS;
  uint32_t tmoMs = DEFAULT_SERIAL_TMO_MS;

  void load();                       // EEPROM -> fields (validated; defaults on bad data)
  void save();                       // fields -> EEPROM (update-style, wear-aware)
  void applyTo(AxisConfig &cfg) const;
  bool loadedFromEeprom() const { return _loaded; }

private:
  bool _loaded = false;

  void reset();
};

#endif // FOCUS_SETTINGS_H
