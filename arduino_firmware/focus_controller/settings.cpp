// ============================================================================
// Settings implementation - see settings.h
// ============================================================================
#include "settings.h"

#include <avr/eeprom.h>

// EEPROM layout at EEPROM_ADDR (28 bytes)
struct __attribute__((packed)) EepromBlob {
  uint16_t magic;
  uint8_t version;
  uint8_t slimOn;
  int32_t slimMin;
  int32_t slimMax;
  uint32_t accel;
  uint32_t maxSpeed;
  uint32_t tmoMs;
};

void Settings::load() {
  EepromBlob b;
  eeprom_read_block(&b, (const void *)EEPROM_ADDR, sizeof(b));
  if (b.magic != EEPROM_MAGIC || b.version != EEPROM_VERSION) {
    reset();
    return;
  }
  slimOn = b.slimOn ? 1 : 0;
  slimMin = b.slimMin;
  slimMax = b.slimMax;
  accel = constrain(b.accel, 1UL, HARD_MAX_ACCEL_SPS2);
  maxSpeed = constrain(b.maxSpeed, MIN_SPEED_SPS, HARD_MAX_SPEED_SPS);
  tmoMs = constrain(b.tmoMs, 0UL, 60000UL);
  _loaded = true;
}

void Settings::save() {
  EepromBlob b;
  b.magic = EEPROM_MAGIC;
  b.version = EEPROM_VERSION;
  b.slimOn = slimOn ? 1 : 0;
  b.slimMin = slimMin;
  b.slimMax = slimMax;
  b.accel = accel;
  b.maxSpeed = maxSpeed;
  b.tmoMs = tmoMs;
  eeprom_update_block(&b, (void *)EEPROM_ADDR, sizeof(b));
}

void Settings::applyTo(AxisConfig &cfg) const {
  cfg.slimOn = slimOn ? 1 : 0;
  cfg.slimMin = slimMin;
  cfg.slimMax = slimMax;
  cfg.accel = accel;
  cfg.maxSpeed = maxSpeed;
}

void Settings::reset() {
  slimOn = DEFAULT_SLIM_ON;
  slimMin = DEFAULT_SLIM_MIN;
  slimMax = DEFAULT_SLIM_MAX;
  accel = DEFAULT_ACCEL_SPS2;
  maxSpeed = DEFAULT_MAX_SPEED_SPS;
  tmoMs = DEFAULT_SERIAL_TMO_MS;
}
