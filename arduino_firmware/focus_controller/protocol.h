// ============================================================================
// Protocol - non-blocking serial parser, command dispatcher, replies/events
// ============================================================================
// Line-based ASCII protocol at SERIAL_BAUD 8N1. See docs/protocol.md.
// All replies are single lines <= MAX_CMD_LEN chars, never printed from
// inside the step engine (pulse timing must stay jitter-free).
#ifndef FOCUS_PROTOCOL_H
#define FOCUS_PROTOCOL_H

#include <Arduino.h>
#include "config.h"
#include "axis_engine.h"
#include "settings.h"

class Protocol {
public:
  Protocol(StepAxis *axis, Settings *settings)
    : _axis(axis), _settings(settings), _len(0), _overflow(false), _lastRxMs(0) {}

  void begin();           // boot banner (EV:WDT first if watchdog reset)
  void poll();            // read + dispatch serial input (bounded per call)
  void drainEvents();     // print pending axis events
  void checkSerialTmo();  // serial inactivity stop (PC unplug safety)

private:
  StepAxis *_axis;
  Settings *_settings;
  char _buf[MAX_CMD_LEN + 1];
  uint8_t _len;
  bool _overflow;
  uint32_t _lastRxMs;

  void finishLine();
  void dispatch();
  void reply(const char *fmt, ...) __attribute__((format(printf, 2, 3)));

  // Handlers (arg = text after the command token, may be empty)
  void hPing(const char *arg);
  void hVer(const char *arg);
  void hStatus(const char *arg);
  void hSpd(const char *arg);
  void hMvSpd(const char *arg);
  void hMvSpdQ(const char *arg);
  void hMove(const char *arg);
  void hGoto(const char *arg);
  void hStop(const char *arg);
  void hZero(const char *arg);
  void hSlim(const char *arg);
  void hSlimQ(const char *arg);
  void hAwoff(const char *arg);
  void hAwoffQ(const char *arg);
  void hCutb(const char *arg);
  void hCutbQ(const char *arg);
  void hCfg(const char *arg);
  void hCfgQ(const char *arg);
};

#endif // FOCUS_PROTOCOL_H
