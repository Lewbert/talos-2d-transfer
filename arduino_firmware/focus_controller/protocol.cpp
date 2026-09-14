// ============================================================================
// Protocol implementation - see protocol.h
// ============================================================================
#include "protocol.h"

#include <avr/wdt.h>
#include <ctype.h>
#include <stdarg.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

void Protocol::begin() {
  if (MCUSR & _BV(WDRF)) {   // watchdog reset: tell the host the firmware restarted
    MCUSR = 0;
    Serial.println("EV:WDT");
  }
  Serial.println("BOOT:" FW_NAME_STR ":" FW_VERSION_STR);
  Serial.println("READY");
  _lastRxMs = millis();
}

// ---------------------------------------------------------------------------
// Serial input
// ---------------------------------------------------------------------------

static bool isCmdChar(char c) {
  // The full RX vocabulary: commands, queries, digits, signs, separators.
  // Anything else (high-bit junk from USB re-enumeration, control chars)
  // is a framing break, never folded into a command.
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
         (c >= '0' && c <= '9') ||
         c == ':' || c == '?' || c == ',' || c == '-' || c == '+';
}

void Protocol::poll() {
  // Stale partial line: a truncated/half-written command that never
  // completed. Drop it after STALE_LINE_MS of silence so it can never
  // glue onto the next command. (_lastRxMs itself is untouched — the
  // serial-inactivity stop depends on it.) A full 64-byte line takes
  // ~5.6 ms at 115200, so 50 ms cannot flush a legitimate command.
  if (_len > 0 && (uint32_t)(millis() - _lastRxMs) >= STALE_LINE_MS) {
    _len = 0;
    _overflow = false;
  }
  for (uint8_t i = 0; i < MAX_CHARS_PER_LOOP; i++) {
    if (!Serial.available()) break;
    char c = (char)Serial.read();
    _lastRxMs = millis();
    if (c == '\n') {
      if (_len > 0 || _overflow) finishLine();
      continue;
    }
    if (c == '\r') continue;
    if (!isCmdChar(c)) {
      // Non-protocol byte = a framing break: discard the byte AND any
      // partial line before it (it may be a torn half-command).
      _len = 0;
      _overflow = false;
      continue;
    }
    if (_len < MAX_CMD_LEN) _buf[_len++] = c;
    else _overflow = true;
  }
}

void Protocol::finishLine() {
  if (_overflow) {
    reply("ERR:OVERFLOW");
  } else {
    _buf[_len] = '\0';
    for (uint8_t i = 0; i < _len; i++) _buf[i] = toupper(_buf[i]);  // case-insensitive
    dispatch();
  }
  _len = 0;
  _overflow = false;
}

void Protocol::dispatch() {
  // Split token at first ':' - queries ("VER?", "STATUS?") carry no colon
  char *tok = _buf;
  char *arg = nullptr;
  for (uint8_t i = 0; i < _len; i++) {
    if (_buf[i] == ':') {
      _buf[i] = '\0';
      arg = &_buf[i + 1];
      break;
    }
  }

  struct CmdEntry {
    const char *tok;
    void (Protocol::*fn)(const char *arg);
  };
  static const CmdEntry kCmds[] = {
    {"PING",    &Protocol::hPing},
    {"VER?",    &Protocol::hVer},
    {"STATUS?", &Protocol::hStatus},
    {"SPD",     &Protocol::hSpd},
    {"MVSPD?",  &Protocol::hMvSpdQ},
    {"MVSPD",   &Protocol::hMvSpd},
    {"MOVE",    &Protocol::hMove},
    {"GOTO",    &Protocol::hGoto},
    {"STOP",    &Protocol::hStop},
    {"ZERO",    &Protocol::hZero},
    {"SLIM?",   &Protocol::hSlimQ},
    {"SLIM",    &Protocol::hSlim},
    {"AWOFF?",  &Protocol::hAwoffQ},
    {"AWOFF",   &Protocol::hAwoff},
    {"CUTB?",   &Protocol::hCutbQ},
    {"CUTB",    &Protocol::hCutb},
    {"CFG?",    &Protocol::hCfgQ},
    {"CFG",     &Protocol::hCfg},
  };

  for (const CmdEntry &e : kCmds) {
    if (strcmp(tok, e.tok) == 0) {
      (this->*e.fn)(arg ? arg : "");
      return;
    }
  }
  reply("ERR:UNKNOWN:%s", _buf);
}

void Protocol::reply(const char *fmt, ...) {
  char out[MAX_CMD_LEN];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(out, sizeof(out), fmt, ap);
  va_end(ap);
  Serial.println(out);
}

// ---------------------------------------------------------------------------
// Events and watchdog
// ---------------------------------------------------------------------------

void Protocol::drainEvents() {
  AxisEvent e;
  while (_axis->popEvent(e)) {
    switch (e.type) {
      case EV_STOP: reply("EV:STOP:%ld", (long)e.arg1); break;
      case EV_DONE: reply("EV:DONE:%ld", (long)e.arg1); break;
      case EV_LIM:  reply("EV:LIM:%c:%ld", e.arg2 > 0 ? '+' : '-', (long)e.arg1); break;
    }
  }
}

void Protocol::checkSerialTmo() {
  if (_settings->tmoMs == 0) return;
  if (!_axis->isMoving()) return;
  if ((uint32_t)(millis() - _lastRxMs) >= _settings->tmoMs) {
    _axis->stopNow();  // pushes EV_STOP
    reply("EV:TMO:%ld", (long)_axis->position());
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static bool parseInt(const char *s, int32_t &out) {
  if (!s || !*s) return false;
  char *end = nullptr;
  long v = strtol(s, &end, 10);
  if (end == s || *end != '\0') return false;
  out = (int32_t)v;
  return true;
}

// Split "min:max" in place (buffer is mutable). Returns false on malformed input.
static bool parseTwoInts(char *s, int32_t &a, int32_t &b) {
  char *colon = strchr(s, ':');
  if (!colon) return false;
  *colon = '\0';
  return parseInt(s, a) && parseInt(colon + 1, b);
}

static const char *modeStr(AxisMode m) {
  switch (m) {
    case MODE_CONT:  return "CONT";
    case MODE_TRAP:  return "TRAP";
    case MODE_LIMIT: return "LIMIT";
    default:         return "IDLE";
  }
}

// ---------------------------------------------------------------------------
// Command handlers
// ---------------------------------------------------------------------------

void Protocol::hPing(const char *arg) {
  (void)arg;
  reply("PONG");
}

void Protocol::hVer(const char *arg) {
  (void)arg;
  reply("VER:" FW_NAME_STR ":" FW_VERSION_STR);
}

void Protocol::hStatus(const char *arg) {
  (void)arg;
  int8_t b = _axis->blockedDir();
  reply("S:POS:%ld,MODE:%s,V:%ld,SPD:%ld,LIM:%c,SLIM:%d",
        (long)_axis->position(), modeStr(_axis->mode()),
        (long)_axis->currentSpeed(), (long)_axis->targetSpeed(),
        b > 0 ? '+' : (b < 0 ? '-' : '0'), _axis->slimEnabled() ? 1 : 0);
}

void Protocol::hSpd(const char *arg) {
  int32_t v;
  if (!parseInt(arg, v)) { reply("ERR:BAD_FORMAT"); return; }
  int32_t applied;
  int8_t err = _axis->setContinuous(v, applied);
  switch (err) {
    case AXIS_OK:        reply("OK:SPD:%ld", (long)applied); break;
    case AXIS_ERR_BUSY:  reply("ERR:BUSY"); break;
    case AXIS_ERR_LIMIT: reply("ERR:LIMIT"); break;
    default:             reply("ERR:BAD_VALUE:%s", arg); break;
  }
}

void Protocol::hMvSpd(const char *arg) {
  int32_t v;
  if (!parseInt(arg, v) || v <= 0) { reply("ERR:BAD_VALUE:%s", arg); return; }
  _axis->setMvSpeed((uint32_t)v);
  reply("OK:MVSPD:%lu", (unsigned long)_axis->mvSpeed());
}

void Protocol::hMvSpdQ(const char *arg) {
  (void)arg;
  reply("MVSPD:%lu", (unsigned long)_axis->mvSpeed());
}

void Protocol::hMove(const char *arg) {
  int32_t n;
  if (!parseInt(arg, n)) { reply("ERR:BAD_FORMAT"); return; }
  int8_t err = _axis->startMove(n);
  switch (err) {
    case AXIS_OK:        reply("OK:MOVE:%ld", (long)n); break;
    case AXIS_ERR_BUSY:  reply("ERR:BUSY"); break;
    case AXIS_ERR_LIMIT: reply("ERR:LIMIT"); break;
    case AXIS_ERR_RANGE: reply("ERR:RANGE"); break;
    default:             reply("ERR:BAD_VALUE:%s", arg); break;
  }
}

void Protocol::hGoto(const char *arg) {
  int32_t absPos;
  if (!parseInt(arg, absPos)) { reply("ERR:BAD_FORMAT"); return; }
  int64_t rel = (int64_t)absPos - _axis->position();
  if (rel > INT32_MAX || rel < INT32_MIN) { reply("ERR:RANGE"); return; }
  int8_t err = _axis->startMove((int32_t)rel);
  switch (err) {
    case AXIS_OK:        reply("OK:GOTO:%ld", (long)absPos); break;
    case AXIS_ERR_BUSY:  reply("ERR:BUSY"); break;
    case AXIS_ERR_LIMIT: reply("ERR:LIMIT"); break;
    case AXIS_ERR_RANGE: reply("ERR:RANGE"); break;
    default:             reply("ERR:BAD_VALUE:%s", arg); break;
  }
}

void Protocol::hStop(const char *arg) {
  (void)arg;
  _axis->stopNow();
  reply("OK:STOP");
}

void Protocol::hZero(const char *arg) {
  (void)arg;
  if (_axis->zeroPosition()) reply("OK:ZERO");
  else reply("ERR:BUSY");
}

void Protocol::hSlim(const char *arg) {
  if (strcmp(arg, "0") == 0 || strcmp(arg, "1") == 0) {
    bool on = (arg[0] == '1');
    int8_t err = _axis->setSlim(on);
    if (err != AXIS_OK) { reply("ERR:RANGE"); return; }
    int32_t mn, mx;
    _axis->getSlim(mn, mx);
    _settings->slimOn = on ? 1 : 0;
    _settings->save();
    reply("OK:SLIM:%d:%ld:%ld", on ? 1 : 0, (long)mn, (long)mx);
    return;
  }
  // SLIM:SET:<min>:<max>
  if (strncmp(arg, "SET:", 4) != 0) { reply("ERR:BAD_FORMAT"); return; }
  int32_t mn, mx;
  if (!parseTwoInts(const_cast<char *>(arg) + 4, mn, mx)) { reply("ERR:BAD_FORMAT"); return; }
  int8_t err = _axis->setSlimBounds(mn, mx);
  if (err != AXIS_OK) { reply("ERR:RANGE"); return; }
  _settings->slimOn = 1;
  _settings->slimMin = mn;
  _settings->slimMax = mx;
  _settings->save();
  reply("OK:SLIM:1:%ld:%ld", (long)mn, (long)mx);
}

void Protocol::hSlimQ(const char *arg) {
  (void)arg;
  int32_t mn, mx;
  _axis->getSlim(mn, mx);
  reply("SLIM:%d:%ld:%ld", _axis->slimEnabled() ? 1 : 0, (long)mn, (long)mx);
}

void Protocol::hAwoff(const char *arg) {
  if (strcmp(arg, "0") != 0 && strcmp(arg, "1") != 0) { reply("ERR:BAD_FORMAT"); return; }
  bool on = (arg[0] == '1');
  int8_t err = _axis->setAwoff(on);
  if (err == AXIS_ERR_NOHW) { reply("ERR:NOHW"); return; }
  reply("OK:AWOFF:%d", on ? 1 : 0);
}

void Protocol::hAwoffQ(const char *arg) {
  (void)arg;
  reply("AWOFF:%d", _axis->awoffActive() ? 1 : 0);
}

void Protocol::hCutb(const char *arg) {
  if (strcmp(arg, "0") != 0 && strcmp(arg, "1") != 0) { reply("ERR:BAD_FORMAT"); return; }
  bool on = (arg[0] == '1');
  int8_t err = _axis->setCutback(on);
  if (err == AXIS_ERR_NOHW) { reply("ERR:NOHW"); return; }
  reply("OK:CUTB:%d", on ? 1 : 0);
}

void Protocol::hCutbQ(const char *arg) {
  (void)arg;
  reply("CUTB:%d", _axis->cutbackActive() ? 1 : 0);
}

void Protocol::hCfg(const char *arg) {
  if (strncmp(arg, "ACC:", 4) == 0) {
    int32_t v;
    if (!parseInt(arg + 4, v) || v < 1) { reply("ERR:BAD_VALUE:%s", arg + 4); return; }
    _axis->setAccel((uint32_t)v);
    _settings->accel = _axis->accel();
    _settings->save();
    reply("OK:CFG:ACC:%lu", (unsigned long)_axis->accel());
  } else if (strncmp(arg, "MAX:", 4) == 0) {
    int32_t v;
    if (!parseInt(arg + 4, v) || v <= 0) { reply("ERR:BAD_VALUE:%s", arg + 4); return; }
    _axis->setMaxSpeed((uint32_t)v);
    _settings->maxSpeed = _axis->maxSpeed();
    _settings->save();
    reply("OK:CFG:MAX:%lu", (unsigned long)_axis->maxSpeed());
  } else if (strncmp(arg, "TMO:", 4) == 0) {
    int32_t v;
    if (!parseInt(arg + 4, v) || v < 0) { reply("ERR:BAD_VALUE:%s", arg + 4); return; }
    uint32_t tmo = constrain((uint32_t)v, 0UL, 60000UL);
    _settings->tmoMs = tmo;
    _settings->save();
    reply("OK:CFG:TMO:%lu", (unsigned long)tmo);
  } else {
    reply("ERR:BAD_FORMAT");
  }
}

void Protocol::hCfgQ(const char *arg) {
  (void)arg;
  reply("CFG:MIN:%lu,MAX:%lu,ACC:%lu,TMO:%lu",
        (unsigned long)MIN_SPEED_SPS, (unsigned long)_axis->maxSpeed(),
        (unsigned long)_axis->accel(), (unsigned long)_settings->tmoMs);
}
