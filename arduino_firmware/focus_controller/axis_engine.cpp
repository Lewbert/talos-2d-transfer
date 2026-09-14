// ============================================================================
// StepAxis implementation - see axis_engine.h for the design notes.
// ============================================================================
#include "axis_engine.h"
#include "config.h"

#include <math.h>

// Direct PORTB access for D11/D12 (ATmega328P PB3/PB4) removes ~4us of
// digitalWrite() overhead per edge. Falls back to digitalWrite() elsewhere.
static inline void pulseWrite(uint8_t pin, uint8_t level) {
#if defined(__AVR_ATmega328P__) || defined(__AVR_ATmega328__)
  if (pin == 11) {
    if (level) PORTB |= _BV(3); else PORTB &= ~_BV(3);
    return;
  }
  if (pin == 12) {
    if (level) PORTB |= _BV(4); else PORTB &= ~_BV(4);
    return;
  }
#endif
  digitalWrite(pin, level);
}

void StepAxis::begin(const AxisConfig &cfg) {
  _cwPin = cfg.cwPin; _ccwPin = cfg.ccwPin;
  _hasAwoff = cfg.hasAwoff; _awoffPin = cfg.awoffPin; _awoffActiveLow = cfg.awoffActiveLow;
  _hasCdin = cfg.hasCdin; _cdinPin = cfg.cdinPin; _cdinActiveLow = cfg.cdinActiveLow;
  _maxSpeed = cfg.maxSpeed; _accel = cfg.accel; _mvSpeed = cfg.mvSpeed;

  pinMode(_cwPin, OUTPUT);  pulseWrite(_cwPin, LOW);
  pinMode(_ccwPin, OUTPUT); pulseWrite(_ccwPin, LOW);
  if (_hasAwoff) {
    pinMode(_awoffPin, OUTPUT);
    digitalWrite(_awoffPin, _awoffActiveLow ? HIGH : LOW);  // off = engaged (normal)
  }
  if (_hasCdin) {
    pinMode(_cdinPin, OUTPUT);
    digitalWrite(_cdinPin, _cdinActiveLow ? HIGH : LOW);    // off = cutback active (normal)
  }

  _mode = MODE_IDLE;
  _trapPhase = PHASE_ACCEL;
  _pos = 0; _v = 0.0f; _dir = 0; _pendingDir = 0;
  _targetSps = 0;
  _halfPeriodUs = HALF_PERIOD_MAX_US;
  _lastToggleUs = 0; _pinLevel = 0; _holdUntilMs = 0;
  _stepsRemain = 0; _nAccel = 0; _nCruise = 0; _nDecel = 0; _phaseSteps = 0; _vPeak = 0.0f;
  _slimOn = cfg.slimOn ? 1 : 0;
  _slimMin = cfg.slimMin; _slimMax = cfg.slimMax;
  if (_slimOn && (_pos < _slimMin || _pos > _slimMax)) _slimOn = 0;  // stale EEPROM bounds
  _blockedDir = 0;
  _awoffState = 0; _cdinState = 0;
  _evHead = 0; _evTail = 0;
}

// ---------------------------------------------------------------------------
// Motion commands
// ---------------------------------------------------------------------------

int8_t StepAxis::setContinuous(int32_t targetSps, int32_t &applied) {
  // Clamp magnitude (0 = stop passes through unchanged)
  if (targetSps != 0) {
    int32_t mag = targetSps > 0 ? targetSps : -targetSps;
    if (mag < (int32_t)MIN_SPEED_SPS) mag = MIN_SPEED_SPS;
    if (mag > (int32_t)_maxSpeed) mag = _maxSpeed;
    targetSps = (targetSps > 0) ? mag : -mag;
  }
  applied = targetSps;

  if (_awoffState) return AXIS_ERR_BUSY;  // windings off: refuse motion

  switch (_mode) {
    case MODE_TRAP:
      return AXIS_ERR_BUSY;
    case MODE_LIMIT: {
      if (targetSps == 0) return AXIS_OK;  // already stopped
      int8_t d = (targetSps > 0) ? 1 : -1;
      if (d == _blockedDir) return AXIS_ERR_LIMIT;
      _mode = MODE_CONT;                   // moving away unblocks
      _blockedDir = 0;
      break;
    }
    case MODE_IDLE:
      if (targetSps == 0) return AXIS_OK;
      _mode = MODE_CONT;
      break;
    case MODE_CONT:
      break;
  }

  _targetSps = targetSps;
  if (targetSps != 0) {
    int8_t tdir = (targetSps > 0) ? 1 : -1;
    if (_dir == 0) {
      _dir = tdir;
      _v = 0.0f;
      _lastToggleUs = micros();
      _halfPeriodUs = HALF_PERIOD_MAX_US;
    } else if (tdir != _dir) {
      _pendingDir = tdir;   // ramp to zero, then swap pin
    } else {
      _pendingDir = 0;
    }
  } else {
    _pendingDir = 0;        // ramp stop handled in contKinematics()
  }
  return AXIS_OK;
}

int8_t StepAxis::startMove(int32_t relSteps) {
  if (relSteps == 0) return AXIS_OK;
  if (relSteps == INT32_MIN) return AXIS_ERR_BADVAL;  // negation overflow guard
  if (_mode == MODE_CONT || _mode == MODE_TRAP) return AXIS_ERR_BUSY;
  if (_awoffState) return AXIS_ERR_BUSY;

  int64_t target = (int64_t)_pos + relSteps;
  if (target > INT32_MAX || target < INT32_MIN) return AXIS_ERR_RANGE;

  if (_slimOn) {
    if (target < _slimMin || target > _slimMax) return AXIS_ERR_LIMIT;
  }

  int8_t dir = (relSteps > 0) ? 1 : -1;
  int32_t n = relSteps > 0 ? relSteps : -relSteps;

  // Trapezoidal profile: nAcc = V^2/2A, symmetric unless triangular
  float vCruise = (float)_mvSpeed;
  float twoA = 2.0f * (float)_accel;
  int32_t nAcc = (int32_t)((vCruise * vCruise) / twoA);
  if (nAcc < 1) nAcc = 1;
  if ((int64_t)nAcc * 2 > n) {
    // Move too short for full cruise: triangular profile
    nAcc = n / 2;
    _nDecel = n - nAcc;
    _nCruise = 0;
    _vPeak = sqrtf(twoA * (float)nAcc);
  } else {
    _nDecel = nAcc;
    _nCruise = n - 2 * nAcc;
    _vPeak = vCruise;
  }
  _nAccel = nAcc;
  _stepsRemain = n;
  _phaseSteps = 0;
  _trapPhase = (_nAccel > 0) ? PHASE_ACCEL : ((_nCruise > 0) ? PHASE_CRUISE : PHASE_DECEL);

  _dir = dir;
  _v = 0.0f;
  _pinLevel = 0;
  _pendingDir = 0;
  _mode = MODE_TRAP;
  _lastToggleUs = micros();
  recomputeHalfPeriod();
  return AXIS_OK;
}

void StepAxis::stopNow() {
  bool wasMoving = (_mode == MODE_CONT || _mode == MODE_TRAP);
  pulseWrite(_cwPin, LOW);
  pulseWrite(_ccwPin, LOW);
  _pinLevel = 0;
  _v = 0.0f;
  _dir = 0;
  _pendingDir = 0;
  _targetSps = 0;
  if (_mode != MODE_LIMIT) _mode = MODE_IDLE;
  if (wasMoving) pushEvent(EV_STOP, _pos, 0);
}

bool StepAxis::zeroPosition() {
  if (_mode == MODE_CONT || _mode == MODE_TRAP) return false;
  _pos = 0;
  recomputeBlocked();
  return true;
}

// ---------------------------------------------------------------------------
// Software limits
// ---------------------------------------------------------------------------

int8_t StepAxis::setSlim(bool on) {
  if (on && (_pos < _slimMin || _pos > _slimMax)) return AXIS_ERR_RANGE;
  _slimOn = on ? 1 : 0;
  recomputeBlocked();
  return AXIS_OK;
}

int8_t StepAxis::setSlimBounds(int32_t mn, int32_t mx) {
  if (mn >= mx) return AXIS_ERR_RANGE;
  if (_pos < mn || _pos > mx) return AXIS_ERR_RANGE;
  _slimMin = mn; _slimMax = mx;
  _slimOn = 1;
  recomputeBlocked();
  return AXIS_OK;
}

void StepAxis::recomputeBlocked() {
  if (_slimOn) {
    if (_pos >= _slimMax) _blockedDir = 1;
    else if (_pos <= _slimMin) _blockedDir = -1;
    else _blockedDir = 0;
  } else {
    _blockedDir = 0;
  }
  if (_mode == MODE_LIMIT && _blockedDir == 0) _mode = MODE_IDLE;
}

// ---------------------------------------------------------------------------
// Hardware aux outputs
// ---------------------------------------------------------------------------

int8_t StepAxis::setAwoff(bool on) {
  if (!_hasAwoff) return AXIS_ERR_NOHW;
  if (on) stopNow();  // only free the shaft from standstill
  _awoffState = on ? 1 : 0;
  digitalWrite(_awoffPin, (on != (_awoffActiveLow != 0)) ? LOW : HIGH);
  if (!on) _holdUntilMs = millis() + 100UL;  // driver needs >=0.1s settle before pulses
  return AXIS_OK;
}

int8_t StepAxis::setCutback(bool on) {
  if (!_hasCdin) return AXIS_ERR_NOHW;
  _cdinState = on ? 1 : 0;
  digitalWrite(_cdinPin, (on != (_cdinActiveLow != 0)) ? LOW : HIGH);
  return AXIS_OK;
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

void StepAxis::setMaxSpeed(uint32_t sps) {
  _maxSpeed = constrain(sps, MIN_SPEED_SPS, HARD_MAX_SPEED_SPS);
}

void StepAxis::setAccel(uint32_t a) {
  _accel = constrain(a, 1UL, HARD_MAX_ACCEL_SPS2);
}

void StepAxis::setMvSpeed(uint32_t sps) {
  _mvSpeed = constrain(sps, MIN_SPEED_SPS, _maxSpeed);
}

int32_t StepAxis::targetSpeed() const {
  switch (_mode) {
    case MODE_CONT: return _targetSps;
    case MODE_TRAP: return _dir * (int32_t)_vPeak;
    default: return 0;
  }
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

void StepAxis::pushEvent(uint8_t type, int32_t arg1, int8_t arg2) {
  uint8_t next = (_evHead + 1) % EVENT_QUEUE;
  if (next == _evTail) {          // full: drop the oldest, keep the newest
    _evTail = (_evTail + 1) % EVENT_QUEUE;
  }
  _events[_evHead].type = type;
  _events[_evHead].arg1 = arg1;
  _events[_evHead].arg2 = arg2;
  _evHead = next;
}

bool StepAxis::popEvent(AxisEvent &e) {
  if (_evHead == _evTail) return false;
  e = _events[_evTail];
  _evTail = (_evTail + 1) % EVENT_QUEUE;
  return true;
}

void StepAxis::enterLimit(int8_t dir) {
  pulseWrite(_cwPin, LOW);
  pulseWrite(_ccwPin, LOW);
  _pinLevel = 0;
  _v = 0.0f;
  _dir = 0;
  _pendingDir = 0;
  _targetSps = 0;
  _mode = MODE_LIMIT;
  _blockedDir = dir;
  pushEvent(EV_LIM, _pos, dir);
}

// ---------------------------------------------------------------------------
// Pulse engine
// ---------------------------------------------------------------------------

void StepAxis::update() {
  if (_mode != MODE_CONT && _mode != MODE_TRAP) return;
  if (_holdUntilMs) {
    if ((int32_t)(millis() - _holdUntilMs) < 0) return;  // A.W.OFF re-engage settle
    _holdUntilMs = 0;
  }

  uint32_t now = micros();
  if ((uint32_t)(now - _lastToggleUs) < _halfPeriodUs) return;
  _lastToggleUs = now;

  if (_pinLevel) {
    // Falling edge: the driver steps here (negative logic)
    pulseWrite(activePin(), LOW);
    _pinLevel = 0;
    _pos += _dir;
    if (_mode == MODE_TRAP) _stepsRemain--;
    kinematics();
    recomputeHalfPeriod();
  } else {
    // Rising edge: software limit check first
    if (_slimOn) {
      if (_dir > 0 && _pos + 1 > _slimMax) { enterLimit(1); return; }
      if (_dir < 0 && _pos - 1 < _slimMin) { enterLimit(-1); return; }
    }
    pulseWrite(activePin(), HIGH);
    _pinLevel = 1;
  }
}

void StepAxis::kinematics() {
  if (_mode == MODE_CONT) contKinematics();
  else if (_mode == MODE_TRAP) trapKinematics();
}

// Continuous mode: slew-limited speed toward the signed target using
// v^2 += 2A per step (exact discrete integration). Sign flips go through
// a ramp-to-zero so the pins are never switched at speed.
void StepAxis::contKinematics() {
  int32_t t = _targetSps;
  float twoA = 2.0f * (float)_accel;

  if (_dir == 0) {  // defensive; normally set at command time
    _dir = (t > 0) ? 1 : -1;
    _v = 0.0f;
    return;
  }

  if (_pendingDir != 0) {
    _v = sqrtf(fmaxf(_v * _v - twoA, 0.0f));
    if (_v * _v <= twoA) {  // reached zero: swap direction pin
      _v = 0.0f;
      _dir = _pendingDir;
      _pendingDir = 0;
    }
  } else if (t == 0) {
    _v = sqrtf(fmaxf(_v * _v - twoA, 0.0f));
    if (_v * _v <= twoA) {  // ramp-stop complete
      _v = 0.0f;
      _dir = 0;
      _mode = MODE_IDLE;
    }
  } else if ((t > 0) == (_dir > 0)) {
    float tt = fabsf((float)t);
    if (fabsf(_v * _v - tt * tt) <= twoA) _v = tt;      // snap to target
    else _v = sqrtf(fminf(_v * _v + twoA, tt * tt));     // accelerate
  } else {
    _pendingDir = (t > 0) ? 1 : -1;                      // decelerate, then swap
    _v = sqrtf(fmaxf(_v * _v - twoA, 0.0f));
    if (_v * _v <= twoA) {
      _v = 0.0f;
      _dir = _pendingDir;
      _pendingDir = 0;
    }
  }
}

// Trapezoidal point-to-point: speed affects timing only, never the step
// count, so the move lands exactly.
void StepAxis::trapKinematics() {
  float twoA = 2.0f * (float)_accel;

  if (_stepsRemain <= 0) {  // final step emitted: done
    _v = 0.0f;
    _dir = 0;
    _mode = MODE_IDLE;
    pushEvent(EV_DONE, _pos, 0);
    return;
  }

  switch (_trapPhase) {
    case PHASE_ACCEL:
      if (_stepsRemain <= _nDecel) {  // vPeak not reached: go straight to decel
        _trapPhase = PHASE_DECEL;
        _phaseSteps = 0;
        _v = sqrtf(fmaxf(_v * _v - twoA, 0.0f));
      } else {
        _v = fminf(sqrtf(_v * _v + twoA), _vPeak);
        _phaseSteps++;
        if (_phaseSteps >= _nAccel) { _trapPhase = PHASE_CRUISE; _phaseSteps = 0; }
      }
      break;
    case PHASE_CRUISE:
      _v = _vPeak;
      _phaseSteps++;
      if (_phaseSteps >= _nCruise) { _trapPhase = PHASE_DECEL; _phaseSteps = 0; }
      break;
    case PHASE_DECEL:
      _v = sqrtf(fmaxf(_v * _v - twoA, 0.0f));
      break;
  }
}

void StepAxis::recomputeHalfPeriod() {
  float vt = _v;
  if (vt < MIN_SPEED_SPS) vt = MIN_SPEED_SPS;
  uint32_t hp = (uint32_t)(500000UL / vt);  // half period = 1e6 / (2 * sps)
  if (hp < HALF_PERIOD_MIN_US) hp = HALF_PERIOD_MIN_US;
  if (hp > HALF_PERIOD_MAX_US) hp = HALF_PERIOD_MAX_US;
  _halfPeriodUs = hp;
}
