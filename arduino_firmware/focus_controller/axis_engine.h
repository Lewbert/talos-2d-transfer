// ============================================================================
// StepAxis - non-blocking CW/CCW pulse engine for the focus axis
// ============================================================================
// Generates a square wave on the active pin (CW or CCW) via micros() compares.
// The driver steps on the FALLING edge (negative logic), so position counting
// happens on the falling edge of each full pulse. No delay() is ever used.
#ifndef FOCUS_AXIS_ENGINE_H
#define FOCUS_AXIS_ENGINE_H

#include <Arduino.h>

enum AxisMode : uint8_t {
  MODE_IDLE = 0,
  MODE_CONT,   // continuous (gamepad) motion
  MODE_TRAP,   // trapezoidal point-to-point move
  MODE_LIMIT   // stopped at a software limit
};

// Axis error codes (mirrored in the PC protocol)
enum AxisError : int8_t {
  AXIS_OK = 0,
  AXIS_ERR_BUSY,
  AXIS_ERR_LIMIT,
  AXIS_ERR_RANGE,
  AXIS_ERR_NOHW,
  AXIS_ERR_BADVAL
};

// Pending axis events (drained and printed by the protocol layer)
enum AxisEventType : uint8_t {
  EV_NONE = 0,
  EV_STOP,      // motion aborted (STOP / AWOFF / serial TMO)
  EV_DONE,      // point-to-point move completed
  EV_LIM        // stopped at software limit (direction sign in arg2)
};

struct AxisEvent {
  uint8_t type;
  int32_t arg1;
  int8_t arg2;
};

struct AxisConfig {
  uint8_t cwPin;
  uint8_t ccwPin;
  uint8_t hasAwoff;
  uint8_t awoffPin;
  uint8_t awoffActiveLow;
  uint8_t hasCdin;
  uint8_t cdinPin;
  uint8_t cdinActiveLow;
  uint32_t maxSpeed;      // clamped to [MIN_SPEED_SPS, HARD_MAX_SPEED_SPS]
  uint32_t accel;         // clamped to [1, HARD_MAX_ACCEL_SPS2]
  uint32_t mvSpeed;       // point-to-point cruise speed (runtime adjustable)
  uint8_t slimOn;
  int32_t slimMin, slimMax;
};

class StepAxis {
public:
  void begin(const AxisConfig &cfg);

  // --- Motion commands ---
  // Continuous: signed target speed in steps/s (0 = ramp stop). The applied
  // (clamped) target is returned in `applied`.
  int8_t setContinuous(int32_t targetSps, int32_t &applied);
  // Point-to-point: relative move with trapezoidal profile at mvSpeed.
  int8_t startMove(int32_t relSteps);
  // Immediate abort (no ramp). Emits EV_STOP if moving.
  void stopNow();

  // --- Software limits ---
  int8_t setSlim(bool on);                    // AXIS_ERR_RANGE if pos outside bounds
  int8_t setSlimBounds(int32_t mn, int32_t mx); // sets + enables; AXIS_ERR_RANGE on bad bounds
  void getSlim(int32_t &mn, int32_t &mx) const { mn = _slimMin; mx = _slimMax; }
  bool slimEnabled() const { return _slimOn != 0; }

  // --- Hardware aux outputs ---
  int8_t setAwoff(bool on);     // AXIS_ERR_NOHW; on = free shaft (stops motion first)
  int8_t setCutback(bool on);   // AXIS_ERR_NOHW; on = release current cutback
  bool awoffActive() const { return _awoffState != 0; }
  bool cutbackActive() const { return _cdinState != 0; }

  // --- Settings ---
  void setMaxSpeed(uint32_t sps);  // clamped, runtime
  void setAccel(uint32_t a);       // clamped, runtime
  void setMvSpeed(uint32_t sps);   // clamped, runtime-only
  uint32_t maxSpeed() const { return _maxSpeed; }
  uint32_t accel() const { return _accel; }
  uint32_t mvSpeed() const { return _mvSpeed; }

  // --- Status ---
  AxisMode mode() const { return _mode; }
  int32_t position() const { return _pos; }
  int32_t currentSpeed() const { return _dir * (int32_t)lroundf(_v); }
  int32_t targetSpeed() const;
  int8_t blockedDir() const { return (_mode == MODE_LIMIT) ? _blockedDir : 0; }
  bool isMoving() const { return _mode == MODE_CONT || _mode == MODE_TRAP; }

  // --- Position ----
  bool zeroPosition();   // IDLE/LIMIT only

  // --- Events ---
  bool popEvent(AxisEvent &e);

  // --- Per-loop update (call every loop() iteration) ---
  void update();

private:
  enum TrapPhase : uint8_t { PHASE_ACCEL = 0, PHASE_CRUISE, PHASE_DECEL };

  // Config
  uint8_t _cwPin, _ccwPin;
  uint8_t _hasAwoff, _awoffPin, _awoffActiveLow;
  uint8_t _hasCdin, _cdinPin, _cdinActiveLow;
  uint32_t _maxSpeed, _accel, _mvSpeed;

  // Motion state
  AxisMode _mode;
  TrapPhase _trapPhase;
  int32_t _pos;
  float _v;               // speed magnitude (steps/s)
  int8_t _dir;            // active pin selector: +1 = CW, -1 = CCW, 0 = none
  int8_t _pendingDir;     // CONT: direction to resume after ramp-to-zero sign flip
  int32_t _targetSps;     // CONT: signed target speed
  uint32_t _halfPeriodUs;
  uint32_t _lastToggleUs;
  uint8_t _pinLevel;      // current level of the active pulse pin
  uint32_t _holdUntilMs;  // block first pulse until then (A.W.OFF re-engage settle)

  // Trap state
  int32_t _stepsRemain;
  int32_t _nAccel, _nCruise, _nDecel;
  int32_t _phaseSteps;
  float _vPeak;

  // Software limits
  uint8_t _slimOn;
  int32_t _slimMin, _slimMax;
  int8_t _blockedDir;

  // Aux outputs
  uint8_t _awoffState, _cdinState;

  // Event ring buffer
  static const uint8_t EVENT_QUEUE = 8;
  AxisEvent _events[EVENT_QUEUE];
  uint8_t _evHead, _evTail;

  void pushEvent(uint8_t type, int32_t arg1, int8_t arg2);
  void enterLimit(int8_t dir);
  void contKinematics();
  void trapKinematics();
  void recomputeHalfPeriod();
  void recomputeBlocked();
  void kinematics();
  uint8_t activePin() const { return _dir > 0 ? _cwPin : _ccwPin; }
};

#endif // FOCUS_AXIS_ENGINE_H
