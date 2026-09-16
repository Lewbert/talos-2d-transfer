# FocusControl Serial Protocol

Line-based ASCII over 115200 8N1, newline (`\n`) terminated, commands are
case-insensitive. Every reply is a single line ≤ 64 chars. The firmware
never blocks: stepping is a non-blocking `micros()` pulse engine and the
serial parser processes at most 8 chars per loop pass.

Firmware: `arduino_firmware/focus_controller` (Arduino Uno/Nano, CRD5103PB driver).
Host-side driver: `talos/hal/devices/focus.py`. How the autofocus algorithm uses
this axis: [AUTOFOCUS.md](../../AUTOFOCUS.md); the bench it belongs to:
[DESIGN.md](../../DESIGN.md).

Line discipline: bytes outside the command alphabet `[A-Za-z0-9?:,+-]` (e.g.
USB re-enumeration noise) are discarded and act as a framing break, and a
partial line silent for 50 ms is dropped — junk can never fold into a
command. Hosts may send commands at any rate.

## Boot banner

On power-up / reset:

```
BOOT:FOCUSCTRL:1.0
READY
```

After a **watchdog reset** the first line is `EV:WDT` instead.

## Sign convention

- **+speed / +position** = pulses on the **CW** pin (D11)
- **−speed / −position** = pulses on the **CCW** pin (D12)
- The driver steps on the **falling edge** of each pulse (negative logic);
  position counting happens on the falling edge, one count per full pulse.

## Commands

| Command | Reply | Semantics |
|---|---|---|
| `PING` | `PONG` | liveness |
| `VER?` | `VER:FOCUSCTRL:1.0` | firmware identity |
| `STATUS?` | `S:POS:<p>,MODE:<m>,V:<v>,SPD:<t>,LIM:<b>,SLIM:<o>` | polled by the GUI at 5 Hz |
| `SPD:<spd>` | `OK:SPD:<applied>` | signed target, steps/s. 0 = ramp stop. Magnitude clamped to `[10, maxSpeed]` (reply = applied). From IDLE starts continuous mode. During a point-to-point move → `ERR:BUSY`. Into the blocked limit direction → `ERR:LIMIT`. |
| `MOVE:<rel>` | `OK:MOVE:<rel>` | relative trapezoidal move at `MVSPD`; exact landing |
| `GOTO:<abs>` | `OK:GOTO:<abs>` | absolute trapezoidal move (computed relative internally) |
| `STOP` | `OK:STOP` | immediate abort (always legal); `EV:STOP:<pos>` if moving |
| `ZERO` | `OK:ZERO` | position := 0 (IDLE/LIMIT only, else `ERR:BUSY`) |
| `MVSPD:<spd>` | `OK:MVSPD:<n>` | trap cruise speed, runtime-only, clamp `[10, maxSpeed]` |
| `MVSPD?` | `MVSPD:<n>` | |
| `SLIM:<0\|1>` | `OK:SLIM:<o>:<min>:<max>` | software limits on/off; enable with position outside bounds → `ERR:RANGE`; persisted |
| `SLIM:SET:<min>:<max>` | `OK:SLIM:1:<min>:<max>` | set bounds + enable; `min>=max` or position outside → `ERR:RANGE`; persisted |
| `SLIM?` | `SLIM:<o>:<min>:<max>` | |
| `AWOFF:<0\|1>` | `OK:AWOFF:<0\|1>` | 1 = release windings (free shaft) — stops motion first. `ERR:NOHW` if CN2 A.W.OFF not wired. Motion commands refused while active (`ERR:BUSY`) to protect position bookkeeping. |
| `AWOFF?` | `AWOFF:<0\|1>` | |
| `CUTB:<0\|1>` | `OK:CUTB:<0\|1>` | 1 = release the automatic current cutback (full holding torque). `ERR:NOHW` if CN2 C.D.INH not wired. |
| `CUTB?` | `CUTB:<0\|1>` | |
| `CFG?` | `CFG:MIN:<n>,MAX:<n>,ACC:<n>,TMO:<ms>` | |
| `CFG:ACC:<a>` | `OK:CFG:ACC:<a>` | accel, steps/s², clamp `[1, 100000]`; persisted |
| `CFG:MAX:<m>` | `OK:CFG:MAX:<m>` | max speed, clamp `[10, 5000]`; persisted |
| `CFG:TMO:<ms>` | `OK:CFG:TMO:<ms>` | serial inactivity stop, clamp `[0, 60000]`; 0 = off; persisted |

`STATUS?` fields: `p` position (steps), `m` mode ∈ `IDLE/CONT/TRAP/LIMIT`,
`v` current signed speed (steps/s), `t` target speed (CONT) or trap cruise
(TRAP), `b` blocked limit direction ∈ `0/+/-` (LIMIT state only), `o` SLIM
on/off.

## Errors

`ERR:UNKNOWN:<cmd>` · `ERR:BAD_FORMAT` · `ERR:BAD_VALUE:<arg>` ·
`ERR:BUSY` · `ERR:LIMIT` · `ERR:RANGE` · `ERR:NOHW` · `ERR:OVERFLOW`
(command longer than 64 chars).

## Events (unsolicited)

| Event | Meaning |
|---|---|
| `EV:STOP:<pos>` | motion aborted by STOP / A.W.OFF / serial timeout |
| `EV:DONE:<pos>` | point-to-point move completed (exact landing) |
| `EV:LIM:<+->:<pos>` | stopped at a software limit; that direction is blocked until the axis moves away or limits are cleared |
| `EV:TMO:<pos>` | no serial traffic for `CFG:TMO` ms while moving → auto stop (PC unplug safety) |
| `EV:WDT` | firmware restarted after a watchdog reset |

## Motion model

- **Continuous (gamepad):** target speed slew-limited with exact discrete
  integration `v² += 2A` per step, snap to target within `2A`; direction
  changes ramp through zero before the pulse pin swaps.
- **Point-to-point:** trapezoidal profile computed from step counts —
  speed affects timing only, never the count, so every move lands exactly.
  Short moves degrade to a triangular profile; the final step is forced.
- **Timing:** half-period `1e6/(2·v)` µs clamped to `[100, 50000]` µs
  (10–5000 steps/s). At the default 500 steps/rev and 2000 steps/s the
  axis does 4 rev/s.
- **Watchdogs:** AVR hardware watchdog (2 s, fed every 500 ms) + serial
  inactivity stop (default 5 s).

## EEPROM persistence

Stored (magic-validated, written only on `CFG:`/`SLIM:` commands): SLIM
on/bounds, accel, max speed, TMO. Not stored: position, mode, speed,
MVSPD.
