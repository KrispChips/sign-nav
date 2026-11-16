# Step 1: goal-directed sign guidance (spec)

Status: agreed design, not implemented yet. This document is the spec; examples in it are
examples only. Nothing here may be specialised to one building, one campus or one test walk.

## Purpose

sign-nav helps a blind pedestrian reach a destination inside a building they have never
been in, using only what any sighted visitor uses: the building's signs, plus the hallway
layout seen through the camera. Today the system reads every legible sign aloud. Step 1 gives
it a **goal** (a room or a named place) and makes it speak **only what helps reach that goal**,
note everything else silently, and let the user ask for any other sign on demand.

It must work in **any building**, with whatever room-numbering convention that building uses,
without per-building setup. Later steps build on it: step 2 adds a small topological memory
(nodes = junctions, entrances, signs; edges = walked corridors with heading) that carries
instructions to the next junction; that is out of scope here, but step 1 must record what
step 2 needs (see "Noted, not spoken").

## The user stays in charge (added 2026-11-15)

sign-nav assists; it does not decide. The people using it know things the system can't see
and the choice of where to walk is always theirs.
- **Ask for what they know.** After the goal is set, ask once, briefly, whether they know
  anything that helps: which way the goal is, how the building is laid out, a shorter route,
  a route they usually take. "Nothing" is a fine answer. Keep what they say as context for
  the session (logged with the decisions), and let them add to it any time.
- **Their knowledge counts.** If a sign disagrees with what they told us, say both and let
  them choose ("You said it's to the right; this sign points left for 014-190.") — never
  silently override them, and never repeat the disagreement once they've chosen.
- **Information, not orders.** Say what the sign says and where it is ("014-190: sign on
  your left points right"), not "turn right".

## Scope

In scope:
- parsing a goal written in any common room-numbering style;
- interpreting room numbers and ranges on signs relative to that goal;
- deciding, per sign read, what to say (spoken / asked / noted / arrival);
- a user input ("tell me more") from the glasses, and from the phone app as a fallback;
- logging every decision.

Out of scope (later steps): memory/map and route planning across junctions, a learned
"glance" classifier that decides which signs are worth reading, priors about building layout,
setting the goal by voice, reading posted building maps (directory maps, fire-evacuation
plans) to learn the layout.

## Existing pieces to build on

| module | role today |
|---|---|
| `targets.py` | goal parsing (`parse_target`), room IDs, optional per-building YAML rules in `config/buildings/` |
| `entries.py` | normalises and parses one sign entry (room, range, floor, building, area, facility, named) |
| `matcher.py` | `match_sign(target, entries, rule, ocr_text, context)`: match level, arrow, confidence, contradictions, ambiguity |
| `facilities.py` | facility names (lift, stairs, toilets, exit, ...) |
| `direction.py` | arrow + where the sign is + wall/facing mount -> words for the user ("to your left") |
| `main.py` | the loop; reads are spoken by `Announcer`; reads are already OCR cross-checked |
| `live.py`, `android/`, `docs/LIVE.md` | live link: phone app <-> PC (frames, speech, head yaw) |

The parser and matcher exist and are tested, but they rely on per-building YAML rules and are
not wired into `main.py`. Step 1 makes them building-agnostic and wires them in.

## 1. Goal: any room-numbering convention

The user gives the goal as they would write it. The goal's written form is the strongest clue
to how this building numbers rooms, so it is parsed into a **shape** (letters, digit groups,
separators, suffix), and signs are later read through that shape.

Conventions that must be handled (illustrative examples, not an exhaustive list):

| convention | example goals | parts |
|---|---|---|
| building-room | `014-190`, `14-100`, `B14-100` | building (leading zeros ignored), room |
| lettered wing/block | `A140`, `C2.15` | wing/block letter, room |
| plain number | `120`, `1200`, `120B` | room (with suffix letter) |
| mixed | `B2-105`, `2.105`, `L3-07` | wing / level / section, room |
| a name | "Chemistry Lab", "Registrar" | a named place: fuzzy name match |

Floors: "the first digit is the floor" is common but not universal (`1200` could be floor 1 or
12). A floor inferred from a number is a **hint only**: it may break ties or suggest a lift, it
must never rule a sign out. A floor stated in words ("Level 2", "2nd floor", "L2") counts fully.

Per-building YAML rules stay as optional overrides for odd conventions; none is required.

## 2. Reading signs through the goal's shape

Every room ID and range on a sign is parsed generically, then compared with the goal:

- **Same shape**: compare directly; a range of the same shape containing the goal is a match
  (e.g. goal `A140`, sign `A121 - A159`).
- **Ambiguity is resolved by the goal's shape.** E.g. `14-100` on a sign is "building 14,
  room 100" when the goal looks like `014-190`, but "range 14 to 100" when the goal is a plain
  number. Without a hint: different digit counts suggest building-room, equal counts a range.
- **Partial IDs**: a plate showing only part of the goal (e.g. `190` for goal `014-190`) is a
  match with lower confidence, raised to full when the missing part is established (a building
  sign or earlier plates in the same building).
- **Different prefix**: a different wing/building letter or number is not a match (`B140` is not
  `A140`); a missing prefix is a weak match.
- Ranges: "-", "–", "to", "thru", optional odd/even qualifiers; existing matcher logic applies.
- Building-level and area-level matches ("Building 14 →", "Wing A →") are useful when the user
  is still outside the goal's building or wing.

## 3. What gets said

Each OCR-confirmed sign read gets exactly one decision:

| situation | decision | example output |
|---|---|---|
| sign names the goal, with an arrow | **spoken** | "014-190: to your left." |
| the goal's own door plate (no arrow) | **arrival**: spoken, guidance stops | "You've arrived: 014-190, on your right." |
| range / building / wing / floor containing the goal, with an arrow | **spoken** | "Rooms A121 to A159, to your right. That includes A140." |
| lift or stairs, goal on another known floor | **spoken** | "Lift, to your left. Room 1200 is on floor 12." |
| matching entries point different ways | **spoken** | "Sign unclear for A140: one arrow says left, one says right." |
| a sign with content but nothing about the goal | **asked** | "Sign on your left. Tap to hear it." |
| safety signs (EXIT, fire exit, stairs) not relevant to the goal | **noted** (not spoken, not asked) | - |
| door plates of other rooms | **noted** | - |
| nothing legible / not confirmed by OCR | as today (not spoken) | - |

(The example outputs use different goals on purpose; the wording pattern is what matters.)

Rules:
- **Asking**: only for signs with real content (several entries or an arrow); never door
  plates or safety signs; at most once per sign; one pending question at a time; a question
  expires after ~10 s; a new spoken instruction always takes priority over a pending question.
- **No repeats**: keep one *current instruction* (target, direction, source sign, time). Speak
  again only when it changes or becomes more specific (building -> wing -> range -> exact).
- Direction words come from `direction.py` (sign position, wall/facing mount, user heading).
- `--announce all` restores today's behaviour (read every sign).

### Noted, not spoken
Noted items are recorded with time, sign position, the user's heading and the parsed content:
safety signs, other rooms' plates, unasked or unanswered signs. Step 2 uses them: e.g. the
sequence of room numbers along a corridor (rising or falling) hints whether the user is
walking toward or away from the goal. That is a hint only; numbering is not always monotonic.

## 4a. Talking to it: voice commands first (added 2026-11-15)

The main input is a short spoken conversation; glasses gestures and the phone's "Read it"
button stay as extra ways to send the same commands. One command set, three inputs:

| command (examples of what the user says) | effect |
|---|---|
| "read it", "what does it say", "yes" (to "Tap to hear it") | read the pending / last sign |
| "what was that", "repeat" | repeat the current instruction |
| "I usually go left here", "the lab is on the north side", "nothing" | add to the user's context |
| "stop guiding" | end the goal (setting a goal by voice stays a later step) |

Speech recognition runs on the phone (the glasses' microphones over Bluetooth) and the
recognised text is sent to the PC on the two-way link of section 4. Questions to the user
("Tap to hear it" becomes "Want to hear it?") accept a spoken answer as well as a gesture.
Unrecognised speech is not guessed at: "Sorry, I didn't catch that." Listening is always on
(paused while the phone speaks), so it also hears other people: unless a question was just
asked, speech that isn't a clear command or the user's own route knowledge ("I usually go
left", "it's on the right") is logged and ignored, not answered. (Implemented: phone
`Voice.kt`, PC `commands.py` / `guide.py`; protocol in `docs/LIVE.md`.)

## 4. "Tell me more": user input from the glasses

The user answers "Tap to hear it" (and can ask "what was that?" any time) with a gesture on
the glasses, via the DAT SDK's experimental **Inputs** capability (tap/select, swipe/nav,
buttons, capture button). Fallback: a "Read it" button in the phone app.

Behaviour on the PC:
- with a question pending: read the stored sign text (instant; it was already read);
- with no question pending: repeat the current instruction;
- otherwise: a short "Nothing to read" (or silence).

Protocol: the existing speech connection (`<phone>:9002`, PC <- -> phone) becomes two-way;
the phone sends JSON lines such as `{"event": "select", "t": <unix s>}` (also `"capture"`,
`"nav"` with a direction), plus `"speech"` with the recognised text and `"button"`. Done;
the final message set is in `docs/LIVE.md`.

**Experiment first (blocking the gesture choice):** a single tap on the glasses touchpad is
also how the glasses pause the camera stream (Meta docs: stream state `PAUSED`), and it is not
documented whether attaching Inputs changes that. Attach Inputs in the app, log every input
event and the stream state, then try tap, swipe and the capture button: pick the gesture that
arrives reliably and does not pause the stream or start a capture. Inputs (like Motion) may
need enabling for the app in the Wearables Developer Center (`PERMISSION_DENIED` otherwise).

## 5. Logging

Per sign read, add to `debug_out/reads.jsonl`: goal, match level, matching entry and arrow,
confidence, contradictions/ambiguity, and the decision (spoken / asked / read on request /
noted / repeat / arrival). The annotated video's chat panel shows the same (relevant signs
highlighted; noted ones greyed). Log every input event and what it triggered.

## 6. Tests

Building-agnostic by construction: every behaviour is tested across conventions.
- Goal parsing for each convention in section 1, including leading zeros, suffix letters,
  separators (`-`, `.`, space), names.
- Sign interpretation: same-shape matches, ranges, the `14-100` ambiguity resolved by goal
  shape, partial plates, prefix mismatches, building/wing-level matches, floor hints never
  excluding.
- Decision policy: each row of the table in section 3, for at least two conventions each;
  safety signs noted not spoken; asking rules (expiry, once per sign, priority); no repeats;
  arrival stops guidance.
- Input: tap with and without a pending question; protocol parsing; reconnects.
- Regression: existing `tests/` (synthetic signs, SiGNgapore2D labels) still pass.
- Replays on recorded walks with goals chosen from what their signs show. One local example:
  a walk whose destination plate is `014-190` (expected: arrival announced at that door). Such
  walk-specific cases are test data only; they must not shape the parser or the policy.

## 7. Suggested order of work

1. Inputs experiment on the glasses (phone side): which gesture to use.
2. Generic room-ID parser and shape-based matching (PC side, unit tests).
3. Decision policy: spoken / asked / noted / arrival / current instruction.
4. Two-way link: input events phone -> PC; "Read it" button in the app.
5. `--goal` in `main.py` (plus `--announce`), replay on recorded walks, then a live test.

## Conventions

- One writer at a time on `main`: pull before starting, push when done.
- Run `python -m pytest` before pushing Python changes; build the app (`android/`,
  `gradlew :app:assembleDebug`) before pushing app changes.
- Never commit `android/local.properties`, `.gradle/`, build outputs, or local calibration
  files such as `legibility_calib.json`.
