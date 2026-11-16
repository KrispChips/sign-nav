"""Decision policy (docs/step1-goal.md, section 3 and "The user stays in charge"): each row of
the table in at least two numbering conventions, the asking rules, no repeats, arrival, the
user's own knowledge, and voice commands. Goals are convention examples, not one building."""
import pytest

import commands
from guide import ASK_TTL, Guide, Read

LEFT_WALL = dict(rel=-90.0, mount="wall")      # sign on the left wall
AHEAD = dict(rel=0.0, mount="facing")          # sign facing the user


def read(sid, t, *entries, where=AHEAD, ocr=None):
    es = [{"text": a, "arrow": b} for a, b in entries]
    return Read(sign_id=sid, entries=es, t=t, ocr_text=ocr, **where)


# ---------------------------------------------------------------- spoken: names the goal

@pytest.mark.parametrize("goal,text", [("014-190", "014-190"), ("A140", "A140"),
                                       ("C2.15", "C2.15"), ("1200", "Room 1200")])
def test_sign_naming_the_goal_is_spoken(goal, text):
    g = Guide(goal)
    d = g.on_read(read(1, 0, (text, "left")))
    assert d.kind == "spoken"
    assert d.text.startswith("Sign ahead:") and "to the left" in d.text
    assert g.target.room_id in d.text


def test_information_not_orders():
    d = Guide("A140").on_read(read(1, 0, ("A140", "right")))
    assert "turn" not in d.text.lower() and "go " not in d.text.lower()


# ---------------------------------------------------------------- arrival

@pytest.mark.parametrize("goal,plate", [("014-190", "014-190"), ("A140", "A140"),
                                        ("L3-07", "L3-07"), ("120B", "120B")])
def test_own_door_plate_is_arrival_and_stops_guidance(goal, plate):
    g = Guide(goal)
    d = g.on_read(read(1, 0, (plate, "none"), where=dict(rel=70.0, mount="wall")))
    assert d.kind == "arrival" and d.text.startswith("You've arrived") and "on your right" in d.text
    assert g.done
    assert g.on_read(read(2, 1, (plate.replace("1", "2"), "left"))).kind == "noted"
    assert g.hear("what was that", 2).text == d.text


@pytest.mark.parametrize("goal,plate,setup", [
    ("014-190", "190", ("Building 14", "none")),
    ("A140", "140", ("Wing A", "none")),
])
def test_partial_plate_is_tentative_until_building_known(goal, plate, setup):
    g = Guide(goal)
    d = g.on_read(read(1, 0, (plate, "none")))
    assert d.kind == "spoken" and "may be" in d.text and not g.done
    g2 = Guide(goal)
    g2.on_read(read(1, 0, setup))
    d2 = g2.on_read(read(2, 1, (plate, "none")))
    assert d2.kind == "arrival" and g2.done


def test_other_rooms_plates_establish_the_building():
    g = Guide("014-190")
    assert g.on_read(read(1, 0, ("014-180", "none"))).kind == "noted"
    assert g.on_read(read(2, 1, ("190", "none"))).kind == "arrival"


# ---------------------------------------------------------------- spoken: range/building/wing/floor

@pytest.mark.parametrize("goal,text", [("A140", "A121 - A159"), ("014-190", "014-150 to 014-199"),
                                       ("1200", "1100-1299")])
def test_range_containing_goal(goal, text):
    d = Guide(goal).on_read(read(1, 0, (text, "right"), where=LEFT_WALL))
    assert d.kind == "spoken" and "rooms" in d.text and "That includes" in d.text
    assert d.text.startswith("Sign on your left:")


@pytest.mark.parametrize("goal,text,word", [("014-190", "Building 14", "building 14"),
                                            ("A140", "Wing A", "wing A"),
                                            ("C2.15", "Block C", "block C")])
def test_building_or_wing_containing_goal(goal, text, word):
    d = Guide(goal).on_read(read(1, 0, (text, "left")))
    assert d.kind == "spoken" and word in d.text


@pytest.mark.parametrize("goal", ["1200", "L3-07"])
def test_lift_when_goal_on_another_floor(goal):
    g = Guide(goal)
    g.context.user_floor = 1
    d = g.on_read(read(1, 0, ("Lifts", "left")))
    assert d.kind == "spoken" and "lift" in d.text and "probably on floor" in d.text


def test_floor_plate_establishes_floor_for_the_lift():
    g = Guide("1200")
    g.on_read(read(1, 0, ("Level 1", "none")))
    assert g.context.user_floor == 1
    assert g.on_read(read(2, 1, ("Stairs", "right"))).kind == "spoken"


# ---------------------------------------------------------------- spoken: unclear

@pytest.mark.parametrize("goal,a,b", [("A140", "A140", "A140"), ("014-190", "014-190", "14-190")])
def test_matching_entries_pointing_different_ways(goal, a, b):
    d = Guide(goal).on_read(read(1, 0, (a, "left"), (b, "right")))
    assert d.kind == "spoken" and "unclear" in d.text
    assert "to the left" in d.text and "to the right" in d.text


# ---------------------------------------------------------------- asked / noted

@pytest.mark.parametrize("goal", ["A140", "014-190"])
def test_unrelated_sign_with_content_is_asked(goal):
    g = Guide(goal)
    d = g.on_read(read(1, 0, ("Cafeteria", "left"), ("Library", "right"), where=LEFT_WALL))
    assert d.kind == "asked" and d.text == "Sign on your left. Want to hear it?"
    r = g.hear("read it", 3)
    assert r.kind == "read" and "Cafeteria" in r.text and "Library" in r.text


@pytest.mark.parametrize("goal,entries", [
    ("A140", [("EXIT", "right")]),
    ("014-190", [("Fire Exit", "left"), ("Stairs", "left")]),
    ("A140", [("Fire extinguisher", "none")]),
])
def test_safety_signs_are_noted_not_asked(goal, entries):
    g = Guide(goal)
    d = g.on_read(read(1, 0, *entries))
    assert d.kind == "noted" and d.reason == "safety sign" and d.text is None
    assert g.noted[-1]["entries"]


@pytest.mark.parametrize("goal,plate", [("A140", "A121"), ("014-190", "014-150")])
def test_other_door_plates_are_noted(goal, plate):
    d = Guide(goal).on_read(read(1, 0, (plate, "none")))
    assert d.kind == "noted" and d.reason == "door plate"


def test_noted_keeps_what_step_2_needs():
    g = Guide("A140")
    g.on_read(Read(sign_id=4, entries=[{"text": "A121", "arrow": "none"}], t=5.0, rel=80.0,
                   mount="wall", heading=12.0))
    n = g.noted[-1]
    assert (n["t"], n["sign"], n["rel"], n["heading"]) == (5.0, 4, 80.0, 12.0)
    assert n["parsed"] == ["ROOM A121"]


def test_asked_once_per_sign():
    g = Guide("A140")
    assert g.on_read(read(1, 0, ("Cafeteria", "left"), ("Library", "right"))).kind == "asked"
    assert g.on_read(read(1, 20, ("Cafeteria", "left"), ("Library", "right"))).kind == "noted"


def test_one_question_at_a_time_and_expiry():
    g = Guide("A140")
    assert g.on_read(read(1, 0, ("Cafeteria", "left"), ("Library", "right"))).kind == "asked"
    d = g.on_read(read(2, 2, ("Gym", "left"), ("Pool", "right")))
    assert d.kind == "noted" and "pending" in d.reason
    assert g.on_read(read(3, ASK_TTL + 1, ("Gym", "left"), ("Pool", "right"))).kind == "asked"


def test_spoken_instruction_takes_priority_over_question():
    g = Guide("A140")
    g.on_read(read(1, 0, ("Cafeteria", "left"), ("Library", "right")))
    assert g.on_read(read(2, 1, ("A121 - A159", "right"))).kind == "spoken"
    assert g.hear("read it", 2).kind == "repeat"     # the question is gone


def test_saying_no_drops_the_question():
    g = Guide("A140")
    g.on_read(read(1, 0, ("Cafeteria", "left"), ("Library", "right")))
    assert g.hear("no", 1).text == "Okay."
    assert g.hear("read it", 2).kind == "nothing"


def test_read_it_a_little_late_still_works():
    g = Guide("A140")
    g.on_read(read(1, 0, ("Cafeteria", "left"), ("Library", "right")))
    assert g.hear("read it", ASK_TTL + 5).kind == "read"


# ---------------------------------------------------------------- no repeats

def test_same_instruction_not_repeated():
    g = Guide("A140")
    assert g.on_read(read(1, 0, ("A121 - A159", "right"))).kind == "spoken"
    assert g.on_read(read(1, 1, ("A121 - A159", "right"))).kind == "noted"
    assert g.on_read(read(2, 5, ("A121 - A159", "right"))).kind == "noted"


@pytest.mark.parametrize("goal,steps", [
    ("A140", ["Wing A", "A121 - A159", "A140"]),
    ("014-190", ["Building 14", "014-150 to 014-199", "014-190"]),
])
def test_more_specific_is_said_again(goal, steps):
    g = Guide(goal)
    kinds = [g.on_read(read(i, i, (text, "right"))).kind for i, text in enumerate(steps)]
    assert kinds == ["spoken"] * 3


def test_less_specific_after_specific_is_noted():
    g = Guide("A140")
    g.on_read(read(1, 0, ("A121 - A159", "right")))
    assert g.on_read(read(2, 1, ("Wing A", "left"))).kind == "noted"


def test_direction_change_is_said_again():
    g = Guide("A140")
    g.on_read(read(1, 0, ("A121 - A159", "right")))
    assert g.on_read(read(2, 4, ("A121 - A159", "left"))).kind == "spoken"


def test_repeat_command_repeats_current_instruction():
    g = Guide("014-190")
    d = g.on_read(read(1, 0, ("014-150 to 014-199", "left")))
    for words in ("what was that", "repeat", "read it"):
        r = g.hear(words, 2)
        assert r.kind == "repeat" and r.text == d.text
    assert Guide("A140").hear("repeat", 0).text == "Nothing to read yet."


# ---------------------------------------------------------------- the user's own knowledge

def test_opening_question_asks_what_they_know():
    d = Guide("014-190").start(0)
    assert d.kind == "question" and "014-190" in d.text and "route you usually take" in d.text


@pytest.mark.parametrize("reply", ["it's on the left side", "I usually go left",
                                   "past the cafeteria on the left"])
def test_reply_after_question_is_kept_as_a_note(reply):
    g = Guide("A140")
    g.start(0)
    d = g.hear(reply, 3)
    assert d.kind == "note" and g.user_notes[-1]["text"] == reply and g.user_side == "left"


def test_free_reply_right_after_question_is_kept():
    g = Guide("A140")
    g.start(0)
    assert g.hear("the lab is by the big windows", 4).kind == "note"
    assert g.user_notes[-1]["side"] is None


def test_nothing_is_a_fine_answer():
    g = Guide("A140")
    g.start(0)
    assert g.hear("nothing", 2).text == "Okay."
    assert not g.user_notes


@pytest.mark.parametrize("goal,text", [("A140", "A121 - A159"), ("014-190", "014-150 to 014-199")])
def test_disagreement_with_user_is_said_once_and_not_overridden(goal, text):
    g = Guide(goal)
    g.add_user_note("I think it's on the left", 0)
    d = g.on_read(read(1, 1, (text, "right")))
    assert d.kind == "spoken" and "You said it's to the left" in d.text and "your call" in d.text
    d2 = g.on_read(read(2, 5, (text, "up")))   # another sign, still not left
    assert "You said" not in (d2.text or "")


def test_agreement_with_user_adds_nothing():
    g = Guide("A140")
    g.add_user_note("it's on the right", 0)
    d = g.on_read(read(1, 1, ("A121 - A159", "right")))
    assert "You said" not in d.text


def test_new_note_can_raise_disagreement_again():
    g = Guide("A140")
    g.add_user_note("left", 0)
    g.on_read(read(1, 1, ("A121 - A159", "right")))
    g.add_user_note("no wait, I usually take the left corridor", 2)
    d = g.on_read(read(2, 3, ("A140", "right")))
    assert "You said" in d.text


# ---------------------------------------------------------------- commands

@pytest.mark.parametrize("words,kind", [
    ("read it", "read"), ("What does it say?", "read"), ("Read it please", "read"),
    ("what was that", "repeat"), ("say that again", "repeat"), ("Repeat.", "repeat"),
    ("stop guiding", "stop"), ("yes", "yes"), ("nope", "no"), ("nothing", "no"),
    ("I usually go left here", "note"), ("the lab is near the north entrance", "note"),
    ("banana", "unknown"), ("", "unknown"),
])
def test_command_parsing(words, kind):
    assert commands.parse(words).kind == kind


def test_unrecognised_speech_is_not_guessed():
    d = Guide("A140").hear("banana", 50)
    assert d.kind == "unclear" and d.text == "Sorry, I didn't catch that."


def test_gestures_map_to_the_same_commands():
    g = Guide("A140")
    g.on_read(read(1, 0, ("Cafeteria", "left"), ("Library", "right")))
    assert g.on_command(commands.from_gesture("select"), 1).kind == "read"
    assert commands.from_gesture("button").kind == "read"
    assert commands.from_gesture("capture").kind == "repeat"


def test_stop():
    g = Guide("A140")
    assert g.hear("stop guiding", 0).kind == "stop" and g.done


def test_named_goal():
    g = Guide("Registrar")
    d = g.on_read(read(1, 0, ("Registrar Office", "left"), ("Cafeteria", "right")))
    assert d.kind == "spoken" and "to the left" in d.text


def test_decision_log_has_the_match():
    d = Guide("A140").on_read(read(1, 0, ("A121 - A159", "right")))
    log = d.log()
    assert log["decision"] == "spoken" and log["level"] == "RANGE" and log["arrow"] == "right"


# ---------------------------------------------------------------- building numbers on direction signs

@pytest.mark.parametrize("goal", ["014-190", "B14-100", "14-100"])
def test_bare_building_number_with_arrow_is_spoken(goal):
    d = Guide(goal).on_read(read(1, 0, ("14", "up"), ("16", "left"), ("20", "right")))
    assert d.kind == "spoken" and "building 14, straight ahead" in d.text


@pytest.mark.parametrize("goal", ["A140", "1200"])
def test_bare_numbers_are_not_buildings_for_other_conventions(goal):
    d = Guide(goal).on_read(read(1, 0, ("14", "up"), ("16", "left")))
    assert d.kind != "spoken"


def test_list_of_building_numbers_is_not_a_place_label():
    g = Guide("014-190")
    d = g.on_read(read(1, 0, ("14", "none"), ("16", "none"), ("18 thru 60", "none")))
    assert "You're in" not in (d.text or "") and g.context.user_building is None


def test_single_building_label_says_where_you_are():
    g = Guide("014-190")
    d = g.on_read(read(1, 0, ("Building 14", "none")))
    assert d.kind == "spoken" and d.text == "You're in building 14." and g.context.user_building == 14


# ---------------------------------------------------------------- found on the demo walk

def test_coarse_direction_flip_soon_after_is_not_resaid():
    g = Guide("014-190")
    assert g.on_read(read(1, 0, ("14", "right"), ("16", "left"))).kind == "spoken"
    assert g.on_read(read(2, 3, ("14", "up"), ("16", "left"))).kind == "noted"
    assert g.on_read(read(3, 25, ("14", "left"), ("16", "left"))).kind == "spoken"


def test_specific_direction_change_is_said_at_once():
    g = Guide("A140")
    g.on_read(read(1, 0, ("A121 - A159", "right")))
    assert g.on_read(read(2, 2, ("A121 - A159", "left"))).kind == "spoken"


def test_same_side_worded_differently_is_not_resaid():
    g = Guide("A140")
    g.on_read(read(1, 0, ("A121 - A159", "right")))                       # "to the right"
    assert g.on_read(read(2, 1, ("A121 - A159", "up-right"))).kind == "noted"   # "ahead to the right"


@pytest.mark.parametrize("entries", [[("EMERGENCY PLAN", "none"), ("Building 15", "none")],
                                     [("Evacuation plan", "none"), ("Level 3", "none")]])
def test_posted_plans_do_not_say_where_you_are(entries):
    g = Guide("014-190")
    d = g.on_read(read(1, 0, *entries))
    assert "You're" not in (d.text or "")
    assert g.context.user_building is None and g.context.user_floor is None


@pytest.mark.parametrize("label", ["B1", "14"])
def test_bare_codes_do_not_say_where_you_are(label):
    g = Guide("014-190")
    d = g.on_read(read(1, 0, (label, "none")))
    assert "You're" not in (d.text or "")
    assert g.context.user_floor is None and g.context.user_building is None


def test_floor_in_words_establishes_the_floor():
    g = Guide("1200")
    g.on_read(read(1, 0, ("First Level", "none")))
    assert g.context.user_floor == 1


@pytest.mark.parametrize("goal", ["015-100", "B15-100"])
def test_building_directory_without_the_goal_is_not_asked(goal):
    d = Guide(goal).on_read(read(1, 0, ("14", "up"), ("16", "left"), ("20", "right"),
                                 ("Library", "left")))
    assert d.kind == "noted" and d.reason == "building directory"


def test_posted_plan_is_noted_not_asked():
    d = Guide("A140").on_read(read(1, 0, ("EMERGENCY PLAN", "none"), ("Building 3", "none"),
                                   ("Toilets", "left")))
    assert d.kind == "noted" and d.reason == "posted plan"


def test_two_labels_without_arrows_are_not_asked():
    assert Guide("A140").on_read(read(1, 0, ("BIOL", "none"), ("UNIV", "none"))).kind == "noted"
    assert Guide("A140").on_read(read(1, 0, ("Pharmacy", "none"), ("Clinic", "none"),
                                          ("Store", "none"))).kind == "asked"


# ---------------------------------------------------------------- always-on listening (phone)

def test_overheard_chatter_is_ignored_silently():
    g = Guide("A140")
    for words in ("banana bread is great", "let's go to the left bar later"):
        d = g.hear(words, 100, overheard=True)
        assert d.kind == "nothing" and d.text is None
    assert not g.user_notes


def test_overheard_commands_and_own_notes_still_work():
    g = Guide("A140")
    g.on_read(read(1, 0, ("A121 - A159", "right")))
    assert g.hear("repeat", 50, overheard=True).kind == "repeat"
    assert g.hear("I usually go left here", 60, overheard=True).kind == "note"


def test_after_a_question_any_reply_counts():
    g = Guide("A140")
    g.start(0)
    assert g.hear("past the big windows", 5, overheard=True).kind == "note"
    g.on_read(read(1, 40, ("Cafeteria", "left"), ("Library", "right")))
    assert g.hear("blah blah", 42, overheard=True).text == "Sorry, I didn't catch that."


@pytest.mark.parametrize("goal,plate", [("014-190", "014-190"), ("A140", "A140")])
def test_goal_id_with_vertical_arrow_may_be_the_door(goal, plate):
    g = Guide(goal)
    d = g.on_read(read(1, 0, (plate, "down"), where=dict(rel=60.0, mount="wall")))
    assert d.kind == "spoken" and "It may be the door itself" in d.text and not g.done
    assert g.on_read(read(1, 1, (plate, "none"))).kind == "arrival"
