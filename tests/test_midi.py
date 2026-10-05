"""Unit tests for the strict SMF parser and timeline normalizer."""

import unittest
from fractions import Fraction

from app.midi import (
    MAX_TRACKS_PLUS_EVENTS,
    MidiError,
    build_measure_map,
    build_tempo_segments,
    musical_position,
    normalize,
    parse,
    time_at_tick,
    TempoEvent,
    TimeSignatureEvent,
)


# -- SMF builders ---------------------------------------------------------


def varlen(n):
    out = bytes([n & 0x7F])
    n >>= 7
    while n:
        out = bytes([(n & 0x7F) | 0x80]) + out
        n >>= 7
    return out


def track(payload):
    return b"MTrk" + len(payload).to_bytes(4, "big") + payload


def header(fmt, ntracks, division):
    return (
        b"MThd"
        + (6).to_bytes(4, "big")
        + fmt.to_bytes(2, "big")
        + ntracks.to_bytes(2, "big")
        + division.to_bytes(2, "big")
    )


def tempo(delta, us):
    return varlen(delta) + bytes([0xFF, 0x51, 0x03]) + us.to_bytes(3, "big")


def sig(delta, nn, dd, cc=0x24, bb=0x08):
    return varlen(delta) + bytes([0xFF, 0x58, 0x04, nn, dd, cc, bb])


def ev(delta, *bs):
    return varlen(delta) + bytes(bs)


EOT = ev(0, 0xFF, 0x2F, 0x00)


def fractions(result):
    return [Fraction(e["time_us"]["numerator"], e["time_us"]["denominator"])
            for e in result["events"]]


def positions(result):
    return [
        (
            e["musical_position"]["measure"],
            e["musical_position"]["time_signature"]["signature"],
            e["musical_position"]["beat"]["fraction"],
        )
        for e in result["events"]
    ]


# -- header / chunk validation --------------------------------------------


class HeaderTests(unittest.TestCase):
    def test_bad_magic(self):
        with self.assertRaises(MidiError) as ctx:
            parse(b"NOPE" + bytes(10))
        self.assertEqual(ctx.exception.code, "bad_magic")
        self.assertEqual(ctx.exception.offset, 0)

    def test_truncated_header(self):
        with self.assertRaises(MidiError) as ctx:
            parse(b"MThd\x00\x00")
        self.assertEqual(ctx.exception.code, "truncated_header")
        self.assertEqual(ctx.exception.offset, 6)

    def test_bad_header_length(self):
        data = b"MThd" + (8).to_bytes(4, "big") + bytes(8)
        with self.assertRaises(MidiError) as ctx:
            parse(data)
        self.assertEqual(ctx.exception.code, "bad_header_length")
        self.assertEqual(ctx.exception.offset, 4)

    def test_format_2_rejected(self):
        with self.assertRaises(MidiError) as ctx:
            parse(header(2, 2, 480))
        self.assertEqual(ctx.exception.code, "unsupported_format")
        self.assertEqual(ctx.exception.offset, 8)

    def test_smpte_division_rejected(self):
        with self.assertRaises(MidiError) as ctx:
            parse(header(1, 1, 0xE728))  # -25 fps, 40 ticks/frame
        self.assertEqual(ctx.exception.code, "unsupported_division")
        self.assertEqual(ctx.exception.offset, 12)

    def test_zero_ppqn_rejected(self):
        with self.assertRaises(MidiError) as ctx:
            parse(header(1, 1, 0))
        self.assertEqual(ctx.exception.code, "invalid_division")

    def test_zero_tracks_rejected(self):
        with self.assertRaises(MidiError) as ctx:
            parse(header(1, 0, 480))
        self.assertEqual(ctx.exception.code, "no_tracks")

    def test_format_0_requires_single_track(self):
        with self.assertRaises(MidiError) as ctx:
            parse(header(0, 2, 480))
        self.assertEqual(ctx.exception.code, "bad_track_count")

    def test_missing_track_chunk(self):
        with self.assertRaises(MidiError) as ctx:
            parse(header(1, 1, 480))
        self.assertEqual(ctx.exception.code, "truncated_track_header")
        self.assertEqual(ctx.exception.offset, 14)

    def test_wrong_chunk_type(self):
        data = header(1, 1, 480) + b"LIST" + bytes(4)
        with self.assertRaises(MidiError) as ctx:
            parse(data)
        self.assertEqual(ctx.exception.code, "bad_chunk")
        self.assertEqual(ctx.exception.offset, 14)

    def test_truncated_track_body(self):
        data = header(1, 1, 480) + b"MTrk" + (100).to_bytes(4, "big") + b"\x00\xff"
        with self.assertRaises(MidiError) as ctx:
            parse(data)
        self.assertEqual(ctx.exception.code, "truncated_track")
        self.assertEqual(ctx.exception.offset, 18)

    def test_trailing_bytes_rejected(self):
        data = header(1, 1, 480) + track(EOT) + b"\x00"
        with self.assertRaises(MidiError) as ctx:
            parse(data)
        self.assertEqual(ctx.exception.code, "trailing_bytes")
        self.assertEqual(ctx.exception.offset, len(data) - 1)

    def test_file_too_large(self):
        with self.assertRaises(MidiError) as ctx:
            parse(b"\x00" * (1 << 20 | 1))
        self.assertEqual(ctx.exception.code, "file_too_large")


# -- event-level validation ------------------------------------------------


class EventTests(unittest.TestCase):
    def test_varlen_too_long(self):
        payload = b"\x80\x80\x80\x80\x00" + ev(0, 0x90, 60, 100)
        with self.assertRaises(MidiError) as ctx:
            parse(header(1, 1, 480) + track(payload))
        self.assertEqual(ctx.exception.code, "varlen_too_long")
        self.assertEqual(ctx.exception.offset, 22)

    def test_unterminated_varlen(self):
        payload = b"\x80\x80"
        with self.assertRaises(MidiError) as ctx:
            parse(header(1, 1, 480) + track(payload))
        self.assertEqual(ctx.exception.code, "truncated_varlen")
        self.assertEqual(ctx.exception.offset, 22)

    def test_missing_status_byte(self):
        payload = ev(0, 0x90, 60, 100) + varlen(10)
        with self.assertRaises(MidiError) as ctx:
            parse(header(1, 1, 480) + track(payload))
        self.assertEqual(ctx.exception.code, "truncated_event")

    def test_orphan_running_status(self):
        payload = ev(0, 60, 100)  # data bytes before any status
        with self.assertRaises(MidiError) as ctx:
            parse(header(1, 1, 480) + track(payload))
        self.assertEqual(ctx.exception.code, "orphan_running_status")
        self.assertEqual(ctx.exception.offset, 23)

    def test_running_status_cancelled_by_meta(self):
        payload = ev(0, 0x90, 60, 100) + ev(0, 0xFF, 0x01, 0x00) + ev(0, 62, 100)
        with self.assertRaises(MidiError) as ctx:
            parse(header(1, 1, 480) + track(payload))
        self.assertEqual(ctx.exception.code, "orphan_running_status")

    def test_running_status_cancelled_by_sysex(self):
        payload = ev(0, 0x90, 60, 100) + ev(0, 0xF0, 0x00) + ev(0, 62, 100)
        with self.assertRaises(MidiError) as ctx:
            parse(header(1, 1, 480) + track(payload))
        self.assertEqual(ctx.exception.code, "orphan_running_status")

    def test_running_status_ok(self):
        payload = ev(0, 0x90, 60, 100) + ev(10, 62, 100) + EOT
        result = normalize(header(1, 1, 480) + track(payload))
        self.assertEqual(result["channel_event_count"], 2)
        self.assertEqual(result["events"][1]["data"], [62, 100])
        self.assertEqual(result["events"][1]["tick"], 10)

    def test_illegal_status(self):
        payload = ev(0, 0xF8)
        with self.assertRaises(MidiError) as ctx:
            parse(header(1, 1, 480) + track(payload))
        self.assertEqual(ctx.exception.code, "illegal_status")
        self.assertEqual(ctx.exception.offset, 23)

    def test_data_byte_high_bit(self):
        payload = ev(0, 0x90, 60, 0x80)
        with self.assertRaises(MidiError) as ctx:
            parse(header(1, 1, 480) + track(payload))
        self.assertEqual(ctx.exception.code, "invalid_data_byte")
        self.assertEqual(ctx.exception.offset, 25)

    def test_truncated_channel_event(self):
        payload = ev(0, 0x90, 60)  # note on needs two data bytes
        with self.assertRaises(MidiError) as ctx:
            parse(header(1, 1, 480) + track(payload))
        self.assertEqual(ctx.exception.code, "truncated_event")

    def test_program_change_single_data_byte(self):
        payload = ev(0, 0xC3, 5) + ev(0, 0xD0, 7) + EOT
        result = normalize(header(1, 1, 480) + track(payload))
        self.assertEqual(
            [e["type"] for e in result["events"]],
            ["program_change", "channel_pressure"],
        )
        self.assertEqual(result["events"][0]["channel"], 3)

    def test_truncated_meta_event(self):
        payload = ev(0, 0xFF, 0x01, 0x05) + b"ab"  # declares 5, gives 2
        with self.assertRaises(MidiError) as ctx:
            parse(header(1, 1, 480) + track(payload))
        self.assertEqual(ctx.exception.code, "truncated_event")

    def test_bad_tempo_length(self):
        payload = ev(0, 0xFF, 0x51, 0x02, 0x07, 0xA1)
        with self.assertRaises(MidiError) as ctx:
            parse(header(1, 1, 480) + track(payload))
        self.assertEqual(ctx.exception.code, "bad_meta_length")

    def test_zero_tempo_rejected(self):
        payload = tempo(0, 0)
        with self.assertRaises(MidiError) as ctx:
            parse(header(1, 1, 480) + track(payload))
        self.assertEqual(ctx.exception.code, "invalid_tempo")

    def test_bad_end_of_track_length(self):
        payload = ev(0, 0xFF, 0x2F, 0x01, 0x00)
        with self.assertRaises(MidiError) as ctx:
            parse(header(1, 1, 480) + track(payload))
        self.assertEqual(ctx.exception.code, "bad_meta_length")

    def test_truncated_sysex(self):
        payload = ev(0, 0xF0, 0x05) + b"ab"
        with self.assertRaises(MidiError) as ctx:
            parse(header(1, 1, 480) + track(payload))
        self.assertEqual(ctx.exception.code, "truncated_event")

    def test_sysex_and_meta_accepted(self):
        payload = (
            ev(0, 0xF0, 0x03, 0x7E, 0x7F, 0x09)
            + ev(0, 0xFF, 0x03, 0x04) + b"name"
            + ev(0, 0x90, 60, 100)
            + EOT
        )
        result = normalize(header(1, 1, 480) + track(payload))
        self.assertEqual(result["channel_event_count"], 1)


# -- limits ------------------------------------------------------------------


class LimitTests(unittest.TestCase):
    def test_track_and_event_total_limit(self):
        # 1 track + 9999 channel events == 10000: accepted.
        payload = ev(0, 0x90, 60, 100) + ev(0, 60, 100) * 9998 + EOT
        result = normalize(header(1, 1, 480) + track(payload))
        self.assertEqual(result["channel_event_count"], 9999)

        # One more channel event crosses the limit.
        payload = ev(0, 0x90, 60, 100) + ev(0, 60, 100) * 9999 + EOT
        with self.assertRaises(MidiError) as ctx:
            parse(header(1, 1, 480) + track(payload))
        self.assertEqual(ctx.exception.code, "limit_exceeded")

    def test_track_count_limit(self):
        with self.assertRaises(MidiError) as ctx:
            parse(header(1, MAX_TRACKS_PLUS_EVENTS + 1, 480))
        self.assertEqual(ctx.exception.code, "limit_exceeded")
        self.assertEqual(ctx.exception.offset, 10)


# -- tempo map and timing ----------------------------------------------------


class TempoTests(unittest.TestCase):
    def test_default_tempo(self):
        payload = ev(0, 0x90, 60, 100) + ev(480, 0x80, 60, 0) + EOT
        result = normalize(header(1, 1, 480) + track(payload))
        self.assertEqual(fractions(result), [Fraction(0), Fraction(500000)])

    def test_reduced_fraction(self):
        payload = ev(1, 0x90, 60, 100) + EOT
        result = normalize(header(1, 1, 480) + track(payload))
        self.assertEqual(fractions(result), [Fraction(3125, 3)])
        self.assertEqual(result["events"][0]["time_us"]["fraction"], "3125/3")

    def test_tempo_takes_effect_at_its_tick(self):
        conductor = tempo(0, 500000) + tempo(480, 250000) + EOT
        notes = ev(480, 0x90, 60, 100) + ev(480, 0x80, 60, 0) + EOT
        data = header(1, 2, 480) + track(conductor) + track(notes)
        result = normalize(data)
        # Tick 480 itself is still priced with the old tempo; the new tempo
        # only governs the interval that starts at tick 480.
        self.assertEqual(
            fractions(result), [Fraction(500000), Fraction(750000)]
        )

    def test_same_tick_last_tempo_wins(self):
        conductor = tempo(0, 600000) + tempo(0, 250000) + EOT
        notes = ev(480, 0x90, 60, 100) + EOT
        data = header(1, 2, 480) + track(conductor) + track(notes)
        result = normalize(data)
        self.assertEqual(fractions(result), [Fraction(250000)])

    def test_format_1_ignores_non_first_track_tempos(self):
        conductor = tempo(0, 500000) + EOT
        notes = tempo(0, 1000) + ev(480, 0x90, 60, 100) + EOT
        data = header(1, 2, 480) + track(conductor) + track(notes)
        result = normalize(data)
        self.assertEqual(fractions(result), [Fraction(500000)])

    def test_format_0_uses_own_tempos(self):
        payload = tempo(0, 250000) + ev(480, 0x90, 60, 100) + EOT
        result = normalize(header(0, 1, 480) + track(payload))
        self.assertEqual(result["format"], 0)
        self.assertEqual(fractions(result), [Fraction(250000)])

    def test_multi_segment_tempo_map(self):
        segments = build_tempo_segments(
            [TempoEvent(0, 500000), TempoEvent(480, 250000), TempoEvent(960, 1000000)]
        )
        self.assertEqual(time_at_tick(0, segments, 480), Fraction(0))
        self.assertEqual(time_at_tick(240, segments, 480), Fraction(250000))
        self.assertEqual(time_at_tick(480, segments, 480), Fraction(500000))
        self.assertEqual(time_at_tick(960, segments, 480), Fraction(750000))
        self.assertEqual(time_at_tick(1200, segments, 480), Fraction(1250000))

    def test_tempo_beyond_last_event_is_harmless(self):
        conductor = tempo(0, 500000) + tempo(9999, 250000) + EOT
        notes = ev(480, 0x90, 60, 100) + EOT
        data = header(1, 2, 480) + track(conductor) + track(notes)
        result = normalize(data)
        self.assertEqual(fractions(result), [Fraction(500000)])


# -- ordering ----------------------------------------------------------------


class OrderingTests(unittest.TestCase):
    def test_stable_order_by_tick_track_order(self):
        t0 = ev(240, 0x90, 60, 100) + EOT
        t1 = ev(0, 0x91, 61, 100) + ev(240, 0x81, 61, 0) + EOT
        t2 = ev(120, 0x92, 62, 100) + ev(120, 0x82, 62, 0) + EOT
        data = header(1, 3, 480) + track(t0) + track(t1) + track(t2)
        result = normalize(data)
        keys = [(e["tick"], e["track"], e["order"]) for e in result["events"]]
        self.assertEqual(
            keys,
            [(0, 1, 0), (120, 2, 0), (240, 0, 0), (240, 1, 1), (240, 2, 1)],
        )
        self.assertEqual(
            fractions(result),
            [Fraction(0)] + [Fraction(125000)] + [Fraction(250000)] * 3,
        )


# -- musical position projection --------------------------------------------


class MusicalPositionTests(unittest.TestCase):
    def test_default_4_4(self):
        payload = (
            ev(0, 0x90, 60, 100)
            + ev(480, 0x90, 62, 100)
            + ev(1440, 0x90, 64, 100)  # tick 1920
            + ev(480, 0x90, 65, 100)   # tick 2400
            + EOT
        )
        result = normalize(
            header(0, 1, 480) + track(payload),
            projection="musical_position",
        )
        self.assertEqual(
            positions(result),
            [
                (1, "4/4", "1/1"),
                (1, "4/4", "2/1"),
                (2, "4/4", "1/1"),
                (2, "4/4", "2/1"),
            ],
        )

    def test_beat_fraction_is_reduced(self):
        payload = ev(240, 0x90, 60, 100) + EOT
        result = normalize(
            header(0, 1, 480) + track(payload),
            projection="musical_position",
        )
        beat = result["events"][0]["musical_position"]["beat"]
        self.assertEqual((beat["numerator"], beat["denominator"]), (3, 2))
        self.assertEqual(beat["fraction"], "3/2")

    def test_mid_measure_change_opens_new_measure(self):
        conductor = sig(0, 4, 2) + sig(1440, 3, 2) + EOT
        notes = (
            ev(0, 0x90, 60, 100)       # tick 0    -> M1, 4/4, beat 1
            + ev(1440, 0x90, 62, 100)  # tick 1440 -> M2, 3/4, beat 1
            + ev(720, 0x90, 64, 100)   # tick 2160 -> M2, beat 5/2
            + ev(720, 0x90, 65, 100)   # tick 2880 -> M3, beat 1
            + EOT
        )
        data = header(1, 2, 480) + track(conductor) + track(notes)
        result = normalize(data, projection="musical_position")
        self.assertEqual(
            positions(result),
            [
                (1, "4/4", "1/1"),
                (2, "3/4", "1/1"),
                (2, "3/4", "5/2"),
                (3, "3/4", "1/1"),
            ],
        )

    def test_incomplete_measure_still_counts(self):
        # 3/4 declared one beat into the 4/4 first measure: measure 1 is
        # short but still counts, so the change opens measure 2.
        conductor = sig(480, 3, 2) + EOT
        notes = ev(0, 0x90, 60, 100) + ev(480, 0x90, 62, 100) + EOT
        data = header(1, 2, 480) + track(conductor) + track(notes)
        result = normalize(data, projection="musical_position")
        got = [(measure, beat) for measure, _, beat in positions(result)]
        self.assertEqual(got, [(1, "1/1"), (2, "1/1")])

    def test_same_tick_last_signature_wins(self):
        conductor = sig(960, 5, 2) + sig(0, 3, 2) + EOT  # both at tick 960
        notes = ev(960, 0x90, 60, 100) + EOT
        data = header(1, 2, 480) + track(conductor) + track(notes)
        result = normalize(data, projection="musical_position")
        pos = result["events"][0]["musical_position"]
        self.assertEqual(pos["time_signature"]["signature"], "3/4")
        self.assertEqual(pos["measure"], 2)

    def test_tick_zero_signature_replaces_default(self):
        conductor = sig(0, 6, 3) + EOT
        notes = (
            ev(0, 0x90, 60, 100)
            + ev(240, 0x90, 62, 100)   # one eighth-note beat later
            + ev(1200, 0x90, 64, 100)  # tick 1440: measure 2
            + EOT
        )
        data = header(1, 2, 480) + track(conductor) + track(notes)
        result = normalize(data, projection="musical_position")
        self.assertEqual(
            positions(result),
            [(1, "6/8", "1/1"), (1, "6/8", "2/1"), (2, "6/8", "1/1")],
        )

    def test_format_1_ignores_non_first_track_signatures(self):
        conductor = sig(0, 4, 2) + EOT
        notes = sig(0, 6, 3) + ev(480, 0x90, 60, 100) + EOT  # decoy 6/8
        data = header(1, 2, 480) + track(conductor) + track(notes)
        result = normalize(data, projection="musical_position")
        pos = result["events"][0]["musical_position"]
        self.assertEqual(pos["time_signature"]["signature"], "4/4")
        self.assertEqual((pos["measure"], pos["beat"]["fraction"]), (1, "2/1"))

    def test_format_0_uses_own_signatures(self):
        payload = sig(0, 3, 2) + ev(1440, 0x90, 60, 100) + EOT
        result = normalize(
            header(0, 1, 480) + track(payload),
            projection="musical_position",
        )
        pos = result["events"][0]["musical_position"]
        self.assertEqual(
            (pos["measure"], pos["time_signature"]["signature"]), (2, "3/4")
        )

    def test_denominator_exponent_bounds(self):
        # Exponents 0 (whole-note beat) and 7 (128th-note beat) are accepted.
        for dd in (0, 7):
            payload = sig(0, 3, dd) + ev(0, 0x90, 60, 100) + EOT
            result = normalize(
                header(0, 1, 480) + track(payload),
                projection="musical_position",
            )
            pos = result["events"][0]["musical_position"]
            self.assertEqual(pos["time_signature"]["denominator"], 1 << dd)

    def test_no_projection_no_position_keys(self):
        payload = sig(0, 3, 2) + ev(0, 0x90, 60, 100) + EOT
        result = normalize(header(0, 1, 480) + track(payload))
        self.assertNotIn("musical_position", result["events"][0])


class TimeSignatureValidationTests(unittest.TestCase):
    def test_bad_length_rejected_with_projection(self):
        payload = ev(0, 0xFF, 0x58, 0x03, 0x04, 0x02, 0x24) + EOT
        data = header(0, 1, 480) + track(payload)
        with self.assertRaises(MidiError) as ctx:
            normalize(data, projection="musical_position")
        self.assertEqual(ctx.exception.code, "bad_meta_length")
        self.assertIsInstance(ctx.exception.offset, int)

    def test_bad_length_accepted_without_projection(self):
        payload = (
            ev(0, 0xFF, 0x58, 0x03, 0x04, 0x02, 0x24)
            + ev(0, 0x90, 60, 100)
            + EOT
        )
        result = normalize(header(0, 1, 480) + track(payload))
        self.assertEqual(result["channel_event_count"], 1)
        self.assertNotIn("musical_position", result["events"][0])

    def test_zero_numerator_rejected(self):
        payload = sig(0, 0, 2) + EOT
        with self.assertRaises(MidiError) as ctx:
            normalize(
                header(0, 1, 480) + track(payload),
                projection="musical_position",
            )
        self.assertEqual(ctx.exception.code, "invalid_time_signature")

    def test_denominator_exponent_above_7_rejected(self):
        payload = sig(0, 4, 8) + EOT
        with self.assertRaises(MidiError) as ctx:
            normalize(
                header(0, 1, 480) + track(payload),
                projection="musical_position",
            )
        self.assertEqual(ctx.exception.code, "invalid_time_signature")

    def test_invalid_signature_in_any_track_rejected(self):
        # Format 1 uses only track 0 for the map, but malformed signature
        # events anywhere are rejected, mirroring tempo strictness.
        conductor = EOT
        notes = sig(0, 0, 2) + ev(0, 0x90, 60, 100) + EOT
        data = header(1, 2, 480) + track(conductor) + track(notes)
        with self.assertRaises(MidiError) as ctx:
            normalize(data, projection="musical_position")
        self.assertEqual(ctx.exception.code, "invalid_time_signature")

    def test_error_reports_locating_offset(self):
        payload = ev(0, 0x90, 60, 100) + sig(0, 4, 9) + EOT
        data = header(0, 1, 480) + track(payload)
        with self.assertRaises(MidiError) as ctx:
            normalize(data, projection="musical_position")
        # 14-byte header + 8-byte MTrk header + 4-byte note + 1-byte delta
        # lands on the 0xFF status byte of the signature meta event.
        self.assertEqual(ctx.exception.offset, 14 + 8 + 4 + 1)


class MeasureMapTests(unittest.TestCase):
    def test_build_measure_map_default(self):
        self.assertEqual(build_measure_map([], 480), [(0, 1, 4, 4)])

    def test_measure_numbers_across_changes(self):
        events = [
            TimeSignatureEvent(1440, 3, 2),
            TimeSignatureEvent(2880, 6, 3),
        ]
        # 4/4 covers [0, 1440): one incomplete measure; 3/4 covers
        # [1440, 2880): exactly one full measure; 6/8 starts measure 3.
        self.assertEqual(
            build_measure_map(events, 480),
            [(0, 1, 4, 4), (1440, 2, 3, 4), (2880, 3, 6, 8)],
        )

    def test_musical_position_exact_beat(self):
        measure_map = build_measure_map([TimeSignatureEvent(1920, 3, 2)], 480)
        measure, nn, dd, beat = musical_position(1920 + 960, measure_map, 480)
        self.assertEqual((measure, nn, dd, beat), (2, 3, 4, Fraction(3, 1)))


if __name__ == "__main__":
    unittest.main()
