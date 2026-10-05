"""Unit tests for the strict SMF parser and timeline normalizer."""

import unittest
from fractions import Fraction

from app.midi import (
    MAX_TRACKS_PLUS_EVENTS,
    MidiError,
    build_metric_segments,
    build_tempo_segments,
    musical_position_at_tick,
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


def timesig(delta, numerator, denom_exp, cc=24, bb=8):
    return (
        varlen(delta)
        + bytes([0xFF, 0x58, 0x04, numerator, denom_exp, cc, bb])
    )


def ev(delta, *bs):
    return varlen(delta) + bytes(bs)


EOT = ev(0, 0xFF, 0x2F, 0x00)


def fractions(result):
    return [Fraction(e["time_us"]["numerator"], e["time_us"]["denominator"])
            for e in result["events"]]


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


# -- musical position projection ---------------------------------------------


def positions(result):
    return [
        (
            e["musical_position"]["measure"],
            (
                e["musical_position"]["time_signature"]["numerator"],
                e["musical_position"]["time_signature"]["denominator"],
            ),
            e["musical_position"]["beat"]["fraction"],
        )
        for e in result["events"]
    ]


class MusicalPositionTests(unittest.TestCase):
    def test_default_4_4_whole_bars(self):
        # PPQN 480: beat = 480 ticks, bar = 1920 ticks.
        payload = (
            ev(0, 0x90, 60, 100)       # tick 0    m1 beat 1
            + ev(480, 0x80, 60, 0)     # tick 480  m1 beat 2
            + ev(960, 0x90, 62, 100)   # tick 1440 m1 beat 4
            + ev(480, 0x80, 62, 0)     # tick 1920 m2 beat 1
            + ev(1920, 0x90, 64, 100)  # tick 3840 m3 beat 1
            + EOT
        )
        result = normalize(header(1, 1, 480) + track(payload),
                           musical_position=True)
        self.assertEqual(
            positions(result),
            [(1, (4, 4), "1/1"), (1, (4, 4), "2/1"), (1, (4, 4), "4/1"),
             (2, (4, 4), "1/1"), (3, (4, 4), "1/1")],
        )

    def test_fractional_beat_positions(self):
        # PPQN 480: 240 ticks is a half beat -> beat 3/2; 241 ticks -> 1441/1440.
        payload = (
            ev(240, 0x90, 60, 100)
            + ev(1, 0x80, 60, 0)
            + EOT
        )
        result = normalize(header(0, 1, 480) + track(payload),
                           musical_position=True)
        self.assertEqual(
            [p[2] for p in positions(result)],
            ["3/2", "721/480"],
        )

    def test_explicit_time_signature_at_zero(self):
        conductor = timesig(0, 3, 1) + EOT  # 3/2: beat = half note (960 ticks)
        notes = (
            ev(0, 0x90, 60, 100)      # tick 0    m1 beat 1
            + ev(960, 0x80, 60, 0)    # tick 960  m1 beat 2
            + ev(960, 0x90, 62, 100)  # tick 1920 m1 beat 3 (bar = 2880)
            + ev(960, 0x80, 62, 0)    # tick 2880 m2 beat 1
            + EOT
        )
        data = header(1, 2, 480) + track(conductor) + track(notes)
        result = normalize(data, musical_position=True)
        self.assertEqual(
            positions(result),
            [(1, (3, 2), "1/1"), (1, (3, 2), "2/1"),
             (1, (3, 2), "3/1"), (2, (3, 2), "1/1")],
        )

    def test_meter_change_on_bar_line(self):
        # 4/4 for one full bar, 3/4 from tick 1920 (a clean boundary).
        conductor = timesig(0, 4, 2) + timesig(1920, 3, 2) + EOT
        notes = (
            ev(1920, 0x90, 60, 60)   # m2 beat 1 in 3/4
            + ev(1440, 0x80, 60, 0)  # m3 beat 1 (3/4 bar = 1440)
            + EOT
        )
        data = header(1, 2, 480) + track(conductor) + track(notes)
        result = normalize(data, musical_position=True)
        self.assertEqual(
            positions(result),
            [(2, (3, 4), "1/1"), (3, (3, 4), "1/1")],
        )

    def test_midbar_change_truncates_and_counts_short_bar(self):
        # 4/4 at tick 0; 6/8 declared mid-bar at tick 720 (halfway through
        # beat 2).  The unfinished 4/4 bar still counts as measure 1, so the
        # 6/8 segment starts at measure 2.  Beats are denominator-note beats:
        # in 6/8 an eighth note (240 ticks at PPQN 480), six per 1440-tick bar.
        conductor = (
            timesig(0, 4, 2)
            + timesig(720, 6, 3)  # denominator exponent 3 -> 8
            + EOT
        )
        notes = (
            ev(0, 0x90, 60, 100)      # tick 0    m1 b1, meter 4/4
            + ev(480, 0x90, 61, 100)  # tick 480  m1 b2
            + ev(240, 0x90, 62, 100)  # tick 720  m2 b1, meter 6/8
            + ev(240, 0x90, 63, 100)  # tick 960  m2 b2
            + ev(1200, 0x90, 64, 100)  # tick 2160 m3 b1 (6/8 bar 1440)
            + EOT
        )
        data = header(1, 2, 480) + track(conductor) + track(notes)
        result = normalize(data, musical_position=True)
        self.assertEqual(
            positions(result),
            [
                (1, (4, 4), "1/1"),
                (1, (4, 4), "2/1"),
                (2, (6, 8), "1/1"),
                (2, (6, 8), "2/1"),
                (3, (6, 8), "1/1"),
            ],
        )

    def test_events_before_midbar_change_keep_old_meter(self):
        conductor = timesig(0, 4, 2) + timesig(960, 2, 2) + EOT
        notes = (
            ev(720, 0x90, 60, 100)   # tick 720: m1 b2.5 in 4/4
            + ev(240, 0x80, 60, 0)   # tick 960: change tick -> 2/4 m2 b1
            + EOT
        )
        data = header(1, 2, 480) + track(conductor) + track(notes)
        result = normalize(data, musical_position=True)
        self.assertEqual(
            positions(result),
            [(1, (4, 4), "5/2"), (2, (2, 4), "1/1")],
        )

    def test_same_tick_last_declaration_wins(self):
        conductor = (
            timesig(0, 3, 2)      # 3/4 first
            + timesig(0, 6, 3)    # 6/8 last -> wins
            + EOT
        )
        notes = ev(0, 0x90, 60, 100) + ev(720, 0x80, 60, 0) + EOT
        data = header(1, 2, 480) + track(conductor) + track(notes)
        result = normalize(data, musical_position=True)
        # 6/8: eighth-note beats 240 ticks apart -> tick 720 is beat 4.
        self.assertEqual(
            positions(result),
            [(1, (6, 8), "1/1"), (1, (6, 8), "4/1")],
        )

    def test_format_1_ignores_time_signatures_on_other_tracks(self):
        conductor = timesig(0, 4, 2) + EOT
        notes = timesig(0, 7, 3) + ev(480, 0x90, 60, 100) + EOT
        data = header(1, 2, 480) + track(conductor) + track(notes)
        result = normalize(data, musical_position=True)
        self.assertEqual(
            positions(result), [(1, (4, 4), "2/1")]
        )

    def test_non_integer_beats_per_bar_exact(self):
        # 1/1 at PPQN 48: bar = beat = 192 ticks.
        conductor = timesig(0, 1, 0) + EOT
        notes = ev(48, 0x90, 60, 100) + ev(144, 0x80, 60, 0) + EOT
        data = header(1, 2, 48) + track(conductor) + track(notes)
        result = normalize(data, musical_position=True)
        self.assertEqual(
            positions(result),
            [(1, (1, 1), "5/4"), (2, (1, 1), "1/1")],
        )

    def test_metric_segment_helpers(self):
        events = [
            TimeSignatureEvent(0, 3, 4, 0),
            TimeSignatureEvent(0, 2, 4, 5),   # same tick -> last wins
            TimeSignatureEvent(480, 6, 8, 9),
        ]
        segments = build_metric_segments(events)
        self.assertEqual(segments, [(0, 2, 4), (480, 6, 8)])
        # Tick 479 in the short 2/4 opening bar: m1, beat ~1.998.
        measure, num, den, beat = musical_position_at_tick(479, segments, 480)
        self.assertEqual((measure, num, den), (1, 2, 4))
        self.assertEqual(beat, Fraction(1) + Fraction(479, 480))
        # Tick 480 opens measure 2 in 6/8.
        measure, num, den, beat = musical_position_at_tick(480, segments, 480)
        self.assertEqual((measure, num, den, beat), (2, 6, 8, Fraction(1)))

    # -- strict validation (projection only) ------------------------------

    def _ts_payload(self, raw_bytes, delta=0):
        return varlen(delta) + raw_bytes

    def test_bad_time_signature_length(self):
        raw = bytes([0xFF, 0x58, 0x03, 4, 2, 24])  # one byte short
        payload = self._ts_payload(raw) + EOT
        with self.assertRaises(MidiError) as ctx:
            parse(header(0, 1, 480) + track(payload),
                  strict_time_signatures=True)
        self.assertEqual(ctx.exception.code, "bad_time_signature")
        self.assertEqual(ctx.exception.offset, 23)  # status byte offset

    def test_zero_numerator_rejected(self):
        payload = timesig(0, 0, 0) + EOT
        with self.assertRaises(MidiError) as ctx:
            parse(header(0, 1, 480) + track(payload),
                  strict_time_signatures=True)
        self.assertEqual(ctx.exception.code, "invalid_time_signature")

    def test_denominator_exponent_above_7_rejected(self):
        payload = timesig(0, 4, 8) + EOT  # 2^8 = 256th note
        with self.assertRaises(MidiError) as ctx:
            parse(header(0, 1, 480) + track(payload),
                  strict_time_signatures=True)
        self.assertEqual(ctx.exception.code, "invalid_time_signature")

    def test_denominator_exponent_boundaries_accepted(self):
        for exponent in (0, 7):
            payload = timesig(0, 1, exponent) + EOT
            parsed = parse(header(0, 1, 480) + track(payload),
                           strict_time_signatures=True)
            self.assertEqual(parsed.time_signature_events[0].denominator,
                             1 << exponent)

    def test_bad_time_signature_rejected_file_wide(self):
        conductor = EOT
        notes = timesig(0, 4, 9) + EOT  # bad even on a non-governing track
        data = header(1, 2, 480) + track(conductor) + track(notes)
        with self.assertRaises(MidiError) as ctx:
            parse(data, strict_time_signatures=True)
        self.assertEqual(ctx.exception.code, "invalid_time_signature")

    def test_legacy_mode_skips_time_signature_checks(self):
        # Malformed (3-byte) and weird (denom 256) events: fine without the
        # projection, and the response gains no musical_position field.
        conductor = (
            bytes([0x00, 0xFF, 0x58, 0x03, 4, 2, 24])
            + timesig(0, 4, 8)
            + EOT
        )
        notes = ev(480, 0x90, 60, 100) + EOT
        data = header(1, 2, 480) + track(conductor) + track(notes)
        result = normalize(data)
        self.assertNotIn("musical_position", result["events"][0])
        self.assertEqual(result["events"][0]["tick"], 480)


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


if __name__ == "__main__":
    unittest.main()
