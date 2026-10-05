"""Strict Standard MIDI File (SMF) parsing and timeline normalization.

Only formats 0 and 1 with a positive PPQN division are supported.  The
parser is deliberately strict: truncated data, malformed variable-length
integers, broken running status, illegal status bytes, bad event lengths
and undeclared trailing bytes are all rejected with a byte offset that
locates the problem.  No partial timeline is ever produced: either the
whole file parses and every channel event gets an exact microsecond
instant (a reduced fraction), or a single :class:`MidiError` is raised.

An optional ``musical_position`` projection adds, for every channel
event, the 1-based measure number, the time signature in effect and the
1-based beat position as a reduced fraction.  The initial time signature
is 4/4; only the first track's time-signature meta events build the
measure map (for format 0 that is the only track), same-tick
declarations resolve to the last one, and a change at a non-zero tick
opens a new measure there (the previous, possibly incomplete, measure
still counts).  With the projection enabled, time-signature meta events
are strictly validated: exactly 4 data bytes, a positive numerator and a
denominator exponent of 0-7.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import List, Tuple

MAX_FILE_BYTES = 1 << 20  # 1 MiB
MAX_TRACKS_PLUS_EVENTS = 10_000
DEFAULT_TEMPO_US_PER_QUARTER = 500_000

CHANNEL_EVENT_NAMES = {
    0x80: "note_off",
    0x90: "note_on",
    0xA0: "polyphonic_key_pressure",
    0xB0: "control_change",
    0xC0: "program_change",
    0xD0: "channel_pressure",
    0xE0: "pitch_bend",
}

_META_TEMPO = 0x51
_META_END_OF_TRACK = 0x2F
_META_TIME_SIGNATURE = 0x58

DEFAULT_NUMERATOR = 4
DEFAULT_DENOMINATOR_EXPONENT = 2  # 2**2 == 4, i.e. an initial 4/4 signature
MAX_DENOMINATOR_EXPONENT = 7  # denominators up to 2**7 == 128


class MidiError(Exception):
    """A structural MIDI problem located at an exact byte offset."""

    def __init__(self, code: str, message: str, offset: int) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.offset = offset


@dataclass
class ChannelEvent:
    tick: int
    track: int
    order: int  # ordinal among the channel events of this track
    kind: int  # high nibble of the status byte, e.g. 0x90
    channel: int
    data: Tuple[int, ...]


@dataclass
class TempoEvent:
    tick: int
    tempo_us: int


@dataclass
class TimeSignatureEvent:
    tick: int
    numerator: int
    denominator_exponent: int  # denominator is 2 ** exponent


@dataclass
class ParsedMidi:
    fmt: int
    ppqn: int
    ntracks: int
    channel_events: List[ChannelEvent] = field(default_factory=list)
    tempo_events: List[TempoEvent] = field(default_factory=list)  # track 0 only
    # Track 0 only, mirroring the tempo map; collected only when the
    # musical_position projection requested strict validation.
    time_signature_events: List[TimeSignatureEvent] = field(default_factory=list)


def _read_varlen(data: bytes, pos: int, end: int) -> Tuple[int, int]:
    """Read a 1-4 byte variable-length integer from ``data[pos:end]``."""
    start = pos
    value = 0
    for _ in range(4):
        if pos >= end:
            raise MidiError(
                "truncated_varlen",
                f"unterminated variable-length integer starting at offset {start}",
                start,
            )
        byte = data[pos]
        pos += 1
        value = (value << 7) | (byte & 0x7F)
        if byte < 0x80:
            return value, pos
    raise MidiError(
        "varlen_too_long",
        f"variable-length integer at offset {start} exceeds 4 bytes",
        start,
    )


def parse(data: bytes, check_time_signature: bool = False) -> ParsedMidi:
    """Parse and strictly validate a whole SMF file.

    With ``check_time_signature`` every time-signature meta event must
    carry exactly 4 data bytes, a positive numerator and a denominator
    exponent of 0-7, and the well-formed ones from the first track are
    collected for the measure map.  Without it, time-signature meta
    events are skipped like any other unhandled meta event.
    """
    if len(data) > MAX_FILE_BYTES:
        raise MidiError(
            "file_too_large",
            f"file is {len(data)} bytes, limit is {MAX_FILE_BYTES}",
            MAX_FILE_BYTES,
        )
    if len(data) < 14:
        raise MidiError(
            "truncated_header",
            f"file is {len(data)} bytes, a header chunk needs at least 14",
            len(data),
        )
    if data[0:4] != b"MThd":
        raise MidiError("bad_magic", "missing 'MThd' header magic", 0)
    header_len = int.from_bytes(data[4:8], "big")
    if header_len != 6:
        raise MidiError(
            "bad_header_length",
            f"header chunk length must be 6, got {header_len}",
            4,
        )
    fmt = int.from_bytes(data[8:10], "big")
    if fmt not in (0, 1):
        raise MidiError(
            "unsupported_format",
            f"only SMF formats 0 and 1 are supported, got {fmt}",
            8,
        )
    ntracks = int.from_bytes(data[10:12], "big")
    division = int.from_bytes(data[12:14], "big")
    if division & 0x8000:
        raise MidiError(
            "unsupported_division",
            "SMPTE time division is not supported; a positive PPQN is required",
            12,
        )
    if division == 0:
        raise MidiError("invalid_division", "PPQN division must be positive", 12)
    if ntracks == 0:
        raise MidiError("no_tracks", "file declares zero tracks", 10)
    if fmt == 0 and ntracks != 1:
        raise MidiError(
            "bad_track_count",
            f"format 0 must contain exactly one track, got {ntracks}",
            10,
        )
    if ntracks > MAX_TRACKS_PLUS_EVENTS:
        raise MidiError(
            "limit_exceeded",
            f"track count {ntracks} exceeds the limit of {MAX_TRACKS_PLUS_EVENTS} "
            "for tracks plus channel events",
            10,
        )

    parsed = ParsedMidi(fmt=fmt, ppqn=division, ntracks=ntracks)
    pos = 8 + header_len
    for track_index in range(ntracks):
        if pos + 8 > len(data):
            raise MidiError(
                "truncated_track_header",
                f"track {track_index}: missing 8-byte MTrk chunk header at offset {pos}",
                pos,
            )
        if data[pos : pos + 4] != b"MTrk":
            raise MidiError(
                "bad_chunk",
                f"track {track_index}: expected 'MTrk' chunk at offset {pos}, "
                f"found {data[pos:pos + 4]!r}",
                pos,
            )
        track_len = int.from_bytes(data[pos + 4 : pos + 8], "big")
        track_start = pos + 8
        track_end = track_start + track_len
        if track_end > len(data):
            raise MidiError(
                "truncated_track",
                f"track {track_index}: declared length {track_len} at offset "
                f"{pos + 4} exceeds the {len(data) - track_start} remaining bytes",
                pos + 4,
            )
        _parse_track(
            data, track_start, track_end, track_index, parsed,
            check_time_signature,
        )
        pos = track_end
    if pos != len(data):
        raise MidiError(
            "trailing_bytes",
            f"{len(data) - pos} undeclared trailing byte(s) after the last track",
            pos,
        )
    return parsed


def _parse_track(
    data: bytes,
    start: int,
    end: int,
    track_index: int,
    parsed: ParsedMidi,
    check_time_signature: bool,
) -> None:
    pos = start
    tick = 0
    running_status = None
    order = 0
    while pos < end:
        delta, pos = _read_varlen(data, pos, end)
        tick += delta
        if pos >= end:
            raise MidiError(
                "truncated_event",
                f"track {track_index}: event at tick {tick} is missing its "
                f"status byte at offset {pos}",
                pos,
            )
        status_offset = pos
        byte = data[pos]
        if byte >= 0x80:
            status = byte
            pos += 1
            # System-exclusive and meta events cancel running status.
            running_status = status if status < 0xF0 else None
        else:
            if running_status is None:
                raise MidiError(
                    "orphan_running_status",
                    f"track {track_index}: data byte 0x{byte:02X} at offset {pos} "
                    "has no running status",
                    pos,
                )
            status = running_status

        if status < 0xF0:
            kind = status & 0xF0
            needed = 1 if kind in (0xC0, 0xD0) else 2
            if pos + needed > end:
                raise MidiError(
                    "truncated_event",
                    f"track {track_index}: channel event at offset {status_offset} "
                    f"needs {needed} data byte(s), only {end - pos} remain",
                    pos,
                )
            payload = data[pos : pos + needed]
            for i, value in enumerate(payload):
                if value >= 0x80:
                    raise MidiError(
                        "invalid_data_byte",
                        f"track {track_index}: data byte 0x{value:02X} at offset "
                        f"{pos + i} has the high bit set",
                        pos + i,
                    )
            pos += needed
            if parsed.ntracks + len(parsed.channel_events) + 1 > MAX_TRACKS_PLUS_EVENTS:
                raise MidiError(
                    "limit_exceeded",
                    f"tracks plus channel events exceed the limit of "
                    f"{MAX_TRACKS_PLUS_EVENTS}",
                    status_offset,
                )
            parsed.channel_events.append(
                ChannelEvent(
                    tick=tick,
                    track=track_index,
                    order=order,
                    kind=kind,
                    channel=status & 0x0F,
                    data=tuple(payload),
                )
            )
            order += 1
        elif status == 0xFF:
            if pos >= end:
                raise MidiError(
                    "truncated_event",
                    f"track {track_index}: meta event at offset {status_offset} "
                    "is missing its type byte",
                    pos,
                )
            meta_type = data[pos]
            pos += 1
            meta_len, pos = _read_varlen(data, pos, end)
            if pos + meta_len > end:
                raise MidiError(
                    "truncated_event",
                    f"track {track_index}: meta event 0x{meta_type:02X} at offset "
                    f"{status_offset} declares {meta_len} byte(s), only "
                    f"{end - pos} remain",
                    pos,
                )
            payload = data[pos : pos + meta_len]
            pos += meta_len
            if meta_type == _META_TEMPO:
                if meta_len != 3:
                    raise MidiError(
                        "bad_meta_length",
                        f"track {track_index}: tempo meta event at offset "
                        f"{status_offset} must be 3 bytes, got {meta_len}",
                        status_offset,
                    )
                tempo_us = int.from_bytes(payload, "big")
                if tempo_us == 0:
                    raise MidiError(
                        "invalid_tempo",
                        f"track {track_index}: tempo at offset {status_offset} "
                        "must be positive",
                        status_offset,
                    )
                # Only the first track builds the global tempo map.  For
                # format 0 that is also the only track, so the rule is uniform.
                if track_index == 0:
                    parsed.tempo_events.append(TempoEvent(tick=tick, tempo_us=tempo_us))
            elif meta_type == _META_TIME_SIGNATURE and check_time_signature:
                if meta_len != 4:
                    raise MidiError(
                        "bad_meta_length",
                        f"track {track_index}: time signature meta event at offset "
                        f"{status_offset} must be 4 bytes, got {meta_len}",
                        status_offset,
                    )
                numerator, exponent = payload[0], payload[1]
                if numerator == 0:
                    raise MidiError(
                        "invalid_time_signature",
                        f"track {track_index}: time signature meta event at "
                        f"offset {status_offset} must have a positive numerator",
                        status_offset,
                    )
                if exponent > MAX_DENOMINATOR_EXPONENT:
                    raise MidiError(
                        "invalid_time_signature",
                        f"track {track_index}: time signature meta event at "
                        f"offset {status_offset} has denominator exponent "
                        f"{exponent}, the limit is {MAX_DENOMINATOR_EXPONENT}",
                        status_offset,
                    )
                # Only the first track builds the global measure map,
                # mirroring the tempo map rule.
                if track_index == 0:
                    parsed.time_signature_events.append(
                        TimeSignatureEvent(
                            tick=tick,
                            numerator=numerator,
                            denominator_exponent=exponent,
                        )
                    )
            elif meta_type == _META_END_OF_TRACK and meta_len != 0:
                raise MidiError(
                    "bad_meta_length",
                    f"track {track_index}: end-of-track meta event at offset "
                    f"{status_offset} must have length 0, got {meta_len}",
                    status_offset,
                )
        elif status in (0xF0, 0xF7):
            sysex_len, pos = _read_varlen(data, pos, end)
            if pos + sysex_len > end:
                raise MidiError(
                    "truncated_event",
                    f"track {track_index}: sysex event at offset {status_offset} "
                    f"declares {sysex_len} byte(s), only {end - pos} remain",
                    pos,
                )
            pos += sysex_len
        else:
            raise MidiError(
                "illegal_status",
                f"track {track_index}: status byte 0x{status:02X} at offset "
                f"{status_offset} is not allowed in a MIDI file",
                status_offset,
            )


def build_tempo_segments(tempo_events: List[TempoEvent]) -> List[Tuple[int, int]]:
    """Build (start_tick, tempo_us) segments from track-0 tempo events.

    A tempo event takes effect at its own tick; when several tempo events
    share a tick the last one wins.  The default is 500000 us per quarter.
    """
    segments: List[Tuple[int, int]] = [(0, DEFAULT_TEMPO_US_PER_QUARTER)]
    for event in tempo_events:  # ticks are non-decreasing within a track
        if event.tick < segments[-1][0]:
            continue  # defensive; cannot happen for cumulative parsing
        if event.tick == segments[-1][0]:
            segments[-1] = (event.tick, event.tempo_us)
        else:
            segments.append((event.tick, event.tempo_us))
    return segments


def time_at_tick(
    tick: int, segments: List[Tuple[int, int]], ppqn: int
) -> Fraction:
    """Exact time in microseconds of ``tick`` under the tempo segments."""
    total = Fraction(0, 1)
    for i, (seg_start, tempo_us) in enumerate(segments):
        if tick <= seg_start:
            break
        seg_end = segments[i + 1][0] if i + 1 < len(segments) else None
        stop = tick if seg_end is None else min(tick, seg_end)
        total += Fraction((stop - seg_start) * tempo_us, ppqn)
        if seg_end is None or tick <= seg_end:
            break
    return total


def build_measure_map(
    time_signature_events: List[TimeSignatureEvent], ppqn: int
) -> List[Tuple[int, int, int, int]]:
    """Build (start_tick, first_measure, numerator, denominator) segments.

    The initial time signature is 4/4.  A declaration takes effect at its
    own tick; when several declarations share a tick the last one wins.
    A change at a non-zero tick opens a new measure there, so the measure
    it cuts short still counts as one.
    """
    segments = [(0, DEFAULT_NUMERATOR, DEFAULT_DENOMINATOR_EXPONENT)]
    for event in time_signature_events:  # ticks are non-decreasing within a track
        if event.tick < segments[-1][0]:
            continue  # defensive; cannot happen for cumulative parsing
        if event.tick == segments[-1][0]:
            segments[-1] = (event.tick, event.numerator, event.denominator_exponent)
        else:
            segments.append((event.tick, event.numerator, event.denominator_exponent))
    measure_map: List[Tuple[int, int, int, int]] = []
    measure = 1
    for i, (tick, numerator, exponent) in enumerate(segments):
        measure_map.append((tick, measure, numerator, 1 << exponent))
        if i + 1 < len(segments):
            span = segments[i + 1][0] - tick
            # Measures opened inside this segment, including a final
            # incomplete one: ceil(span / measure_length_in_ticks) with
            # measure_length = 4 * ppqn * numerator / 2**exponent ticks.
            measure += -(-(span << exponent) // (4 * ppqn * numerator))
    return measure_map


def musical_position(
    tick: int, measure_map: List[Tuple[int, int, int, int]], ppqn: int
) -> Tuple[int, int, int, Fraction]:
    """Locate ``tick`` as (measure, numerator, denominator, beat).

    ``measure`` is the 1-based measure number, ``numerator`` and
    ``denominator`` describe the time signature in effect, and ``beat``
    is the 1-based position within the measure in units of the
    denominator note (an exact, reduced fraction; beat 1 is the
    downbeat).
    """
    lo, hi = 0, len(measure_map) - 1
    while lo < hi:  # last segment whose start is <= tick
        mid = (lo + hi + 1) // 2
        if measure_map[mid][0] <= tick:
            lo = mid
        else:
            hi = mid - 1
    seg_start, first_measure, numerator, denominator = measure_map[lo]
    measure_ticks = Fraction(4 * ppqn * numerator, denominator)
    offset = tick - seg_start
    elapsed = (offset * denominator) // (4 * ppqn * numerator)
    within = Fraction(offset, 1) - elapsed * measure_ticks
    beat = 1 + within * denominator / (4 * ppqn)
    return first_measure + elapsed, numerator, denominator, beat


def normalize(data: bytes, projection: str | None = None) -> dict:
    """Parse ``data`` and return the normalized channel-event timeline.

    With ``projection="musical_position"`` every event additionally
    carries a ``musical_position`` object (1-based measure, time
    signature in effect, 1-based beat as a reduced fraction), and
    time-signature meta events are strictly validated.  Any other
    projection value leaves the response untouched.
    """
    parsed = parse(data, check_time_signature=projection == "musical_position")
    segments = build_tempo_segments(parsed.tempo_events)
    ordered = sorted(
        parsed.channel_events, key=lambda e: (e.tick, e.track, e.order)
    )
    measure_map = None
    if projection == "musical_position":
        measure_map = build_measure_map(parsed.time_signature_events, parsed.ppqn)
    events = []
    for event in ordered:
        moment = time_at_tick(event.tick, segments, parsed.ppqn)
        entry = {
            "tick": event.tick,
            "track": event.track,
            "order": event.order,
            "type": CHANNEL_EVENT_NAMES[event.kind],
            "channel": event.channel,
            "data": list(event.data),
            "time_us": {
                "numerator": moment.numerator,
                "denominator": moment.denominator,
                "fraction": f"{moment.numerator}/{moment.denominator}",
            },
        }
        if measure_map is not None:
            measure, numerator, denominator, beat = musical_position(
                event.tick, measure_map, parsed.ppqn
            )
            entry["musical_position"] = {
                "measure": measure,
                "time_signature": {
                    "numerator": numerator,
                    "denominator": denominator,
                    "signature": f"{numerator}/{denominator}",
                },
                "beat": {
                    "numerator": beat.numerator,
                    "denominator": beat.denominator,
                    "fraction": f"{beat.numerator}/{beat.denominator}",
                },
            }
        events.append(entry)
    return {
        "format": parsed.fmt,
        "ppqn": parsed.ppqn,
        "track_count": parsed.ntracks,
        "channel_event_count": len(events),
        "events": events,
    }
