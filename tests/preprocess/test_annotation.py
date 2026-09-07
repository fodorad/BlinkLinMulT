"""Tests for the ``.tag`` annotation parser."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from blinklinmult.preprocess.annotation import (
    NO_BLINK,
    TagFile,
    TagParseError,
    parse_tag_file,
    parse_timestamps,
)

# frame:blink:NF:LE_FC:LE_NV:RE_FC:RE_NV:F_X:F_Y:F_W:F_H:LE(4):RE(4)
OPEN_LINE = "{fid}:-1:X:X:X:X:X:100:120:200:200:310:222:334:221:369:220:392:222"
BLINK_LINE = "{fid}:{bid}:X:C:X:C:X:100:120:200:200:310:222:334:221:369:220:392:222"


def tag_text(lines: list[str], header: str = "some header\n") -> str:
    """Wrap record lines in the #start/#end block."""
    return header + "#start\n" + "\n".join(lines) + "\n#end\n"


def simple_records(count: int = 10, blink_at: tuple[int, ...] = ()) -> list[str]:
    """Build `count` record lines, with blinks at the given indices."""
    lines = []
    for index in range(count):
        if index in blink_at:
            lines.append(BLINK_LINE.format(fid=index, bid=5))
        else:
            lines.append(OPEN_LINE.format(fid=index))
    return lines


class TempTagMixin(unittest.TestCase):
    """Writes .tag fixtures into a temporary directory."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def write_tag(self, text: str, name: str = "video1/rec.tag") -> Path:
        path = self.tmp / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path


class TestParseTagFile(TempTagMixin):
    def test_parses_every_record(self):
        path = self.write_tag(tag_text(simple_records(5)))
        records = parse_tag_file(path)
        self.assertEqual(len(records), 5)
        self.assertEqual([r.frame_id for r in records], list(range(5)))

    def test_open_frames_have_no_blink_id(self):
        path = self.write_tag(tag_text(simple_records(3)))
        records = parse_tag_file(path)
        self.assertTrue(all(r.blink_id == NO_BLINK for r in records))
        self.assertFalse(any(r.is_blink for r in records))

    def test_blink_flags_are_parsed(self):
        path = self.write_tag(tag_text(simple_records(3, blink_at=(1,))))
        records = parse_tag_file(path)
        self.assertTrue(records[1].is_blink)
        self.assertTrue(records[1].left_fully_closed)
        self.assertTrue(records[1].right_fully_closed)
        self.assertFalse(records[1].left_not_visible)
        self.assertFalse(records[0].left_fully_closed)

    def test_geometry_is_parsed(self):
        path = self.write_tag(tag_text(simple_records(1)))
        record = parse_tag_file(path)[0]
        np.testing.assert_array_equal(record.face_xywh, [100, 120, 200, 200])
        np.testing.assert_array_equal(record.left_eye_corners, [310, 222, 334, 221])
        np.testing.assert_array_equal(record.right_eye_corners, [369, 220, 392, 222])

    def test_has_face_box_is_false_for_all_zero_box(self):
        line = "0:-1:X:X:X:X:X:0:0:0:0:310:222:334:221:369:220:392:222"
        record = parse_tag_file(self.write_tag(tag_text([line])))[0]
        self.assertFalse(record.has_face_box)

    def test_blank_lines_are_skipped(self):
        path = self.write_tag(tag_text([*simple_records(2), "", "  "]))
        self.assertEqual(len(parse_tag_file(path)), 2)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            parse_tag_file(self.tmp / "nope.tag")

    def test_missing_start_token_raises(self):
        path = self.write_tag("header\n" + "\n".join(simple_records(2)) + "\n#end\n")
        with self.assertRaises(TagParseError) as ctx:
            parse_tag_file(path)
        self.assertIn("#start", str(ctx.exception))

    def test_end_before_start_raises(self):
        path = self.write_tag("#end\n" + OPEN_LINE.format(fid=0) + "\n#start\n")
        with self.assertRaises(TagParseError) as ctx:
            parse_tag_file(path)
        self.assertIn("precedes", str(ctx.exception))

    def test_empty_block_raises(self):
        with self.assertRaises(TagParseError):
            parse_tag_file(self.write_tag("#start\n#end\n"))

    def test_wrong_field_count_raises_with_location(self):
        path = self.write_tag(tag_text(["0:-1:X:X"]))
        with self.assertRaises(TagParseError) as ctx:
            parse_tag_file(path)
        message = str(ctx.exception)
        self.assertIn("19", message)
        # Line 1 is the header, line 2 is #start, so the bad record is line 3.
        self.assertIn("rec.tag:3", message)

    def test_non_integer_field_raises(self):
        bad = OPEN_LINE.format(fid=0).replace("100:120", "abc:120")
        with self.assertRaises(TagParseError) as ctx:
            parse_tag_file(self.write_tag(tag_text([bad])))
        self.assertIn("non-integer", str(ctx.exception))


class TestParseTimestamps(TempTagMixin):
    def test_parses_pairs(self):
        path = self.tmp / "ts.txt"
        path.write_text("0 0.0\n1 0.0333\n2 0.0667\n")
        self.assertEqual(parse_timestamps(path), {0: 0.0, 1: 0.0333, 2: 0.0667})

    def test_blank_lines_are_skipped(self):
        path = self.tmp / "ts.txt"
        path.write_text("0 0.0\n\n1 0.5\n")
        self.assertEqual(len(parse_timestamps(path)), 2)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            parse_timestamps(self.tmp / "nope.txt")

    def test_malformed_line_raises(self):
        path = self.tmp / "ts.txt"
        path.write_text("0 0.0 extra\n")
        with self.assertRaises(TagParseError):
            parse_timestamps(path)

    def test_empty_file_raises(self):
        path = self.tmp / "ts.txt"
        path.write_text("")
        with self.assertRaises(TagParseError):
            parse_timestamps(path)


class TestTagFile(TempTagMixin):
    def build(self, count: int = 10, blink_at: tuple[int, ...] = ()) -> TagFile:
        path = self.write_tag(tag_text(simple_records(count, blink_at)))
        return TagFile.from_path(path)

    def test_video_id_defaults_to_parent_directory(self):
        self.assertEqual(self.build().video_id, "video1")

    def test_len_is_record_count(self):
        self.assertEqual(len(self.build(7)), 7)

    def test_empty_records_rejected(self):
        with self.assertRaises(TagParseError):
            TagFile([], "x")

    def test_blink_presence_is_binary(self):
        tag = self.build(5, blink_at=(1, 2))
        np.testing.assert_array_equal(tag.blink_presence, [0, 1, 1, 0, 0])
        self.assertEqual(tag.blink_presence.dtype, np.float32)

    def test_eye_state_has_two_columns(self):
        tag = self.build(4, blink_at=(2,))
        self.assertEqual(tag.eye_state.shape, (4, 2))
        np.testing.assert_array_equal(tag.eye_state[2], [1.0, 1.0])
        np.testing.assert_array_equal(tag.eye_state[0], [0.0, 0.0])

    def test_validity_is_true_when_eyes_visible(self):
        tag = self.build(3)
        self.assertEqual(tag.validity.shape, (3, 2))
        self.assertTrue(tag.validity.all())

    def test_validity_false_where_eye_not_visible(self):
        line = "0:-1:X:X:N:X:X:100:120:200:200:310:222:334:221:369:220:392:222"
        tag = TagFile.from_path(self.write_tag(tag_text([line])))
        np.testing.assert_array_equal(tag.validity[0], [False, True])

    def test_frame_and_blink_id_arrays(self):
        tag = self.build(4, blink_at=(1,))
        np.testing.assert_array_equal(tag.frame_ids, [0, 1, 2, 3])
        np.testing.assert_array_equal(tag.blink_ids, [-1, 5, -1, -1])


class TestBlinkEvents(TempTagMixin):
    def build_from_ids(self, blink_ids: list[int]) -> TagFile:
        lines = [
            OPEN_LINE.format(fid=i) if bid == NO_BLINK else BLINK_LINE.format(fid=i, bid=bid)
            for i, bid in enumerate(blink_ids)
        ]
        return TagFile.from_path(self.write_tag(tag_text(lines)))

    def test_no_blinks_yields_no_events(self):
        self.assertEqual(self.build_from_ids([-1, -1, -1]).blink_events(), [])

    def test_single_run_is_one_event(self):
        events = self.build_from_ids([-1, 3, 3, 3, -1]).blink_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].start_index, 1)
        self.assertEqual(events[0].length, 3)
        self.assertEqual(events[0].stop_index, 4)
        self.assertEqual(events[0].first_frame_id, 1)

    def test_two_distinct_ids_are_two_events(self):
        events = self.build_from_ids([1, 1, -1, 2, 2]).blink_events()
        self.assertEqual([(e.blink_id, e.length) for e in events], [(1, 2), (2, 2)])

    def test_repeated_id_after_a_gap_is_two_events(self):
        # Grouping by id alone would merge these into one event spanning the gap.
        events = self.build_from_ids([7, 7, -1, 7, 7]).blink_events()
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].start_index, 0)
        self.assertEqual(events[1].start_index, 3)

    def test_adjacent_distinct_ids_split(self):
        events = self.build_from_ids([1, 1, 2, 2]).blink_events()
        self.assertEqual([(e.blink_id, e.start_index) for e in events], [(1, 0), (2, 2)])

    def test_event_at_the_very_end(self):
        events = self.build_from_ids([-1, -1, 4]).blink_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].stop_index, 3)


class TestAlignAndRestrict(TempTagMixin):
    def build(self, count: int = 5) -> TagFile:
        return TagFile.from_path(self.write_tag(tag_text(simple_records(count))))

    def test_align_is_a_no_op_when_ids_match(self):
        tag = self.build()
        tag.align_frame_ids(0)
        np.testing.assert_array_equal(tag.frame_ids, [0, 1, 2, 3, 4])

    def test_align_shifts_zero_based_annotation_onto_one_based_frames(self):
        tag = self.build()
        tag.align_frame_ids(1)
        np.testing.assert_array_equal(tag.frame_ids, [1, 2, 3, 4, 5])

    def test_align_handles_a_negative_offset(self):
        tag = self.build()
        tag.align_frame_ids(-2)
        np.testing.assert_array_equal(tag.frame_ids, [-2, -1, 0, 1, 2])

    def test_align_preserves_labels(self):
        path = self.write_tag(tag_text(simple_records(4, blink_at=(2,))))
        tag = TagFile.from_path(path)
        before = tag.blink_presence.copy()
        tag.align_frame_ids(10)
        np.testing.assert_array_equal(tag.blink_presence, before)

    def test_restrict_drops_unavailable_frames(self):
        tag = self.build()
        tag.restrict_to({0, 2, 4})
        np.testing.assert_array_equal(tag.frame_ids, [0, 2, 4])

    def test_restrict_keeps_everything_when_all_available(self):
        tag = self.build()
        tag.restrict_to(set(range(5)))
        self.assertEqual(len(tag), 5)

    def test_restrict_to_nothing_raises(self):
        tag = self.build()
        with self.assertRaises(TagParseError) as ctx:
            tag.restrict_to({999})
        self.assertIn("do not match", str(ctx.exception))


class TestSummary(TempTagMixin):
    def test_counts_frames_blink_frames_and_events(self):
        path = self.write_tag(tag_text(simple_records(10, blink_at=(2, 3, 7))))
        summary = TagFile.from_path(path).summary()
        self.assertEqual(summary["frames"], 10)
        self.assertEqual(summary["blink_frames"], 3)
        # Frames 2-3 are one run; frame 7 is another.
        self.assertEqual(summary["blink_events"], 2)


if __name__ == "__main__":
    unittest.main()
