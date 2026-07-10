import unittest
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from unittest.mock import patch

import main
import numpy as np


class ShortsOutputTest(unittest.TestCase):
    def test_reserves_distinct_output_paths_concurrently(self):
        with tempfile.TemporaryDirectory() as output_dir, \
             patch.object(main, "OUT_VIDEO_DIR", output_dir):
            with ThreadPoolExecutor(max_workers=2) as executor:
                paths = list(executor.map(
                    lambda _: main.unique_output_path("video", "00m15s", "music"),
                    range(2),
                ))

            self.assertEqual(len(set(paths)), 2)
            self.assertTrue(all(os.path.exists(path) for path in paths))

    def test_removes_reserved_output_when_render_fails(self):
        with tempfile.TemporaryDirectory() as output_dir, \
             patch.object(main, "OUT_VIDEO_DIR", output_dir):
            output_path = main.unique_output_path("video", "00m15s", "music")
            failed = subprocess.CompletedProcess([], 1, stderr="encoder error")
            with patch("main.subprocess.run", return_value=failed):
                with self.assertRaisesRegex(RuntimeError, "encoder error"):
                    main.render(
                        "video.mp4", "audio.wav", 0.0, False, 15.0,
                        output_path, {"width": 1920, "height": 1080},
                    )

            self.assertFalse(os.path.exists(output_path))

    def test_selects_distinct_strong_sections_after_second_video(self):
        scores = np.array([10.0, 9.0, 1.0, 8.0, 7.0, 1.0, 6.0])
        selected = [main._select_strong_window(scores, 2, i, np) for i in range(3)]
        self.assertEqual(selected, [0, 3, 6])

    @patch("main.normalize_terminal_return")
    @patch("builtins.input", side_effect=["abc", "5", ""])
    def test_video_count_menu_accepts_return_and_any_positive_number(self, _input, normalize):
        with patch("main.log_warning") as warning:
            self.assertEqual(main.ask_videos_per_music(), 5)
            self.assertEqual(main.ask_videos_per_music(), 2)
        warning.assert_called_once()
        self.assertEqual(normalize.call_count, 2)

    def test_tui_status_shows_paused_queue_message(self):
        output = StringIO()
        with patch("sys.stdout", output):
            main.tui_status("FILA PAUSADA", "Recursos insuficientes", main._WRN)
        self.assertIn("FILA PAUSADA", output.getvalue())
        self.assertIn("Recursos insuficientes", output.getvalue())

    def test_queue_pauses_only_above_ninety_percent_cpu(self):
        for cpu, should_run in ((90.0, True), (90.1, False)):
            main._throttle_event.set()
            with patch.object(main._monitor_stop, "is_set", side_effect=[False, True]), \
                 patch("main.psutil.cpu_percent", return_value=cpu), \
                 patch("main.tui_status"):
                main._resource_monitor()
            self.assertEqual(main._throttle_event.is_set(), should_run)

    @patch("main.append_report")
    @patch("main.analyze_audio", return_value=(0.0, False))
    @patch("main.render")
    @patch("main.unique_output_path", return_value="output.mp4")
    @patch("main.get_video_info", return_value={"width": 1920, "height": 1080, "duration": 240.0})
    def test_process_task_caps_output_at_three_minutes(
            self, _info, _output, render, _analyze, _report):
        self.assertTrue(main.process_task("video.mp4", "audio.wav", 0, 1))
        self.assertEqual(render.call_args.args[4], main.SHORTS_MAX_DURATION)


if __name__ == "__main__":
    unittest.main()
