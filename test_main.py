import unittest
from io import StringIO
from unittest.mock import patch

import main


class ShortsOutputTest(unittest.TestCase):
    @patch("builtins.input", side_effect=["abc", "5", ""])
    def test_video_count_menu_accepts_return_and_any_positive_number(self, _input):
        with patch("main.log_warning") as warning:
            self.assertEqual(main.ask_videos_per_music(), 5)
            self.assertEqual(main.ask_videos_per_music(), 2)
        warning.assert_called_once()

    def test_tui_status_shows_paused_queue_message(self):
        output = StringIO()
        with patch("sys.stdout", output):
            main.tui_status("FILA PAUSADA", "Recursos insuficientes", main._WRN)
        self.assertIn("FILA PAUSADA", output.getvalue())
        self.assertIn("Recursos insuficientes", output.getvalue())

    @patch("main.append_report")
    @patch("main.analyze_audio", return_value=(0.0, False))
    @patch("main.render")
    @patch("main.get_video_info", return_value={"width": 1920, "height": 1080, "duration": 240.0})
    def test_process_task_caps_output_at_three_minutes(self, _info, render, _analyze, _report):
        self.assertTrue(main.process_task("video.mp4", "audio.wav", 0, 1))
        self.assertEqual(render.call_args.args[4], main.SHORTS_MAX_DURATION)


if __name__ == "__main__":
    unittest.main()
