import unittest
from unittest.mock import patch

import main


class ShortsOutputTest(unittest.TestCase):
    @patch("main.append_report")
    @patch("main.analyze_audio", return_value=(0.0, False))
    @patch("main.render")
    @patch("main.get_video_info", return_value={"width": 1920, "height": 1080, "duration": 240.0})
    def test_process_task_caps_output_at_three_minutes(self, _info, render, _analyze, _report):
        self.assertTrue(main.process_task("video.mp4", "audio.wav", 0, 1))
        self.assertEqual(render.call_args.args[4], main.SHORTS_MAX_DURATION)


if __name__ == "__main__":
    unittest.main()
