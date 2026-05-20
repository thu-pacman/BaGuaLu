import tempfile
import unittest
from pathlib import Path


class MoeTokenPressureTest(unittest.TestCase):
    def test_moe_token_event_format_includes_lifecycle_identity(self):
        from bgl2.adapter import moe_imbalance

        self.assertEqual(
            moe_imbalance._format_moe_token_event(
                event_id=123,
                iteration=7,
                microbatch=3,
                phase="forward",
                event="alloc",
                layer=6,
                rank=24,
                received_tokens=3755,
            ),
            "moe_token_event: event_id=123, step=7, iteration=7, microbatch=3, "
            "phase=forward, event=alloc, layer=6, rank=24, received_tokens=3755",
        )

    def test_parser_reconstructs_live_tokens_from_alloc_free_events(self):
        from bgl2.data.moe_token_pressure import (
            build_live_token_samples,
            parse_moe_log,
            summarize_live_ranks,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "moe.log"
            log_path.write_text(
                "\n".join(
                    [
                        "moe_token_event: event_id=10, step=7, iteration=7, "
                        "microbatch=3, phase=forward, event=alloc, layer=6, "
                        "rank=24, received_tokens=100",
                        "moe_token_event: event_id=10, step=7, iteration=7, "
                        "microbatch=3, phase=backward, event=free, layer=6, "
                        "rank=24, received_tokens=100",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            parse_result = parse_moe_log(log_path)
            samples = build_live_token_samples(parse_result.token_records)
            summaries = summarize_live_ranks(samples)

        self.assertEqual(len(parse_result.token_records), 2)
        self.assertEqual([sample.live_tokens for sample in samples], [100, 0])
        self.assertEqual(summaries[0].rank, 24)
        self.assertEqual(summaries[0].peak_live_tokens, 100)
        self.assertEqual(summaries[0].final_live_tokens, 0)

    def test_parser_keeps_legacy_token_lines_as_pressure_records(self):
        from bgl2.data.moe_token_pressure import parse_moe_log, pressure_records

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "legacy.log"
            log_path.write_text(
                "| step: 0, layer: 6, rank: 24, received_tokens: 3755\n",
                encoding="utf-8",
            )

            parse_result = parse_moe_log(log_path)
            records = pressure_records(parse_result.token_records)

        self.assertEqual(len(records), 1)
        self.assertIsNone(records[0].event)
        self.assertEqual(records[0].received_tokens, 3755)

    def test_pressure_records_prefer_structured_events_when_mixed_with_legacy_lines(self):
        from bgl2.data.moe_token_pressure import parse_moe_log, pressure_records

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "mixed.log"
            log_path.write_text(
                "\n".join(
                    [
                        "moe_token_event: event_id=10, step=7, iteration=7, "
                        "microbatch=3, phase=forward, event=alloc, layer=6, "
                        "rank=24, received_tokens=100",
                        "| step: 7, layer: 6, rank: 24, received_tokens: 100",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            records = pressure_records(parse_moe_log(log_path).token_records)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].event, "alloc")

    def test_lifecycle_issues_reports_unmatched_allocs(self):
        from bgl2.data.moe_token_pressure import lifecycle_issues, parse_moe_log

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "truncated.log"
            log_path.write_text(
                "moe_token_event: event_id=10, step=7, iteration=7, "
                "microbatch=3, phase=forward, event=alloc, layer=6, "
                "rank=24, received_tokens=100\n",
                encoding="utf-8",
            )

            issues = lifecycle_issues(parse_moe_log(log_path).token_records)

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].rank, 24)
        self.assertEqual(issues[0].event_id, 10)
        self.assertEqual(issues[0].net_tokens, 100)
        self.assertEqual(issues[0].status, "alloc_without_free")


if __name__ == "__main__":
    unittest.main()
