import unittest

from veriflow_rag.retrieval.pipeline import ChunkRecord, RetrieverService


class RetrievalPipelineTests(unittest.TestCase):
    def test_expand_context_merges_neighboring_chunks(self) -> None:
        service = RetrieverService.__new__(RetrieverService)
        service.config = type("Config", (), {"expand_context_window": 1})()

        records = [
            ChunkRecord(
                chunk_id="c1",
                parent_id="p1",
                file_name="doc.pdf",
                source_path="data/doc.pdf",
                parser_name="docling",
                doc_title="Doc",
                section_title="Section",
                prev_heading=None,
                page_span="unknown",
                raw_text="alpha",
                parent_text="alpha beta gamma",
                child_position=0,
                child_count=3,
            ),
            ChunkRecord(
                chunk_id="c2",
                parent_id="p1",
                file_name="doc.pdf",
                source_path="data/doc.pdf",
                parser_name="docling",
                doc_title="Doc",
                section_title="Section",
                prev_heading=None,
                page_span="unknown",
                raw_text="beta",
                parent_text="alpha beta gamma",
                child_position=1,
                child_count=3,
            ),
            ChunkRecord(
                chunk_id="c3",
                parent_id="p1",
                file_name="doc.pdf",
                source_path="data/doc.pdf",
                parser_name="docling",
                doc_title="Doc",
                section_title="Section",
                prev_heading=None,
                page_span="unknown",
                raw_text="gamma",
                parent_text="alpha beta gamma",
                child_position=2,
                child_count=3,
            ),
        ]
        service.records_by_parent = {"p1": records}

        expanded = service._expand_context(records[1])
        self.assertEqual(expanded, "alpha\n\nbeta\n\ngamma")

    def test_confidence_label_uses_rerank_score_thresholds(self) -> None:
        self.assertEqual(RetrieverService._confidence_label(1, 0.9), "high")
        self.assertEqual(RetrieverService._confidence_label(2, 0.3), "medium")
        self.assertEqual(RetrieverService._confidence_label(5, -0.1), "low")
        self.assertEqual(RetrieverService._confidence_label(1, None), "baseline")


if __name__ == "__main__":
    unittest.main()
