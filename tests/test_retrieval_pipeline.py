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
                heading_path="Doc > Section",
                prev_heading=None,
                block_type="paragraph",
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
                heading_path="Doc > Section",
                prev_heading=None,
                block_type="paragraph",
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
                heading_path="Doc > Section",
                prev_heading=None,
                block_type="paragraph",
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

    def test_query_classifier_detects_intent_and_domain(self) -> None:
        service = RetrieverService.__new__(RetrieverService)
        profile = service._classify_query("Какие симптомы и рекомендации по лечению перечислены?")
        self.assertEqual(profile.intent, "enumeration")
        self.assertEqual(profile.domain, "medical")

    def test_structural_boost_prefers_list_for_enumeration(self) -> None:
        service = RetrieverService.__new__(RetrieverService)
        list_record = ChunkRecord(
            chunk_id="c1",
            parent_id="p1",
            file_name="doc.pdf",
            source_path="data/doc.pdf",
            parser_name="docling",
            doc_title="Doc",
            section_title="Symptoms",
            heading_path="Doc > Symptoms",
            prev_heading="Overview",
            block_type="list",
            page_span="unknown",
            raw_text="- fever\n- cough\n- pain",
            parent_text="- fever\n- cough\n- pain",
            child_position=0,
            child_count=1,
        )
        paragraph_record = ChunkRecord(
            chunk_id="c2",
            parent_id="p2",
            file_name="doc.pdf",
            source_path="data/doc.pdf",
            parser_name="docling",
            doc_title="Doc",
            section_title="Symptoms",
            heading_path="Doc > Symptoms",
            prev_heading="Overview",
            block_type="paragraph",
            page_span="unknown",
            raw_text="Symptoms may vary and should be evaluated clinically.",
            parent_text="Symptoms may vary and should be evaluated clinically.",
            child_position=0,
            child_count=1,
        )
        profile = service._classify_query("Какие симптомы перечислены?")
        self.assertGreater(
            service._structural_boost("Какие симптомы перечислены?", profile, list_record),
            service._structural_boost("Какие симптомы перечислены?", profile, paragraph_record),
        )


if __name__ == "__main__":
    unittest.main()
