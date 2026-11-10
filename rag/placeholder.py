"""Walking-skeleton placeholder. NOT a story. Delete once US-05/06/07 are wired up."""
from rag.types import Answer, Chunk, RetrievedChunk

PLACEHOLDER_NOTICE = "PLACEHOLDER - replace in US-05/06/07"


def placeholder_answer(question: str) -> Answer:
    chunk = Chunk(chunk_id="placeholder:0", doc_id="placeholder", text="Fake source excerpt.")
    return Answer(
        text=f"[{PLACEHOLDER_NOTICE}] You asked: {question!r}. The RAG pipeline is not implemented yet.",
        sources=[RetrievedChunk(chunk=chunk, score=1.0, source="Placeholder Protocol (fake)")],
        prompt_type="placeholder",
        metadata={"placeholder": True},
    )
