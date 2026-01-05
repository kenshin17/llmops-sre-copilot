from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question or incident description")


class RunbookHit(BaseModel):
    id: str
    score: float
    text: str | None = None


class SearchResponse(BaseModel):
    query: str
    prompt_injection: bool
    pii_matches: list[str]
    results: list[RunbookHit]
    reason: str | None = None


class AnswerResponse(SearchResponse):
    answer: str
