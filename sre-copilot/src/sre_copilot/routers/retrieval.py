from fastapi import APIRouter, Depends

from sre_copilot.middleware.auth import verify_api_key
from sre_copilot.middleware.rate_limit import enforce_rate_limit
from sre_copilot.schemas.retrieval import (
    AnswerResponse,
    QueryRequest,
    SearchResponse,
)
from sre_copilot.services.retrieval import RetrievalService, get_retrieval_service
from sre_copilot.guardrails import GuardrailsEngine

router = APIRouter(dependencies=[Depends(verify_api_key), Depends(enforce_rate_limit)])


@router.post("/search", response_model=SearchResponse)
async def search_runbooks(
    payload: QueryRequest, svc: RetrievalService = Depends(get_retrieval_service)
) -> SearchResponse:
    result = await svc.search_runbooks(payload.query)
    return SearchResponse(**result)


@router.post("/answer", response_model=AnswerResponse)
async def answer(
    payload: QueryRequest, svc: RetrievalService = Depends(get_retrieval_service)
) -> AnswerResponse:
    result = await svc.answer(payload.query)
    return AnswerResponse(**result)
