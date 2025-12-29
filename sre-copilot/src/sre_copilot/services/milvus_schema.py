from __future__ import annotations

import asyncio
from typing import Any

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from sre_copilot.config import Settings
from sre_copilot.utils.logger import get_logger

logger = get_logger(__name__)


def _connect(settings: Settings) -> None:
    connections.connect(
        alias="default",
        uri=settings.milvus_uri,
        user=settings.milvus_user,
        password=settings.milvus_password,
    )


def build_collection(settings: Settings) -> Collection:
    _connect(settings)
    if utility.has_collection(settings.milvus_collection):
        return Collection(settings.milvus_collection)

    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1536),
    ]
    schema = CollectionSchema(fields=fields, description="Runbook embeddings")
    collection = Collection(name=settings.milvus_collection, schema=schema)
    collection.create_index(
        field_name="vector",
        index_params={"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 1024}},
    )
    collection.load()
    logger.info("Created Milvus collection %s", settings.milvus_collection)
    return collection


def insert_documents(collection: Collection, docs: list[dict[str, Any]]) -> None:
    collection.insert(
        [
            [doc["id"] for doc in docs],
            [doc["text"] for doc in docs],
            [doc["vector"] for doc in docs],
        ]
    )
    collection.flush()


async def ensure_collection(settings: Settings) -> None:
    await asyncio.to_thread(build_collection, settings)
