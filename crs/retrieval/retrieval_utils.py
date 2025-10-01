"""Retrieval utilities with caching and error handling."""

import logging
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from .retrieval import ItemRetriever

# Set up logging
logger = logging.getLogger(__name__)

# Global retriever cache
_retriever_cache: Dict[str, ItemRetriever] = {}


class RetrievalError(Exception):
    """Custom exception for retrieval errors."""

    pass


def get_retriever(
    domain: str = "default", force_reload: bool = False
) -> ItemRetriever:
    """Get a cached retriever instance for the given domain.

    Args:
        domain: Domain identifier (for future domain-specific retrievers)
        force_reload: Whether to force reload the retriever

    Returns:
        ItemRetriever instance

    Raises:
        RetrievalError: If retriever cannot be initialized
    """
    global _retriever_cache

    cache_key = f"{domain}"

    if force_reload or cache_key not in _retriever_cache:
        try:
            logger.info(f"Initializing retriever for domain: {domain}")

            retriever = ItemRetriever()
            _retriever_cache[cache_key] = retriever

            logger.info(
                f"Retriever initialized successfully for domain: {domain}"
            )

        except Exception as e:
            logger.error(
                f"Failed to initialize retriever for domain {domain}: {e}"
            )
            raise RetrievalError(f"Failed to initialize retriever: {e}")

    return _retriever_cache[cache_key]


def safe_retrieve(
    query: str, domain: str = "default", top_k: int = 5
) -> List[Tuple[Dict[str, Any], float]]:
    """Safely retrieve items with proper error handling.

    Args:
        query: Search query
        domain: Domain identifier
        top_k: Number of items to retrieve

    Returns:
        List of (metadata, score) tuples

    Raises:
        RetrievalError: If retrieval fails
    """
    try:
        retriever = get_retriever(domain)
        results = retriever.retrieve(query, top_k=top_k)

        logger.debug(f"Retrieved {len(results)} items for query: '{query}'")
        return results

    except Exception as e:
        logger.error(f"Retrieval failed for query '{query}': {e}")
        raise RetrievalError(f"Retrieval failed: {e}")


def clear_retriever_cache():
    """Clear the retriever cache (useful for testing or memory management)."""
    global _retriever_cache
    _retriever_cache.clear()
    logger.info("Retriever cache cleared")


# Convenience function for backward compatibility
def get_cached_retriever() -> Optional[ItemRetriever]:
    """Get the default cached retriever, if available."""
    global _retriever_cache
    return _retriever_cache.get("default")
