"""Retrieval quality monitoring for Vectorize queries.

Tracks query performance, result quality, and namespace hit rates to
ensure the episodic memory system is working effectively.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class QueryLog:
    """Log entry for a single Vectorize query."""

    timestamp: datetime
    namespace: str
    filters_used: list[str]
    results_count: int
    top_similarity_score: float
    latency_ms: float


@dataclass
class RetrievalMetrics:
    """Aggregated retrieval metrics for monitoring."""

    query_count: int
    avg_similarity_score: float
    avg_results_returned: float
    namespace_hit_rate: dict[str, float]
    filter_usage: dict[str, int]
    avg_latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "query_count": self.query_count,
            "avg_similarity_score": self.avg_similarity_score,
            "avg_results_returned": self.avg_results_returned,
            "namespace_hit_rate": self.namespace_hit_rate,
            "filter_usage": self.filter_usage,
            "avg_latency_ms": self.avg_latency_ms,
        }


class RetrievalMonitor:
    """Monitor for tracking Vectorize retrieval quality.

    Tracks:
    - Query counts and latency
    - Similarity score distribution
    - Namespace utilization
    - Filter usage patterns

    Used to detect:
    - Quality degradation (low similarity scores)
    - Underutilized namespaces
    - Query performance issues
    """

    def __init__(self, d1_binding: Any | None = None):
        """Initialize the retrieval monitor.

        Args:
            d1_binding: Optional D1 database binding for persistent logging
        """
        self.db = d1_binding
        self._query_logs: list[QueryLog] = []
        self._max_logs = 1000  # Keep last 1000 queries in memory

    async def log_query(
        self,
        namespace: str,
        filters_used: list[str],
        results_count: int,
        top_similarity_score: float,
        latency_ms: float,
    ) -> None:
        """Log a Vectorize query for monitoring.

        Args:
            namespace: The namespace queried (e.g., "spy-trades")
            filters_used: List of metadata filters applied
            results_count: Number of results returned
            top_similarity_score: Highest similarity score in results
            latency_ms: Query latency in milliseconds
        """
        log_entry = QueryLog(
            timestamp=datetime.now(),
            namespace=namespace,
            filters_used=filters_used,
            results_count=results_count,
            top_similarity_score=top_similarity_score,
            latency_ms=latency_ms,
        )

        self._query_logs.append(log_entry)

        # Trim to max size
        if len(self._query_logs) > self._max_logs:
            self._query_logs = self._query_logs[-self._max_logs:]

        # Persist to D1 if available
        if self.db:
            await self._persist_log(log_entry)

        # Log quality alerts
        if top_similarity_score < 0.5 and results_count > 0:
            logger.warning(
                f"Low similarity query: namespace={namespace}, "
                f"top_score={top_similarity_score:.3f}, count={results_count}"
            )

    async def _persist_log(self, log_entry: QueryLog) -> None:
        """Persist log entry to D1 database."""
        await self.db.prepare("""
            INSERT INTO retrieval_logs (
                timestamp, namespace, filters_used, results_count,
                top_similarity_score, latency_ms
            ) VALUES (?, ?, ?, ?, ?, ?)
        """).bind(
            log_entry.timestamp.isoformat(),
            log_entry.namespace,
            ",".join(log_entry.filters_used),
            log_entry.results_count,
            log_entry.top_similarity_score,
            log_entry.latency_ms,
        ).run()

    async def get_daily_metrics(self, date: str | None = None) -> RetrievalMetrics:
        """Get aggregated metrics for a day.

        Args:
            date: Date string (YYYY-MM-DD), defaults to today

        Returns:
            RetrievalMetrics with aggregated statistics
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        # Filter logs for the specified date
        day_logs = [
            log for log in self._query_logs
            if log.timestamp.strftime("%Y-%m-%d") == date
        ]

        if not day_logs:
            return RetrievalMetrics(
                query_count=0,
                avg_similarity_score=0.0,
                avg_results_returned=0.0,
                namespace_hit_rate={},
                filter_usage={},
                avg_latency_ms=0.0,
            )

        # Calculate aggregates
        query_count = len(day_logs)
        avg_similarity = sum(log.top_similarity_score for log in day_logs) / query_count
        avg_results = sum(log.results_count for log in day_logs) / query_count
        avg_latency = sum(log.latency_ms for log in day_logs) / query_count

        # Namespace hit rate (queries with results / total queries per namespace)
        namespace_counts: dict[str, dict[str, int]] = {}
        for log in day_logs:
            if log.namespace not in namespace_counts:
                namespace_counts[log.namespace] = {"total": 0, "with_results": 0}
            namespace_counts[log.namespace]["total"] += 1
            if log.results_count > 0:
                namespace_counts[log.namespace]["with_results"] += 1

        namespace_hit_rate = {
            ns: counts["with_results"] / counts["total"]
            for ns, counts in namespace_counts.items()
        }

        # Filter usage counts
        filter_usage: dict[str, int] = {}
        for log in day_logs:
            for filter_name in log.filters_used:
                filter_usage[filter_name] = filter_usage.get(filter_name, 0) + 1

        return RetrievalMetrics(
            query_count=query_count,
            avg_similarity_score=avg_similarity,
            avg_results_returned=avg_results,
            namespace_hit_rate=namespace_hit_rate,
            filter_usage=filter_usage,
            avg_latency_ms=avg_latency,
        )

    async def check_quality_alert(
        self,
        min_similarity_threshold: float = 0.6,
        min_hit_rate_threshold: float = 0.8,
    ) -> list[str]:
        """Check for quality issues that need attention.

        Args:
            min_similarity_threshold: Alert if avg similarity below this
            min_hit_rate_threshold: Alert if namespace hit rate below this

        Returns:
            List of alert messages (empty if no issues)
        """
        metrics = await self.get_daily_metrics()
        alerts = []

        if metrics.query_count == 0:
            return alerts

        # Check similarity threshold
        if metrics.avg_similarity_score < min_similarity_threshold:
            alerts.append(
                f"Low average similarity: {metrics.avg_similarity_score:.3f} "
                f"(threshold: {min_similarity_threshold})"
            )

        # Check namespace hit rates
        for namespace, hit_rate in metrics.namespace_hit_rate.items():
            if hit_rate < min_hit_rate_threshold:
                alerts.append(
                    f"Low hit rate for {namespace}: {hit_rate:.1%} "
                    f"(threshold: {min_hit_rate_threshold:.1%})"
                )

        # Check for high latency
        if metrics.avg_latency_ms > 500:
            alerts.append(
                f"High average latency: {metrics.avg_latency_ms:.1f}ms"
            )

        return alerts

    def get_namespace_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics per namespace.

        Returns:
            Dict mapping namespace to stats (query_count, avg_similarity, avg_results)
        """
        namespace_logs: dict[str, list[QueryLog]] = {}
        for log in self._query_logs:
            if log.namespace not in namespace_logs:
                namespace_logs[log.namespace] = []
            namespace_logs[log.namespace].append(log)

        stats = {}
        for namespace, logs in namespace_logs.items():
            count = len(logs)
            stats[namespace] = {
                "query_count": count,
                "avg_similarity": sum(l.top_similarity_score for l in logs) / count,
                "avg_results": sum(l.results_count for l in logs) / count,
                "avg_latency_ms": sum(l.latency_ms for l in logs) / count,
            }

        return stats
