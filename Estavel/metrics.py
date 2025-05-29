#metrics.py
from prometheus_client import start_http_server, Counter, Histogram, Gauge
import time

# ===============================================================
# Métricas Prometheus para Monitoramento de Busca Semântica (RAG)
# ===============================================================

# Número total de execuções de busca RAG invocadas
QUERY_EXECUTIONS = Counter(
    'rag_query_executions_total',
    'Contagem total de buscas RAG executadas'
)

# Histogram de duração das execuções de busca RAG em segundos
QUERY_DURATION = Histogram(
    'rag_query_duration_seconds',
    'Distribuição das durações das buscas RAG em segundos',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
)

# Gauge para número de resultados retornados na última busca RAG
LAST_QUERY_RESULT_COUNT = Gauge(
    'rag_last_query_result_count',
    'Número de documentos retornados pela última busca RAG'
)

# Inicia servidor HTTP para expor métricas em /metrics (porta 8000)
start_http_server(8000)


def record_metrics(func):
    """Decorator para coletar métricas de execuções de busca RAG."""
    def wrapper(*args, **kwargs):
        # Incrementa contador de execuções
        QUERY_EXECUTIONS.inc()

        # Mede duração da função
        start_time = time.time()
        results = func(*args, **kwargs)
        elapsed = time.time() - start_time
        QUERY_DURATION.observe(elapsed)

        # Atualiza gauge de resultados retornados
        if isinstance(results, list):
            LAST_QUERY_RESULT_COUNT.set(len(results))

        return results

    return wrapper