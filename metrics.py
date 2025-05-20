# metrics.py
from prometheus_client import start_http_server, Counter, Histogram, Gauge
import time

# Métricas Prometheus
QUERY_COUNT = Counter('rag_queries_total', 'Total de queries RAG executadas')
QUERY_LATENCY = Histogram('rag_query_latency_seconds', 'Latência das queries RAG')
RESULT_COUNT = Gauge('rag_results_returned', 'Número de documentos retornados por query')

# Inicia servidor HTTP para métricas (porta configurável se desejar)
start_http_server(8000)


def record_metrics(func):
    """Decorator para medir execução de funções de busca RAG."""
    def wrapper(*args, **kwargs):
        QUERY_COUNT.inc()
        start = time.time()
        results = func(*args, **kwargs)
        duration = time.time() - start
        QUERY_LATENCY.observe(duration)
        # assume results is a list of docs
        try:
            count = len(results)
            RESULT_COUNT.set(count)
        except Exception:
            pass
        return results
    return wrapper