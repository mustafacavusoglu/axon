package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	InferenceRequests = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "inference_requests_total",
		Help: "Total inference requests",
	}, []string{"model", "version", "status"})

	InferenceLatency = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "inference_latency_ms",
		Help:    "End-to-end inference latency in ms",
		Buckets: prometheus.ExponentialBuckets(1, 2, 15),
	}, []string{"model", "version"})

	InferenceQueueDepth = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "inference_queue_depth",
		Help: "Pending requests in batcher",
	}, []string{"model"})

	ModelLoadEvents = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "model_load_total",
		Help: "Model load/unload events",
	}, []string{"model", "status"})

	EngineLatency = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "inference_engine_latency_ms",
		Help:    "Rust engine execution latency in ms",
		Buckets: prometheus.ExponentialBuckets(1, 2, 15),
	}, []string{"model", "version"})

	BatchSize = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "inference_batch_size",
		Help:    "Actual flushed batch sizes",
		Buckets: prometheus.LinearBuckets(1, 1, 64),
	}, []string{"model"})

	ActiveModels = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "inference_engine_models_loaded",
		Help: "Number of loaded models",
	})
)

func Init() {
	prometheus.MustRegister(
		InferenceRequests,
		InferenceLatency,
		InferenceQueueDepth,
		ModelLoadEvents,
		EngineLatency,
		BatchSize,
		ActiveModels,
	)
}
