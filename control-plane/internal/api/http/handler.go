package http

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"time"

	"github.com/gofiber/fiber/v2"
	enginev1 "github.com/mustafacavusoglu/axon/control-plane/inference/engine/v1"

	"github.com/mustafacavusoglu/axon/control-plane/internal/client"
	"github.com/mustafacavusoglu/axon/control-plane/internal/health"
	"github.com/mustafacavusoglu/axon/control-plane/internal/manager"
	"github.com/mustafacavusoglu/axon/control-plane/internal/metrics"
)

type Handler struct {
	registry  *manager.ModelRegistry
	lifecycle *manager.LifecycleManager
	client    *client.InferenceClient
	checker   *health.Checker
}

func NewHandler(
	registry *manager.ModelRegistry,
	lifecycle *manager.LifecycleManager,
	client *client.InferenceClient,
	checker *health.Checker,
) *Handler {
	return &Handler{
		registry:  registry,
		lifecycle: lifecycle,
		client:    client,
		checker:   checker,
	}
}

func (h *Handler) RegisterRoutes(app *fiber.App) {
	v2 := app.Group("/v2")

	v2.Get("/health/live", h.Live)
	v2.Get("/health/ready", h.Ready)
	v2.Get("/models", h.ListModels)
	v2.Get("/models/:name", h.ModelMeta)
	v2.Get("/models/:name/versions/:version", h.ModelVersion)
	v2.Post("/models/:name/load", h.LoadModel)
	v2.Post("/models/:name/unload", h.UnloadModel)
	v2.Post("/models/:name/infer", h.Infer)
}

func (h *Handler) Live(c *fiber.Ctx) error {
	if h.checker.IsLive() {
		return c.JSON(fiber.Map{"live": true})
	}
	return c.Status(fiber.StatusServiceUnavailable).JSON(fiber.Map{"live": false})
}

func (h *Handler) Ready(c *fiber.Ctx) error {
	if h.checker.IsReady() {
		return c.JSON(fiber.Map{"ready": true})
	}
	return c.Status(fiber.StatusServiceUnavailable).JSON(fiber.Map{"ready": false})
}

func (h *Handler) ListModels(c *fiber.Ctx) error {
	names := h.registry.AllNames()
	if names == nil {
		names = []string{}
	}
	return c.JSON(fiber.Map{
		"models": names,
	})
}

func (h *Handler) ModelMeta(c *fiber.Ctx) error {
	name := c.Params("name")
	entry, found := h.registry.GetLatestVersion(name)
	if !found {
		return c.Status(fiber.StatusNotFound).JSON(fiber.Map{
			"error": fmt.Sprintf("model %s not found", name),
		})
	}

	return c.JSON(fiber.Map{
		"name":     name,
		"versions": h.registry.GetAvailableVersions(name),
		"platform": entry.Config.Platform,
	})
}

func (h *Handler) ModelVersion(c *fiber.Ctx) error {
	name := c.Params("name")
	versionStr := c.Params("version")
	version, err := strconv.Atoi(versionStr)
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": "invalid version",
		})
	}

	entry, found := h.registry.Get(name, version)
	if !found || entry.State != manager.StateReady {
		return c.Status(fiber.StatusNotFound).JSON(fiber.Map{
			"error": fmt.Sprintf("model %s version %d not found", name, version),
		})
	}

	return c.JSON(fiber.Map{
		"name":     name,
		"version":  version,
		"state":    entry.State.String(),
		"platform": entry.Config.Platform,
	})
}

func (h *Handler) LoadModel(c *fiber.Ctx) error {
	name := c.Params("name")

	var req struct {
		Version int `json:"version"`
	}
	if err := c.BodyParser(&req); err != nil {
		req.Version = 1
	}
	if req.Version == 0 {
		req.Version = 1
	}

	if err := h.lifecycle.LoadModel(name, req.Version); err != nil {
		metrics.ModelLoadEvents.WithLabelValues(name, "error").Inc()
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": err.Error(),
		})
	}

	metrics.ModelLoadEvents.WithLabelValues(name, "loaded").Inc()
	return c.JSON(fiber.Map{"status": "loaded"})
}

func (h *Handler) UnloadModel(c *fiber.Ctx) error {
	name := c.Params("name")

	var req struct {
		Version int `json:"version"`
	}
	if err := c.BodyParser(&req); err != nil {
		req.Version = 1
	}

	if err := h.lifecycle.UnloadModel(name, req.Version); err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": err.Error(),
		})
	}

	metrics.ModelLoadEvents.WithLabelValues(name, "unloaded").Inc()
	return c.JSON(fiber.Map{"status": "unloaded"})
}

func (h *Handler) Infer(c *fiber.Ctx) error {
	name := c.Params("name")

	var req inferRequest
	if err := json.Unmarshal(c.Body(), &req); err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
			"error": fmt.Sprintf("invalid request body: %v", err),
		})
	}

	version := 1
	if req.ModelVersion != "" {
		if v, err := strconv.Atoi(req.ModelVersion); err == nil {
			version = v
		}
	}

	entry, found := h.registry.Get(name, version)
	if !found || entry.State != manager.StateReady {
		return c.Status(fiber.StatusNotFound).JSON(fiber.Map{
			"error": fmt.Sprintf("model %s version %d not ready", name, version),
		})
	}

	h.registry.Touch(name, version)

	var internalInputs []*enginev1.InferInput
	for _, inp := range req.Inputs {
		rawBytes, err := inp.dataToBytes()
		if err != nil {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"error": fmt.Sprintf("input %s: %v", inp.Name, err),
			})
		}

		shape := inp.Shape
		if entry.Config.MaxBatchSize > 1 && len(shape) > 0 && len(entry.Config.Inputs) == 1 {
			shape = append([]int64{1}, shape...)
		}

		dtype := dtypeToInternal(inp.Datatype)
		internalInputs = append(internalInputs, &enginev1.InferInput{
			Name:  inp.Name,
			Shape: shape,
			Data:  rawBytes,
			Dtype: dtype,
		})
	}

	start := time.Now()
	resp, err := h.client.BatchInfer(c.Context(), name, uint32(version), internalInputs, 1)
	elapsed := time.Since(start).Milliseconds()

	if err != nil {
		metrics.InferenceRequests.WithLabelValues(name, strconv.Itoa(version), "error").Inc()
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": fmt.Sprintf("inference failed: %v", err),
		})
	}

	metrics.InferenceRequests.WithLabelValues(name, strconv.Itoa(version), "ok").Inc()
	metrics.InferenceLatency.WithLabelValues(name, strconv.Itoa(version)).Observe(float64(elapsed))
	metrics.EngineLatency.WithLabelValues(name, strconv.Itoa(version)).Observe(resp.LatencyMs)

	modelVersionStr := strconv.Itoa(version)
	return c.JSON(inferResponse{
		Id:           req.Id,
		ModelName:    name,
		ModelVersion: modelVersionStr,
		Outputs:      buildOutputs(resp.Outputs),
	})
}

type inferRequest struct {
	Id           string        `json:"id"`
	ModelVersion string        `json:"model_version"`
	Inputs       []inferInput  `json:"inputs"`
}

type inferInput struct {
	Name     string    `json:"name"`
	Shape    []int64   `json:"shape"`
	Datatype string    `json:"datatype"`
	Data     []float64 `json:"data"`
	RawData  []byte    `json:"raw_data"`
}

func (inp *inferInput) dataToBytes() ([]byte, error) {
	if len(inp.RawData) > 0 {
		return inp.RawData, nil
	}
	if len(inp.Data) > 0 {
		switch inp.Datatype {
		case "FP32":
			bytes := make([]byte, len(inp.Data)*4)
			for i, v := range inp.Data {
				bits := math.Float32bits(float32(v))
				binary.LittleEndian.PutUint32(bytes[i*4:], bits)
			}
			return bytes, nil
		case "INT64":
			bytes := make([]byte, len(inp.Data)*8)
			for i, v := range inp.Data {
				binary.LittleEndian.PutUint64(bytes[i*8:], uint64(int64(v)))
			}
			return bytes, nil
		case "INT32":
			bytes := make([]byte, len(inp.Data)*4)
			for i, v := range inp.Data {
				binary.LittleEndian.PutUint32(bytes[i*4:], uint32(int32(v)))
			}
			return bytes, nil
		default:
			return nil, fmt.Errorf("unsupported dtype %s for JSON data", inp.Datatype)
		}
	}
	return nil, fmt.Errorf("no data provided")
}

type inferResponse struct {
	Id           string         `json:"id"`
	ModelName    string         `json:"model_name"`
	ModelVersion string         `json:"model_version"`
	Outputs      []inferOutput  `json:"outputs"`
}

type inferOutput struct {
	Name     string        `json:"name"`
	Datatype string        `json:"datatype"`
	Shape    []int64       `json:"shape"`
	Data     []interface{} `json:"data"`
}

func buildOutputs(outputs []*enginev1.InferOutput) []inferOutput {
	var result []inferOutput
	for _, out := range outputs {
		o := inferOutput{
			Name:  out.Name,
			Shape: out.Shape,
		}
		dt := internalDtypeToString(enginev1.DataType(out.Dtype))
		o.Datatype = dt

		switch {
		case dt == "FP32" && len(out.Data) >= 4:
			n := len(out.Data) / 4
			o.Data = make([]interface{}, n)
			for i := 0; i < n; i++ {
				bits := binary.LittleEndian.Uint32(out.Data[i*4 : i*4+4])
				o.Data[i] = float64(math.Float32frombits(bits))
			}

		case dt == "INT64" && len(out.Data) >= 8:
			n := len(out.Data) / 8
			o.Data = make([]interface{}, n)
			for i := 0; i < n; i++ {
				val := int64(binary.LittleEndian.Uint64(out.Data[i*8 : i*8+8]))
				o.Data[i] = val
			}

		case dt == "INT32" && len(out.Data) >= 4:
			n := len(out.Data) / 4
			o.Data = make([]interface{}, n)
			for i := 0; i < n; i++ {
				val := int32(binary.LittleEndian.Uint32(out.Data[i*4 : i*4+4]))
				o.Data[i] = val
			}
		}

		result = append(result, o)
	}
	return result
}

func dtypeToInternal(dt string) enginev1.DataType {
	switch dt {
	case "FP32":
		return enginev1.DataType_TYPE_FP32
	case "FP64":
		return enginev1.DataType_TYPE_FP64
	case "INT32":
		return enginev1.DataType_TYPE_INT32
	case "INT64":
		return enginev1.DataType_TYPE_INT64
	case "INT8":
		return enginev1.DataType_TYPE_INT8
	case "UINT8":
		return enginev1.DataType_TYPE_UINT8
	case "BOOL":
		return enginev1.DataType_TYPE_BOOL
	case "BYTES":
		return enginev1.DataType_TYPE_STRING
	default:
		return enginev1.DataType_TYPE_INVALID
	}
}

func internalDtypeToString(dt enginev1.DataType) string {
	switch dt {
	case enginev1.DataType_TYPE_FP32:
		return "FP32"
	case enginev1.DataType_TYPE_FP64:
		return "FP64"
	case enginev1.DataType_TYPE_INT32:
		return "INT32"
	case enginev1.DataType_TYPE_INT64:
		return "INT64"
	case enginev1.DataType_TYPE_INT8:
		return "INT8"
	case enginev1.DataType_TYPE_UINT8:
		return "UINT8"
	case enginev1.DataType_TYPE_BOOL:
		return "BOOL"
	case enginev1.DataType_TYPE_STRING:
		return "BYTES"
	default:
		return "INVALID"
	}
}
