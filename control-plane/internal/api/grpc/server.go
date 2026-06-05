package grpc

import (
	"context"
	"strconv"

	kfs "github.com/mustafacavusoglu/axon/control-plane/inference/kfs"
	enginev1 "github.com/mustafacavusoglu/axon/control-plane/inference/engine/v1"
	"github.com/mustafacavusoglu/axon/control-plane/internal/client"
	"github.com/mustafacavusoglu/axon/control-plane/internal/manager"
	"github.com/mustafacavusoglu/axon/control-plane/internal/metrics"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

type GRPCServer struct {
	kfs.UnimplementedGRPCInferenceServiceServer
	registry  *manager.ModelRegistry
	client    *client.InferenceClient
}

func NewGRPCServer(registry *manager.ModelRegistry, client *client.InferenceClient) *GRPCServer {
	return &GRPCServer{
		registry: registry,
		client:   client,
	}
}

func (s *GRPCServer) ServerLive(ctx context.Context, req *kfs.ServerLiveRequest) (*kfs.ServerLiveResponse, error) {
	_, err := s.client.Healthcheck(ctx)
	return &kfs.ServerLiveResponse{Live: err == nil}, nil
}

func (s *GRPCServer) ServerReady(ctx context.Context, req *kfs.ServerReadyRequest) (*kfs.ServerReadyResponse, error) {
	_, err := s.client.Healthcheck(ctx)
	return &kfs.ServerReadyResponse{Ready: err == nil}, nil
}

func (s *GRPCServer) ModelReady(ctx context.Context, req *kfs.ModelReadyRequest) (*kfs.ModelReadyResponse, error) {
	version := 1
	if req.Version != "" {
		if v, err := strconv.Atoi(req.Version); err == nil {
			version = v
		}
	}
	entry, found := s.registry.Get(req.Name, version)
	ready := found && entry.State == manager.StateReady
	return &kfs.ModelReadyResponse{Ready: ready}, nil
}

func (s *GRPCServer) ServerMetadata(ctx context.Context, req *kfs.ServerMetadataRequest) (*kfs.ServerMetadataResponse, error) {
	return &kfs.ServerMetadataResponse{
		Name:    "axon-inference-server",
		Version: "1.0.0",
	}, nil
}

func (s *GRPCServer) ModelMetadata(ctx context.Context, req *kfs.ModelMetadataRequest) (*kfs.ModelMetadataResponse, error) {
	version := 1
	if req.Version != "" {
		if v, err := strconv.Atoi(req.Version); err == nil {
			version = v
		}
	}

	entry, found := s.registry.Get(req.Name, version)
	if !found || entry.State != manager.StateReady {
		return nil, status.Errorf(codes.NotFound, "model %s version %d not found", req.Name, version)
	}

	versionStrings := []string{}
	for _, v := range s.registry.GetAvailableVersions(req.Name) {
		versionStrings = append(versionStrings, strconv.Itoa(v))
	}

	var inputs []*kfs.TensorMetadata
	for _, inp := range entry.Config.Inputs {
		inputs = append(inputs, &kfs.TensorMetadata{
			Name:     inp.Name,
			Datatype: dtypeToString(inp.DataType),
			Shape:    inp.Dims,
		})
	}

	var outputs []*kfs.TensorMetadata
	for _, out := range entry.Config.Outputs {
		outputs = append(outputs, &kfs.TensorMetadata{
			Name:     out.Name,
			Datatype: dtypeToString(out.DataType),
			Shape:    out.Dims,
		})
	}

	return &kfs.ModelMetadataResponse{
		Name:     req.Name,
		Versions: versionStrings,
		Platform: entry.Config.Platform,
		Inputs:   inputs,
		Outputs:  outputs,
	}, nil
}

func (s *GRPCServer) ModelInfer(ctx context.Context, req *kfs.ModelInferRequest) (*kfs.ModelInferResponse, error) {
	version := 1
	if req.ModelVersion != "" {
		if v, err := strconv.Atoi(req.ModelVersion); err == nil {
			version = v
		}
	}

	entry, found := s.registry.Get(req.ModelName, version)
	if !found || entry.State != manager.StateReady {
		return nil, status.Errorf(codes.NotFound, "model %s version %d not ready", req.ModelName, version)
	}

	s.registry.Touch(req.ModelName, version)

	var internalInputs []*enginev1.InferInput
	for _, inp := range req.Inputs {
		internalInputs = append(internalInputs, &enginev1.InferInput{
			Name:  inp.Name,
			Shape: inp.Shape,
			Data:  inp.RawData,
			Dtype: strToInternalDataType(inp.Datatype),
		})
	}

	resp, err := s.client.BatchInfer(ctx, req.ModelName, uint32(version), internalInputs, 1)
	if err != nil {
		metrics.InferenceRequests.WithLabelValues(req.ModelName, strconv.Itoa(version), "error").Inc()
		return nil, status.Errorf(codes.Internal, "inference failed: %v", err)
	}

	metrics.InferenceRequests.WithLabelValues(req.ModelName, strconv.Itoa(version), "ok").Inc()

	var outputs []*kfs.InferOutput
	for _, out := range resp.Outputs {
		outputs = append(outputs, &kfs.InferOutput{
			Name:     out.Name,
			Shape:    out.Shape,
			Datatype: internalDtypeToStr(enginev1.DataType(out.Dtype)),
			RawData:  out.Data,
		})
	}

	return &kfs.ModelInferResponse{
		Id:           req.Id,
		ModelName:    req.ModelName,
		ModelVersion: strconv.Itoa(version),
		Outputs:      outputs,
	}, nil
}

func dtypeToString(dt manager.DataType) string {
	switch dt {
	case manager.DTFP32:
		return "FP32"
	case manager.DTFP64:
		return "FP64"
	case manager.DTINT32:
		return "INT32"
	case manager.DTINT64:
		return "INT64"
	case manager.DTINT8:
		return "INT8"
	case manager.DTUINT8:
		return "UINT8"
	case manager.DTBOOL:
		return "BOOL"
	case manager.DTSTRING:
		return "BYTES"
	default:
		return "INVALID"
	}
}

func strToInternalDataType(dt string) enginev1.DataType {
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

func internalDtypeToStr(dt enginev1.DataType) string {
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
