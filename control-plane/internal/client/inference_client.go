package client

import (
	"context"
	"fmt"

	enginev1 "github.com/mustafacavusoglu/axon/control-plane/inference/engine/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type InferenceClient struct {
	conn   *grpc.ClientConn
	client enginev1.InferenceEngineClient
	socket string
}

func NewInferenceClient(socketPath string) (*InferenceClient, error) {
	conn, err := grpc.Dial(
		"unix://"+socketPath,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to inference engine: %w", err)
	}

	client := enginev1.NewInferenceEngineClient(conn)
	return &InferenceClient{
		conn:   conn,
		client: client,
		socket: socketPath,
	}, nil
}

func (c *InferenceClient) Close() error {
	return c.conn.Close()
}

func (c *InferenceClient) BatchInfer(
	ctx context.Context,
	modelName string,
	version uint32,
	inputs []*enginev1.InferInput,
	batchSize uint32,
) (*enginev1.BatchInferResponse, error) {
	return c.client.BatchInfer(ctx, &enginev1.BatchInferRequest{
		ModelName: modelName,
		Version:   version,
		Inputs:    inputs,
		BatchSize: batchSize,
	})
}

func (c *InferenceClient) LoadModel(ctx context.Context, name string, version uint32, modelPath string, concurrency uint32) error {
	resp, err := c.client.LoadModel(ctx, &enginev1.LoadModelRequest{
		Name:        name,
		Version:     version,
		ModelPath:   modelPath,
		Concurrency: concurrency,
	})
	if err != nil {
		return err
	}
	if !resp.Success {
		return fmt.Errorf("engine refused: %s", resp.Error)
	}
	return nil
}

func (c *InferenceClient) UnloadModel(ctx context.Context, name string, version uint32) error {
	_, err := c.client.UnloadModel(ctx, &enginev1.UnloadModelRequest{
		Name:    name,
		Version: version,
	})
	return err
}

func (c *InferenceClient) ModelStatus(ctx context.Context, name string, version uint32) (*enginev1.ModelStatusResponse, error) {
	return c.client.ModelStatus(ctx, &enginev1.ModelStatusRequest{
		Name:    name,
		Version: version,
	})
}

func (c *InferenceClient) Healthcheck(ctx context.Context) (*enginev1.HealthResponse, error) {
	return c.client.Healthcheck(ctx, &enginev1.HealthRequest{})
}
