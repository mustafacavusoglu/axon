package health

import (
	"context"
	"log"
	"time"

	"github.com/mustafacavusoglu/axon/control-plane/internal/client"
	"github.com/mustafacavusoglu/axon/control-plane/internal/manager"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

type Checker struct {
	registry *manager.ModelRegistry
	client   *client.InferenceClient
}

func NewChecker(registry *manager.ModelRegistry, client *client.InferenceClient) *Checker {
	return &Checker{
		registry: registry,
		client:   client,
	}
}

func (c *Checker) IsLive() bool {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	resp, err := c.client.Healthcheck(ctx)
	if err != nil {
		log.Printf("healthcheck failed: %v", err)
		return false
	}
	log.Printf("healthcheck OK, uptime=%ds", resp.UptimeSec)
	return true
}

func (c *Checker) IsReady() bool {
	if !c.IsLive() {
		return false
	}

	entries := c.registry.List()
	hasLoaded := false
	hasReady := false
	for _, e := range entries {
		hasLoaded = true
		if e.State == manager.StateReady {
			hasReady = true
		} else if e.State == manager.StateError {
			return false
		}
	}

	if !hasLoaded {
		return true
	}
	return hasReady
}

func (c *Checker) ModelReady(name string, version string) error {
	entries := c.registry.List()
	for _, e := range entries {
		if e.Config != nil && e.Config.Name == name {
			if e.State == manager.StateReady {
				return nil
			}
			return status.Errorf(codes.Unavailable, "model %s not ready", name)
		}
	}
	return status.Errorf(codes.NotFound, "model %s not found", name)
}
