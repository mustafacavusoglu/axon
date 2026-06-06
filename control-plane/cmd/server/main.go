package main

import (
	"context"
	"fmt"
	"log"
	"net"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gofiber/fiber/v2/middleware/adaptor"
	kfs "github.com/mustafacavusoglu/axon/control-plane/inference/kfs"
	grpcsvc "github.com/mustafacavusoglu/axon/control-plane/internal/api/grpc"
	httphandler "github.com/mustafacavusoglu/axon/control-plane/internal/api/http"
	"github.com/mustafacavusoglu/axon/control-plane/internal/client"
	"github.com/mustafacavusoglu/axon/control-plane/internal/config"
	"github.com/mustafacavusoglu/axon/control-plane/internal/health"
	"github.com/mustafacavusoglu/axon/control-plane/internal/manager"
	"github.com/mustafacavusoglu/axon/control-plane/internal/metrics"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"google.golang.org/grpc"
	"google.golang.org/grpc/reflection"
)

func main() {
	cfg, err := config.Load()
	if err != nil {
		log.Fatalf("failed to load config: %v", err)
	}

	metrics.Init()

	engineClient, err := client.NewInferenceClient(cfg.InferenceSocket)
	if err != nil {
		log.Fatalf("failed to connect to inference engine: %v", err)
	}
	defer engineClient.Close()

	registry := manager.NewModelRegistry()
	lifecycle := manager.NewLifecycleManager(registry, engineClient, cfg.ModelRepoPath)
	checker := health.NewChecker(registry, engineClient)

	httpHandler := httphandler.NewHandler(registry, lifecycle, engineClient, checker)
	app := httphandler.NewApp(httpHandler)
	app.Get("/metrics", adaptor.HTTPHandler(promhttp.Handler()))

	go func() {
		addr := fmt.Sprintf(":%d", cfg.HTTPPort)
		log.Printf("HTTP server listening on %s", addr)
		if err := app.Listen(addr); err != nil {
			log.Fatalf("HTTP server error: %v", err)
		}
	}()

	grpcServer := grpc.NewServer(
		grpc.ChainUnaryInterceptor(loggingInterceptor),
	)
	grpcSvc := grpcsvc.NewGRPCServer(registry, engineClient)
	kfs.RegisterGRPCInferenceServiceServer(grpcServer, grpcSvc)
	reflection.Register(grpcServer)

	go func() {
		addr := fmt.Sprintf(":%d", cfg.GRPCPort)
		listener, err := net.Listen("tcp", addr)
		if err != nil {
			log.Fatalf("gRPC server listen error: %v", err)
		}
		log.Printf("gRPC server listening on %s", addr)
		if err := grpcServer.Serve(listener); err != nil {
			log.Fatalf("gRPC server error: %v", err)
		}
	}()

	go func() {
		log.Print("waiting for inference engine...")
		for i := 0; i < 30; i++ {
			if checker.IsLive() {
				log.Print("engine ready, loading models...")
				break
			}
			time.Sleep(time.Second)
		}
		if err := lifecycle.LoadAllFromRepo(cfg.ModelRepoPath); err != nil {
			log.Printf("warning: failed to load models from repo: %v", err)
		}
		log.Print("model loading complete")
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGTERM, syscall.SIGINT)
	<-quit

	log.Println("shutting down...")

	ctx, cancel := context.WithTimeout(context.Background(), cfg.DrainTimeout)
	defer cancel()

	if err := app.ShutdownWithTimeout(cfg.DrainTimeout); err != nil {
		log.Printf("HTTP shutdown error: %v", err)
	}

	grpcServer.GracefulStop()

	<-ctx.Done()
	log.Println("server stopped")
}

func loggingInterceptor(
	ctx context.Context,
	req interface{},
	info *grpc.UnaryServerInfo,
	handler grpc.UnaryHandler,
) (interface{}, error) {
	start := time.Now()
	resp, err := handler(ctx, req)
	elapsed := time.Since(start)
	log.Printf("gRPC %s | %s | %v", info.FullMethod, elapsed, err)
	return resp, err
}
