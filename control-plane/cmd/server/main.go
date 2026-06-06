package main

import (
	"context"
	"fmt"
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
	zaplog "github.com/mustafacavusoglu/axon/control-plane/internal/log"
	"github.com/mustafacavusoglu/axon/control-plane/internal/manager"
	"github.com/mustafacavusoglu/axon/control-plane/internal/metrics"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"google.golang.org/grpc"
	"google.golang.org/grpc/reflection"
)

func main() {
	cfg, err := config.Load()
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to load config: %v\n", err)
		os.Exit(1)
	}

	zaplog.Init(cfg.LogLevel)
	defer zaplog.Sync()
	log := zaplog.L

	metrics.Init()

	engineClient, err := client.NewInferenceClient(cfg.InferenceSocket)
	if err != nil {
		log.Fatalw("failed to connect to inference engine", "error", err)
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
		log.Infow("HTTP server listening", "addr", addr)
		if err := app.Listen(addr); err != nil {
			log.Fatalw("HTTP server error", "error", err)
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
			log.Fatalw("gRPC server listen error", "error", err)
		}
		log.Infow("gRPC server listening", "addr", addr)
		if err := grpcServer.Serve(listener); err != nil {
			log.Fatalw("gRPC server error", "error", err)
		}
	}()

	go func() {
		log.Info("waiting for inference engine...")
		for i := 0; i < 30; i++ {
			if checker.IsLive() {
				log.Info("engine ready, loading models...")
				break
			}
			time.Sleep(time.Second)
		}
		if err := lifecycle.LoadAllFromRepo(cfg.ModelRepoPath); err != nil {
			log.Warnw("failed to load models from repo", "error", err)
		}
		log.Info("model loading complete")
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGTERM, syscall.SIGINT)
	<-quit

	log.Info("shutting down...")

	ctx, cancel := context.WithTimeout(context.Background(), cfg.DrainTimeout)
	defer cancel()

	if err := app.ShutdownWithTimeout(cfg.DrainTimeout); err != nil {
		log.Warnw("HTTP shutdown error", "error", err)
	}

	grpcServer.GracefulStop()

	<-ctx.Done()
	log.Info("server stopped")
}

func loggingInterceptor(
	ctx context.Context,
	req interface{},
	info *grpc.UnaryServerInfo,
	handler grpc.UnaryHandler,
) (interface{}, error) {
	start := time.Now()
	resp, err := handler(ctx, req)
	zaplog.L.Infow("gRPC call",
		"method", info.FullMethod,
		"duration_ms", time.Since(start).Milliseconds(),
		"error", err,
	)
	return resp, err
}
