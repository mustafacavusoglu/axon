.PHONY: proto buf-lint buf-gen build-go build-rust build run-control run-engine docker docker-compose clean help

PROTO_DIR         := proto
CONTROL_PLANE_DIR := control-plane
INFERENCE_DIR     := inference-engine
BUF               := $(shell which buf)

default: build

help:
	@echo "make proto        - Generate protobuf code (buf)"
	@echo "make buf-lint     - Lint proto files"
	@echo "make build-go     - Build Go control plane"
	@echo "make build-rust   - Build Rust inference engine (release)"
	@echo "make build        - Build both"
	@echo "make run-control  - Run Go control plane"
	@echo "make run-engine   - Run Rust inference engine"
	@echo "make docker       - Build both Docker images"
	@echo "make clean        - Remove build artifacts"
	@echo "make test-go      - Run Go tests"
	@echo "make test-rust    - Run Rust tests"
	@echo "make test         - Run all tests"

proto: buf-lint buf-gen

buf-lint:
	$(BUF) lint

buf-gen:
	$(BUF) generate

build-go:
	cd $(CONTROL_PLANE_DIR) && go build -o ../bin/control-plane ./cmd/server

build-rust:
	cd $(INFERENCE_DIR) && cargo build --release

build: build-go build-rust

run-control:
	cd $(CONTROL_PLANE_DIR) && go run ./cmd/server

run-engine:
	cd $(INFERENCE_DIR) && cargo run

docker:
	docker build -t axon/control-plane:latest -f Dockerfile.control-plane .
	docker build -t axon/inference-engine:latest -f Dockerfile.inference-engine .

docker-compose:
	docker-compose up --build

test-go:
	cd $(CONTROL_PLANE_DIR) && go test ./...

test-rust:
	cd $(INFERENCE_DIR) && cargo test

test: test-go test-rust

clean:
	rm -rf bin/
	cd $(INFERENCE_DIR) && cargo clean
