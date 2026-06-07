fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .build_server(true)
        .build_client(false)
        .compile_protos(
            &[
                "../proto/inference/engine/v1/inference_internal.proto",
                "../proto/inference/kfs/kserve_grpc.proto",
            ],
            &["../proto"],
        )?;
    Ok(())
}
