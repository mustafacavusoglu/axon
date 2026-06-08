fn clean(values) {
    let cleaned = [];
    for v in values {
        if v == -1.0 {
            cleaned.push(0.0);
        } else if v < 0.0 {
            cleaned.push(0.0);
        } else {
            cleaned.push(v);
        }
    }
    return cleaned;
}

fn execute(inputs) {
    let feature_names = [
        "median_income", "house_age", "avg_rooms", "avg_bedrooms",
        "population", "avg_occupancy", "latitude", "longitude"
    ];

    let cleaned_inputs = #{};

    for name in feature_names {
        let tensor = inputs.get(name);
        let vals = tensor.as_f64();
        let cleaned_vals = clean(vals);
        cleaned_inputs[name] = create_tensor_f64(name, tensor.shape, cleaned_vals);
    }

    return infer("xgb_housing", cleaned_inputs);
}
