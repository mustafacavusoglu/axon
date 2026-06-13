fn execute(inputs) {
    let income = inputs.get("median_income").as_f64();
    let age = inputs.get("house_age").as_f64();
    let rooms = inputs.get("avg_rooms").as_f64();
    let bedrooms = inputs.get("avg_bedrooms").as_f64();
    let population = inputs.get("population").as_f64();
    let occupancy = inputs.get("avg_occupancy").as_f64();
    let lat = inputs.get("latitude").as_f64();
    let lon = inputs.get("longitude").as_f64();

    let income = clip(income, 0.0, 15.0);
    let age = clip(age, 1.0, 52.0);
    let rooms = clip(rooms, 1.0, 40.0);
    let bedrooms = clip(bedrooms, 0.3, 5.0);
    let population = clip(population, 3.0, 35000.0);
    let occupancy = clip(occupancy, 1.0, 1000.0);
    let lat = clip(lat, 32.0, 42.0);
    let lon = clip(lon, -125.0, -114.0);

    return #{
        "median_income": create_tensor_f64("median_income", [1], income),
        "house_age": create_tensor_f64("house_age", [1], age),
        "avg_rooms": create_tensor_f64("avg_rooms", [1], rooms),
        "avg_bedrooms": create_tensor_f64("avg_bedrooms", [1], bedrooms),
        "population": create_tensor_f64("population", [1], population),
        "avg_occupancy": create_tensor_f64("avg_occupancy", [1], occupancy),
        "latitude": create_tensor_f64("latitude", [1], lat),
        "longitude": create_tensor_f64("longitude", [1], lon),
    };
}
