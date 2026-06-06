# Sample Inference Requests

## Health Check
```bash
curl -s http://localhost:8080/v2/health/live | jq
# {"live":true}
```

## List Models
```bash
curl -s http://localhost:8080/v2/models | jq
# {"models":["cb_credit_risk","lgbm_credit_risk","skl_gradient_boosting_diabetes","skl_random_forest_breast_cancer","xgb_california_housing"]}
```

## Model Metadata
```bash
curl -s http://localhost:8080/v2/models/skl_gradient_boosting_diabetes | jq
```

---

## Single-Input Inference (regression)
```bash
curl -s -X POST http://localhost:8080/v2/models/skl_random_forest_breast_cancer/infer \
  -H 'Content-Type: application/json' \
  -d '{
    "inputs":[{
      "name":"input",
      "shape":[30],
      "datatype":"FP32",
      "data":[0.5,0.2,0.1,0.8,0.3,0.6,0.9,0.4,0.7,0.2,
             0.1,0.5,0.3,0.8,0.6,0.9,0.2,0.4,0.7,0.1,
             0.5,0.3,0.8,0.6,0.2,0.9,0.4,0.7,0.1,0.3]
    }]
  }' | jq
```

---

## Multi-Input Inference (diabetes — 10 features)
```bash
curl -s -X POST http://localhost:8080/v2/models/skl_gradient_boosting_diabetes/infer \
  -H 'Content-Type: application/json' \
  -d '{
    "inputs":[
      {"name":"age","shape":[1],"datatype":"FP32","data":[25.0]},
      {"name":"sex","shape":[1],"datatype":"FP32","data":[1.0]},
      {"name":"bmi","shape":[1],"datatype":"FP32","data":[22.5]},
      {"name":"blood_pressure","shape":[1],"datatype":"FP32","data":[85.0]},
      {"name":"total_cholesterol","shape":[1],"datatype":"FP32","data":[180.0]},
      {"name":"ldl","shape":[1],"datatype":"FP32","data":[100.0]},
      {"name":"hdl","shape":[1],"datatype":"FP32","data":[50.0]},
      {"name":"tch_ldl_ratio","shape":[1],"datatype":"FP32","data":[3.5]},
      {"name":"ltg","shape":[1],"datatype":"FP32","data":[4.0]},
      {"name":"glucose","shape":[1],"datatype":"FP32","data":[95.0]}
    ]
  }' | jq
```

---

## K8s (NodePort)
```bash
# Port: kubectl get svc inference-server -o jsonpath='{.spec.ports[0].nodePort}'
curl -s -X POST http://localhost:30080/v2/models/skl_gradient_boosting_diabetes/infer \
  -H 'Content-Type: application/json' \
  -d '{"inputs":[
    {"name":"age","shape":[1],"datatype":"FP32","data":[25.0]},
    {"name":"sex","shape":[1],"datatype":"FP32","data":[1.0]},
    {"name":"bmi","shape":[1],"datatype":"FP32","data":[22.5]},
    {"name":"blood_pressure","shape":[1],"datatype":"FP32","data":[85.0]},
    {"name":"total_cholesterol","shape":[1],"datatype":"FP32","data":[180.0]},
    {"name":"ldl","shape":[1],"datatype":"FP32","data":[100.0]},
    {"name":"hdl","shape":[1],"datatype":"FP32","data":[50.0]},
    {"name":"tch_ldl_ratio","shape":[1],"datatype":"FP32","data":[3.5]},
    {"name":"ltg","shape":[1],"datatype":"FP32","data":[4.0]},
    {"name":"glucose","shape":[1],"datatype":"FP32","data":[95.0]}
  ]}' | jq
```

---

## Response Format (KServe v2 / Triton-compatible)
```json
{
  "id": "",
  "model_name": "skl_gradient_boosting_diabetes",
  "model_version": "1",
  "outputs": [
    {
      "name": "variable",
      "datatype": "FP32",
      "shape": [1, 1],
      "data": [269.69]
    }
  ]
}
```
