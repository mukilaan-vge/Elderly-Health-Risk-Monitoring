# Elderly Health Risk Monitoring — Frontend

A lightweight, same-origin dashboard for the existing FastAPI prediction service.

## Access

When served by the FastAPI application, open:

`http://localhost:8000/dashboard/`

## Integration

The UI calls the existing endpoints:

- `GET /health`
- `POST /predict`

No frontend prediction values are hardcoded.
