## Service

**Name:** FastAPI Agent Server

**Description:** REST API server that enables remote execution of Hive agent graphs - allows external systems to trigger agent workflows via HTTP requests.

---

## Credential Identity

- **credential_id:** `fastapi_agent`
- **env_var:** Not required (runs locally)
- **credential_key:** Not applicable

---

## Tools

Tool function names that require this credential:

- Not applicable - this is a server, not a tool credential

---

## Auth Methods

- **Direct API key supported:** No
- **Aden OAuth supported:** No

This service runs locally and doesn't require external authentication. In production, you would add authentication middleware (e.g., API key, JWT) as needed.

---

## How to Get the Credential

No credential required. To run the FastAPI server:

```bash
cd hive/core
set PYTHONIOENCODING=utf-8
set PYTHONPATH=core
python -X utf8 -m uvicorn examples.fastapi_agent:app --host 0.0.0.0 --port 8000
```

---

## Health Check

A lightweight API call to validate the service:

- **Endpoint:** `http://localhost:8000/health`
- **Method:** `GET`
- **Auth header:** None
- **Parameters:** None
- **200 means:** Server is running and healthy
- **503 means:** Server is not healthy

---

## Credential Group

- [x] No, single credential

---

## Additional Context

- **API Docs:** http://localhost:8000/docs (Swagger UI)
- **Base URL:** http://localhost:8000
- **Endpoints:**
  - `GET /` - API info
  - `GET /health` - Health check
  - `GET /goals` - List goals
  - `POST /goals` - Create goal
  - `GET /goals/{goal_id}` - Get goal
  - `GET /graphs` - List graphs
  - `POST /graphs` - Create graph
  - `GET /graphs/{graph_id}` - Get graph
  - `GET /functions` - List functions
  - `POST /execute` - Execute agent
  - `GET /history` - Execution history

**Example Execution:**
```bash
curl -X POST http://localhost:8000/execute \
  -H "Content-Type: application/json" \
  -d '{"input_data": {"name": "Alice"}}'
```

**Response:**
```json
{
  "success": true,
  "output": {
    "name": "Alice",
    "greeting": "Hello, Alice!",
    "final_greeting": "HELLO, ALICE!"
  },
  "steps_executed": 2,
  "path": ["greeter", "uppercaser"]
}
```
