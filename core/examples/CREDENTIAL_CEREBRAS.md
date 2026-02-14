## Service

**Name:** Cerebras API

**Description:** API key for Cerebras' LLM inference service - provides fast and cost-effective LLM inference, used for output validation and cleanup in the agent framework.

---

## Credential Identity

- **credential_id:** `cerebras`
- **env_var:** `CEREBRAS_API_KEY`
- **credential_key:** `api_key`

---

## Tools

Tool function names that require this credential:

- `hive.core.framework.graph.output_cleaner.OutputCleaner` (for LLM-based output cleaning)
- `hive.core.framework.llm.litellm.LiteLLMProvider` (when using Cerebras as backend)

---

## Auth Methods

- **Direct API key supported:** Yes
- **Aden OAuth supported:** No

---

## How to Get the Credential

Link: https://cloud.cerebras.ai/

Step-by-step instructions:

1. Go to https://cloud.cerebras.ai/
2. Sign up for an account or log in
3. Navigate to API Keys or Settings
4. Create a new API key
5. Copy the API key

---

## Health Check

A lightweight API call to validate the credential:

- **Endpoint:** `https://api.cerebras.ai/v1/models`
- **Method:** `GET`
- **Auth header:** `Authorization: Bearer {CEREBRAS_API_KEY}`
- **Parameters:** None
- **200 means:** Key is valid
- **401 means:** Invalid or expired key

---

## Credential Group

- [ ] No, single credential
- [x] Yes — group with:
  - `anthropic` (used together for output cleaning/validation)

---

## Additional Context

- **Free tier:** Check https://cloud.cerebras.ai/ for current offerings
- **Rate limits:** Varies by plan
- **Models available:** Llama 3.3 70B, and other Cerebras-hosted models
- **Use case:** Primarily used for fast, low-cost LLM inference and output validation in the agent framework
