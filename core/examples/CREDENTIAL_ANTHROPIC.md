## Service

**Name:** Anthropic API (Claude)

**Description:** API key for Anthropic's Claude AI models - enables agents to use Claude for text generation, tool use, and reasoning capabilities.

---

## Credential Identity

- **credential_id:** `anthropic`
- **env_var:** `ANTHROPIC_API_KEY`
- **credential_key:** `api_key`

---

## Tools

Tool function names that require this credential:

- All LLM-powered nodes in the agent framework
- `hive.core.framework.llm.anthropic.AnthropicProvider`
- `hive.core.framework.graph.node.LLMNode` (for LLM operations)
- `hive.core.framework.graph.node.NodeResult.to_summary()` (for generating summaries)

---

## Auth Methods

- **Direct API key supported:** Yes
- **Aden OAuth supported:** No

---

## How to Get the Credential

Link: https://console.anthropic.com/settings/keys

Step-by-step instructions:

1. Go to https://console.anthropic.com/settings/keys
2. Click "Create Key" button
3. Enter a name for the key (e.g., "Hive Agent Framework")
4. Select appropriate permissions (default is fine for most use cases)
5. Copy the API key (starts with `sk-ant-`)
6. The key can be set as `ANTHROPIC_API_KEY` environment variable

---

## Health Check

A lightweight API call to validate the credential:

- **Endpoint:** `https://api.anthropic.com/v1/messages`
- **Method:** `POST`
- **Auth header:** `x-api-key: {ANTHROPIC_API_KEY}`
- **Parameters:**
  ```json
  {
    "model": "claude-3-haiku-20240307",
    "max_tokens": 1,
    "messages": [{"role": "user", "content": "hi"}]
  }
  ```
- **200 means:** Key is valid
- **401 means:** Invalid or expired key
- **429 means:** Rate limited but key is valid

---

## Credential Group

- [x] No, single credential

---

## Additional Context

- **Free tier:** Yes - includes limited free credits for new users
- **Rate limits:** Varies by plan; check https://docs.anthropic.com/en/docs/api-overview
- **Models available:** Claude 3.5 Haiku (fastest/cheapest), Claude 3.5 Sonnet (balanced), Claude 3.5 Opus (most capable)
- **Context window:** Up to 200K tokens for newer models

---

## Example Usage in Agent Framework

```python
# Setting the credential
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-your-key-here")

# The framework automatically uses this for:
# - LLM node execution
# - Output validation/cleaning
# - Node result summarization
```
