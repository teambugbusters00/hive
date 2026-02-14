# Contributing Issues for Hive Agent Framework

This document lists potential issues and features you can contribute to the Hive Agent Framework. Issues are categorized by difficulty and area.

---

## 📋 Table of Contents

1. [Good First Issues](#-good-first-issues)
2. [Phase 1: Foundation Issues](#-phase-1-foundation-issues)
3. [Phase 2: Expansion Issues](#-phase-2-expansion-issues)
4. [Tool Integration Issues](#-tool-integration-issues)
5. [Documentation Issues](#-documentation-issues)

---

## 🌟 Good First Issues

### 1. Default Monitoring Hooks
**Priority:** Medium | **Difficulty:** Easy | **Category:** Core

**Description:**
Add default monitoring hooks to the Node SDK for observability.

**Location:** `core/framework/graph/node.py`

**Task:**
- Add default hooks for:
  - Execution start/end events
  - Error tracking
  - Performance metrics collection

**Acceptance Criteria:**
- [ ] Node execution emits start/end events
- [ ] Errors are logged with context
- [ ] Performance metrics are collected

---

### 2. Foundation Unit Tests for Credentials
**Priority:** Medium | **Difficulty:** Easy | **Category:** Testing

**Description:**
Add unit tests for credential source abstractions.

**Location:** `core/framework/credentials/`

**Task:**
- Add unit tests for `CredentialSource` base class
- Test `EnvVarSource` implementation
- Test priority chain mechanism

**Acceptance Criteria:**
- [ ] Unit tests cover all credential source methods
- [ ] Edge cases are tested
- [ ] All tests pass

---

### 3. Documentation for Credential Sources
**Priority:** Low | **Difficulty:** Easy | **Category:** Documentation

**Description:**
Write comprehensive documentation for credential sources.

**Location:** `docs/credential-store-usage.md`

**Task:**
- Document all credential source providers
- Add example configurations
- Include troubleshooting section

**Acceptance Criteria:**
- [ ] All providers documented
- [ ] Working examples for each provider
- [ ] Troubleshooting guide included

---

### 4. CLI Tools for Memory Management
**Priority:** Medium | **Difficulty:** Easy | **Category:** Developer Experience

**Description:**
Add CLI commands for memory management.

**Location:** `core/framework/runner/cli.py`

**Task:**
- Add command to list memory sessions
- Add command to clear memory
- Add command to export memory

**Acceptance Criteria:**
- [ ] `python -m framework memory list` works
- [ ] `python -m framework memory clear` works
- [ ] `python -m framework memory export` works

---

### 5. CLI Tools for Credential Management
**Priority:** Medium | **Difficulty:** Easy | **Category:** Developer Experience

**Description:**
Add CLI commands for credential management.

**Location:** `core/framework/runner/cli.py`

**Task:**
- Add command to list credentials
- Add command to validate credentials
- Add command to add new credentials

**Acceptance Criteria:**
- [ ] `python -m framework creds list` works
- [ ] `python -m framework creds validate` works
- [ ] `python -m framework creds add` works

---

## 🚀 Phase 1: Foundation Issues

### 6. Use Template Agent as Start
**Priority:** High | **Difficulty:** Medium | **Category:** Builder

**Description:**
Allow Worker Agent Creation to use template agents as starting points.

**Location:** `core/framework/builder/`

**Related:** ROADMAP.md line 117

**Task:**
- Implement template agent loading mechanism
- Add template selection UI
- Support template customization

**Acceptance Criteria:**
- [ ] Templates can be loaded from file
- [ ] Templates can be customized
- [ ] New agents inherit template structure

---

### 7. Streaming Interface for Real-time Monitoring
**Priority:** High | **Difficulty:** Medium | **Category:** Runtime

**Description:**
Implement streaming interface for real-time agent monitoring.

**Location:** `core/framework/runtime/`

**Related:** ROADMAP.md line 124

**Task:**
- Add WebSocket streaming for execution events
- Implement real-time metrics broadcasting
- Add client-side event handling

**Acceptance Criteria:**
- [ ] WebSocket endpoint for streaming
- [ ] Events are broadcast in real-time
- [ ] Client can subscribe to specific event types

---

### 8. AWS Secrets Manager Integration
**Priority:** Medium | **Difficulty:** Medium | **Category:** Credentials

**Description:**
Add AWS Secrets Manager as a credential source.

**Location:** `core/framework/credentials/`

**Related:** ROADMAP.md line 138

**Task:**
- Create `AWSSecretsSource` class
- Implement AWS SDK integration
- Add unit tests

**Acceptance Criteria:**
- [ ] `AWSSecretsSource` class implemented
- [ ] Can fetch credentials from AWS Secrets Manager
- [ ] Unit tests included

---

### 9. Azure Key Vault Integration
**Priority:** Medium | **Difficulty:** Medium | **Category:** Credentials

**Description:**
Add Azure Key Vault as a credential source.

**Location:** `core/framework/credentials/`

**Related:** ROADMAP.md line 139

**Task:**
- Create `AzureKeyVaultSource` class
- Implement Azure SDK integration
- Add unit tests

**Acceptance Criteria:**
- [ ] `AzureKeyVaultSource` class implemented
- [ ] Can fetch credentials from Azure Key Vault
- [ ] Unit tests included

---

### 10. Audit Logging for Credentials
**Priority:** Medium | **Difficulty:** Medium | **Category:** Credentials

**Description:**
Add audit logging for credential access and modifications.

**Location:** `core/framework/credentials/`

**Related:** ROADMAP.md line 143

**Task:**
- Implement audit log entry creation
- Add log storage for audit records
- Create audit report generation

**Acceptance Criteria:**
- [ ] All credential access is logged
- [ ] Log entries include user, action, timestamp
- [ ] Audit reports can be generated

---

### 11. Per-Environment Configuration Support
**Priority:** Medium | **Difficulty:** Medium | **Category:** Credentials

**Description:**
Add support for per-environment credential configuration.

**Location:** `core/framework/credentials/`

**Related:** ROADMAP.md line 144

**Task:**
- Add environment-aware credential resolution
- Support dev/staging/prod environments
- Add environment variable overrides

**Acceptance Criteria:**
- [ ] Credentials can be configured per environment
- [ ] Environment can be set via configuration
- [ ] Fallback to default environment works

---

### 12. Excel Tools
**Priority:** Medium | **Difficulty:** Medium | **Category:** Tools

**Description:**
Add Excel file processing tools.

**Location:** `tools/src/aden_tools/tools/`

**Related:** ROADMAP.md line 176

**Task:**
- Create Excel read tool
- Create Excel write tool
- Add support for formatting

**Acceptance Criteria:**
- [ ] Can read Excel files
- [ ] Can write Excel files
- [ ] Basic formatting supported

---

### 13. Email Tools
**Priority:** Medium | **Difficulty:** Medium | **Category:** Tools

**Description:**
Add email sending and receiving tools.

**Location:** `tools/src/aden_tools/tools/`

**Related:** ROADMAP.md line 177

**Task:**
- Create email send tool
- Create email read tool
- Add SMTP/IMAP support

**Acceptance Criteria:**
- [ ] Can send emails
- [ ] Can read emails
- [ ] Attachment support included

---

### 14. Failure Recording Mechanism
**Priority:** High | **Difficulty:** Medium | **Category:** Eval System

**Description:**
Implement failure recording mechanism for the evaluation system.

**Location:** `core/framework/testing/`

**Related:** ROADMAP.md line 186

**Task:**
- Create failure log schema
- Implement failure recording during execution
- Add failure query API

**Acceptance Criteria:**
- [ ] Failures are recorded with context
- [ ] Failures can be queried by agent/node/date
- [ ] Failure trends can be analyzed

---

### 15. SDK for Defining Failure Conditions
**Priority:** High | **Difficulty:** Medium | **Category:** Eval System

**Description:**
Create SDK for defining custom failure conditions.

**Location:** `core/framework/testing/`

**Related:** ROADMAP.md line 187

**Task:**
- Create failure condition DSL
- Implement condition evaluation
- Add SDK examples

**Acceptance Criteria:**
- [ ] DSL supports common failure patterns
- [ ] Conditions can be defined in agent config
- [ ] Examples provided

---

### 16. Basic Observability Hooks
**Priority:** Medium | **Difficulty:** Medium | **Category:** Eval System

**Description:**
Add basic observability hooks to the framework.

**Location:** `core/framework/runtime/`

**Related:** ROADMAP.md line 188

**Task:**
- Add execution tracing
- Implement metric collection
- Create observability dashboard API

**Acceptance Criteria:**
- [ ] Full execution trace available
- [ ] Metrics collected per node
- [ ] Dashboard API returns observability data

---

### 17. User-Driven Log Analysis
**Priority:** Low | **Difficulty:** Medium | **Category:** Eval System

**Description:**
Implement user-driven log analysis tools (OSS approach).

**Location:** `core/framework/testing/`

**Related:** ROADMAP.md line 189

**Task:**
- Create log search API
- Implement log filtering
- Add log export functionality

**Acceptance Criteria:**
- [ ] Logs can be searched by query
- [ ] Logs can be filtered by date/node
- [ ] Logs can be exported

---

### 18. Debugging Mode
**Priority:** High | **Difficulty:** Medium | **Category:** Developer Experience

**Description:**
Implement debugging mode for agent development.

**Location:** `core/framework/runner/`

**Related:** ROADMAP.md line 196

**Task:**
- Add step-through execution
- Implement breakpoint support
- Add variable inspection

**Acceptance Criteria:**
- [ ] Agents can be run in debug mode
- [ ] Breakpoints can be set
- [ ] Variables can be inspected at breakpoints

---

## 🚀 Phase 2: Expansion Issues

### 19. Basic Monitoring from Agent Node SDK
**Priority:** High | **Difficulty:** Medium | **Category:** Guardrails

**Description:**
Add basic monitoring support to Agent Node SDK.

**Location:** `core/framework/graph/node.py`

**Related:** ROADMAP.md line 222

**Task:**
- Add monitoring hooks to node execution
- Implement metric collection
- Create monitoring API

**Acceptance Criteria:**
- [ ] Node execution emits metrics
- [ ] Monitoring API returns current metrics
- [ ] Example usage documented

---

### 20. SDK Guardrail Implementation
**Priority:** High | **Difficulty:** Medium | **Category:** Guardrails

**Description:**
Implement SDK-level guardrails within nodes.

**Location:** `core/framework/graph/node.py`

**Related:** ROADMAP.md line 223

**Task:**
- Create guardrail base class
- Implement common guardrail types
- Add guardrail configuration to nodes

**Acceptance Criteria:**
- [ ] Guardrails can be configured per node
- [ ] Common guardrails implemented (rate limit, timeout, etc.)
- [ ] Examples provided

---

### 21. Guardrail Type Support
**Priority:** Medium | **Difficulty:** Medium | **Category:** Guardrails

**Description:**
Support determined conditions as guardrails.

**Location:** `core/framework/graph/node.py`

**Related:** ROADMAP.md line 224

**Task:**
- Add condition evaluation for guardrails
- Support threshold-based guardrails
- Implement guardrail actions

**Acceptance Criteria:**
- [ ] Conditions can trigger guardrails
- [ ] Threshold-based guardrails work
- [ ] Guardrail actions are customizable

---

### 22. Streaming Mode Support
**Priority:** High | **Difficulty:** Hard | **Category:** Agent Capability

**Description:**
Implement streaming mode for agent execution.

**Location:** `core/framework/runtime/`

**Related:** ROADMAP.md line 227

**Task:**
- Implement streaming LLM responses
- Add streaming event handling
- Create streaming API

**Acceptance Criteria:**
- [ ] LLM responses stream in real-time
- [ ] Events are emitted during streaming
- [ ] Streaming can be enabled/disabled

---

### 23. Image Generation Support
**Priority:** Medium | **Difficulty:** Medium | **Category:** Agent Capability

**Description:**
Add image generation capability to agents.

**Location:** `core/framework/llm/`

**Related:** ROADMAP.md line 228

**Task:**
- Add image generation LLM provider support
- Create image output node type
- Add image handling utilities

**Acceptance Criteria:**
- [ ] Image generation via DALL-E/Stable Diffusion
- [ ] Images can be used as node outputs
- [ ] Examples provided

---

### 24. Image and Flatfile Input Understanding
**Priority:** Medium | **Difficulty:** Medium | **Category:** Agent Capability

**Description:**
Add capability to understand images and flatfiles as user input.

**Location:** `core/framework/runtime/`

**Related:** ROADMAP.md line 229

**Task:**
- Add image input processing
- Add flatfile (PDF, CSV, Excel) input processing
- Create unified input handler

**Acceptance Criteria:**
- [ ] Images can be uploaded as input
- [ ] Flatfiles can be uploaded as input
- [ ] Input is parsed correctly

---

### 25. Event Bus for Nodes
**Priority:** High | **Difficulty:** Hard | **Category:** Event System

**Description:**
Implement event bus system for node communication.

**Location:** `core/framework/runtime/event_bus.py`

**Related:** ROADMAP.md line 232

**Task:**
- Create event bus implementation
- Implement event publishing/subscribing
- Add event filtering

**Acceptance Criteria:**
- [ ] Events can be published
- [ ] Nodes can subscribe to events
- [ ] Event filtering works

---

### 26. Message Model & Session Management
**Priority:** High | **Difficulty:** Hard | **Category:** Memory System

**Description:**
Implement message model with structured content types and session classes.

**Location:** `core/framework/storage/`

**Related:** ROADMAP.md line 235-237

**Task:**
- Create `Message` class with content types
- Implement `Session` classes
- Add session persistence

**Acceptance Criteria:**
- [ ] Messages support text, image, file content
- [ ] Sessions manage conversation state
- [ ] Sessions persist across runs

---

### 27. Storage Migration
**Priority:** High | **Difficulty:** Medium | **Category:** Memory System

**Description:**
Implement granular per-message file persistence.

**Location:** `core/framework/storage/`

**Related:** ROADMAP.md line 239-240

**Task:**
- Implement `/message/[agentID]/...` storage
- Migrate from monolithic run storage
- Add migration utilities

**Acceptance Criteria:**
- [ ] Messages stored individually
- [ ] Migration script works
- [ ] Backward compatibility maintained

---

### 28. Context Building & Conversation Loop
**Priority:** High | **Difficulty:** Medium | **Category:** Memory System

**Description:**
Implement context building and conversation loop functionality.

**Location:** `core/framework/llm/`

**Related:** ROADMAP.md line 241-244

**Task:**
- Implement `Message.stream(sessionID)`
- Update `LLMNode.execute()` for full context
- Implement `Message.toModelMessages()`

**Acceptance Criteria:**
- [ ] Context builds from session history
- [ ] LLM calls include full context
- [ ] Conversion to model format works

---

### 29. Proactive Compaction
**Priority:** Medium | **Difficulty:** Medium | **Category:** Memory System

**Description:**
Implement proactive overflow detection and pruning.

**Location:** `core/framework/storage/`

**Related:** ROADMAP.md line 245-247

**Task:**
- Implement overflow detection
- Develop backward-scanning pruning strategy
- Integrate with context building

**Acceptance Criteria:**
- [ ] Overflow detected before limits reached
- [ ] Old tool outputs can be pruned
- [ ] Important context preserved

---

### 30. Enhanced Token Tracking
**Priority:** Medium | **Difficulty:** Medium | **Category:** Memory System

**Description:**
Extend LLM response to track reasoning and cache tokens.

**Location:** `core/framework/llm/`

**Related:** ROADMAP.md line 248-250

**Task:**
- Extend `LLMResponse` for token tracking
- Add reasoning token tracking
- Integrate with compaction logic

**Acceptance Criteria:**
- [ ] All token types tracked
- [ ] Token metrics available
- [ ] Compaction uses token data

---

## 🔧 Tool Integration Issues

### 31. Twitter (X) Integration
**Priority:** Medium | **Difficulty:** Medium | **Category:** Tools

**Description:**
Add Twitter/X social media integration.

**Location:** `tools/src/aden_tools/tools/`

**Related:** ROADMAP.md line 151

**Task:**
- Create Twitter API tools
- Implement tweet posting/reading
- Add OAuth support

**Acceptance Criteria:**
- [ ] Can post tweets
- [ ] Can read tweets/timelines
- [ ] OAuth authentication works

---

### 32. Instagram Integration
**Priority:** Medium | **Difficulty:** Medium | **Category:** Tools

**Description:**
Add Instagram social media integration.

**Location:** `tools/src/aden_tools/tools/`

**Related:** ROADMAP.md line 153

**Task:**
- Create Instagram API tools
- Implement post creation/reading
- Add OAuth support

**Acceptance Criteria:**
- [ ] Can create posts
- [ ] Can read posts
- [ ] OAuth authentication works

---

### 33. Hubspot Integration
**Priority:** Medium | **Difficulty:** Medium | **Category:** Tools

**Description:**
Add Hubspot SaaS integration.

**Location:** `tools/src/aden_tools/tools/`

**Related:** ROADMAP.md line 155

**Task:**
- Create Hubspot API tools
- Implement CRM operations
- Add OAuth support

**Acceptance Criteria:**
- [ ] Can create/update contacts
- [ ] Can manage deals
- [ ] OAuth authentication works

---

### 34. Slack Integration
**Priority:** Medium | **Difficulty:** Medium | **Category:** Tools

**Description:**
Add Slack SaaS integration.

**Location:** `tools/src/aden_tools/tools/`

**Related:** ROADMAP.md line 156

**Task:**
- Create Slack API tools
- Implement message sending/channels
- Add OAuth/Bot token support

**Acceptance Criteria:**
- [ ] Can send messages
- [ ] Can read channels
- [ ] Bot authentication works

---

### 35. Teams Integration
**Priority:** Medium | **Difficulty:** Medium | **Category:** Tools

**Description:**
Add Microsoft Teams SaaS integration.

**Location:** `tools/src/aden_tools/tools/`

**Related:** ROADMAP.md line 157

**Task:**
- Create Teams API tools
- Implement message operations
- Add OAuth support

**Acceptance Criteria:**
- [ ] Can send messages
- [ ] Can read channels
- [ ] OAuth authentication works

---

### 36. Zoom Integration
**Priority:** Low | **Difficulty:** Medium | **Category:** Tools

**Description:**
Add Zoom SaaS integration.

**Location:** `tools/src/aden_tools/tools/`

**Related:** ROADMAP.md line 158

**Task:**
- Create Zoom API tools
- Implement meeting operations
- Add OAuth support

**Acceptance Criteria:**
- [ ] Can create meetings
- [ ] Can manage meetings
- [ ] OAuth authentication works

---

### 37. Stripe Integration
**Priority:** Low | **Difficulty:** Medium | **Category:** Tools

**Description:**
Add Stripe SaaS integration.

**Location:** `tools/src/aden_tools/tools/`

**Related:** ROADMAP.md line 159

**Task:**
- Create Stripe API tools
- Implement payment operations
- Add API key support

**Acceptance Criteria:**
- [ ] Can process payments
- [ ] Can manage customers
- [ ] API key authentication works

---

### 38. Salesforce Integration
**Priority:** Low | **Difficulty:** Hard | **Category:** Tools

**Description:**
Add Salesforce SaaS integration.

**Location:** `tools/src/aden_tools/tools/`

**Related:** ROADMAP.md line 160

**Task:**
- Create Salesforce API tools
- Implement CRM operations
- Add OAuth support

**Acceptance Criteria:**
- [ ] Can manage contacts/opportunities
- [ ] Can query data
- [ ] OAuth authentication works

---

## 📚 Documentation Issues

### 39. Introduction Video
**Priority:** Low | **Difficulty:** Medium | **Category:** Documentation

**Description:**
Create introduction video for the framework.

**Location:** `docs/` or external

**Related:** ROADMAP.md line 206

**Task:**
- Create video (5-10 minutes)
- Cover core concepts
- Include demo

**Acceptance Criteria:**
- [ ] Video created and hosted
- [ ] Linked from README
- [ ] Transcripts included

---

### 40. Tool Usage Documentation
**Priority:** Medium | **Difficulty:** Easy | **Category:** Documentation

**Description:**
Add comprehensive documentation for all tools.

**Location:** `docs/tools/`

**Related:** ROADMAP.md line 287

**Task:**
- Document all tools
- Add usage examples
- Include troubleshooting

**Acceptance Criteria:**
- [ ] All tools documented
- [ ] Examples work
- [ ] Troubleshooting included

---

## 🎯 Additional Opportunities

### Claude Code Integration
**Priority:** Medium | **Difficulty:** Hard | **Category:** Coding Agent Support

**Related:** ROADMAP.md line 253

---

### Cursor Integration
**Priority:** Medium | **Difficulty:** Hard | **Category:** Coding Agent Support

**Related:** ROADMAP.md line 254

---

### Opencode Integration
**Priority:** Low | **Difficulty:** Hard | **Category:** Coding Agent Support

**Related:** ROADMAP.md line 255

---

### Semantic Search Integration
**Priority:** Medium | **Difficulty:** Medium | **Category:** File System

**Related:** ROADMAP.md line 259

---

### Custom Tool Integrator
**Priority:** High | **Difficulty:** Medium | **Category:** Tools

**Related:** ROADMAP.md line 263

---

### Node Discovery Tool
**Priority:** Medium | **Difficulty:** Medium | **Category:** Core Agent Tools

**Related:** ROADMAP.md line 266

---

### HITL Tool Enhancement
**Priority:** Medium | **Difficulty:** Medium | **Category:** Core Agent Tools

**Related:** ROADMAP.md line 267

---

### Wake-up Tool
**Priority:** Low | **Difficulty:** Medium | **Category:** Core Agent Tools

**Related:** ROADMAP.md line 268

---

## 📝 How to Contribute

1. **Browse Issues**: Look for issues labeled `good first issue` for beginners
2. **Comment**: Comment on the issue to claim it
3. **Fork**: Fork the repository
4. **Branch**: Create a feature branch (`feature/issue-description`)
5. **Implement**: Make your changes
6. **Test**: Add tests and ensure existing tests pass
7. **Document**: Update documentation if needed
8. **PR**: Create a Pull Request

---

## 🔗 Links

- [GitHub Issues](https://github.com/adenhq/hive/issues)
- [Good First Issues](https://github.com/adenhq/hive/issues?q=label:%22good%20first%20issue%22)
- [Discord Community](https://discord.com/invite/MXE49hrKDk)
flowchart TB
    Input["📥 Input\n(JSON/CLI/MCP)"] --> Runner["🏃 Runner"]
    Runner --> Orch["🎯 Orchestrator"]
    Orch --> Runtime["⚡ Runtime"]
    Runtime --> Graph["🔄 Graph Executor"]
    Graph -->|Executes| Nodes["📦 Nodes"]
    Nodes -->|Uses| LLM["🧠 LLM"]
    Nodes -->|Uses| Tools["🔧 Tools"]
    Nodes -->|Uses| Storage["💾 Storage"]
    Nodes -->|Uses| Creds["🔐 Credentials"]
