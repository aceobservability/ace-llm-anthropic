# ace-llm-anthropic

Compile-time Anthropic Messages adapter for [Ace](https://github.com/aceobservability/ace).

Implements `github.com/aceobservability/ace/backend/pkg/llm` (`AIProvider`). `init` calls `llm.RegisterLLM("anthropic", New)`. Ace blank-imports this module.

```go
import _ "github.com/aceobservability/ace-llm-anthropic"
```

List-models is a static Claude allow-list. Chat talks to `/v1/messages` and writes OpenAI-shaped JSON or SSE. Tests use `httptest` fixtures. No live Anthropic API in CI.
