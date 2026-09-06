package anthropic

import "github.com/aceobservability/ace/backend/pkg/llm"

func init() {
	llm.RegisterLLM("anthropic", New)
}
