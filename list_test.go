package anthropic

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/aceobservability/ace/backend/pkg/llm"
)

func TestListModels_StaticAllowList(t *testing.T) {
	p, err := New(llm.LLMConfig{DisplayName: "Anthropic"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	models, err := p.ListModels(context.Background())
	if err != nil {
		t.Fatalf("ListModels: %v", err)
	}
	if len(models) == 0 {
		t.Fatal("expected static allow-list, got none")
	}
	seen := map[string]bool{}
	for _, m := range models {
		if m.ID == "" {
			t.Fatalf("model missing id: %+v", m)
		}
		if m.Vendor != "anthropic" {
			t.Errorf("expected vendor anthropic for %s, got %q", m.ID, m.Vendor)
		}
		if seen[m.ID] {
			t.Errorf("duplicate model id %q", m.ID)
		}
		seen[m.ID] = true
	}
	for _, id := range []string{"claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"} {
		if !seen[id] {
			t.Errorf("allow-list missing %s", id)
		}
	}
}

func TestRegister_ListModelsAndChatViaRegistry(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"content":[{"type":"text","text":"from-registry"}]}`))
	}))
	defer server.Close()

	p, err := llm.New("anthropic", llm.LLMConfig{
		BaseURL:     server.URL,
		APIKey:      "sk-ant",
		DisplayName: "Anthropic",
	})
	if err != nil {
		t.Fatalf("llm.New anthropic: %v", err)
	}
	got, ok := p.(*Provider)
	if !ok {
		t.Fatalf("expected *Provider, got %T", p)
	}
	if got.APIKey != "sk-ant" {
		t.Errorf("expected plaintext APIKey, got %q", got.APIKey)
	}

	models, err := p.ListModels(context.Background())
	if err != nil {
		t.Fatalf("ListModels: %v", err)
	}
	if len(models) == 0 {
		t.Fatal("expected allow-list from registered factory")
	}

	rr := httptest.NewRecorder()
	if err := p.Chat(context.Background(), llm.ChatRequest{
		Model:    "claude-sonnet-5",
		Messages: []json.RawMessage{json.RawMessage(`{"role":"user","content":"hi"}`)},
	}, rr); err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if !strings.Contains(rr.Body.String(), "from-registry") {
		t.Errorf("expected mapped content, got %s", rr.Body.String())
	}
}
