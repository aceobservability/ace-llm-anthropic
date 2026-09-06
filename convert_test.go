package anthropic

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/aceobservability/ace/backend/pkg/llm"
)

func TestChat_MergesSystemAndConsecutiveRoles(t *testing.T) {
	var gotBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &gotBody)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"content":[{"type":"text","text":"ok"}]}`))
	}))
	defer server.Close()

	p, err := New(llm.LLMConfig{BaseURL: server.URL, APIKey: "k"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	err = p.Chat(context.Background(), llm.ChatRequest{
		Model: "claude-sonnet-5",
		Messages: []json.RawMessage{
			json.RawMessage(`{"role":"system","content":"A"}`),
			json.RawMessage(`{"role":"system","content":"B"}`),
			json.RawMessage(`{"role":"user","content":"one"}`),
			json.RawMessage(`{"role":"user","content":"two"}`),
			json.RawMessage(`{"role":"assistant","content":"prev"}`),
			json.RawMessage(`{"role":"tool","content":"ignored"}`),
			json.RawMessage(`{"role":"user","content":"three"}`),
		},
	}, httptest.NewRecorder())
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if gotBody["system"] != "A\n\nB" {
		t.Errorf("system concat: %#v", gotBody["system"])
	}
	msgs, _ := gotBody["messages"].([]any)
	if len(msgs) != 3 {
		t.Fatalf("expected 3 anthropic messages (merged + skip tool), got %#v", gotBody["messages"])
	}
	first, _ := msgs[0].(map[string]any)
	if first["role"] != "user" || first["content"] != "one\n\ntwo" {
		t.Errorf("first message: %#v", first)
	}
}

func TestChat_ToolsFailClosed(t *testing.T) {
	called := false
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
		t.Error("upstream must not be called when tools are present")
		http.Error(w, "unreachable", http.StatusInternalServerError)
	}))
	defer server.Close()

	p, err := New(llm.LLMConfig{BaseURL: server.URL, APIKey: "k"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	err = p.Chat(context.Background(), llm.ChatRequest{
		Model:    "claude-sonnet-5",
		Messages: []json.RawMessage{json.RawMessage(`{"role":"user","content":"hi"}`)},
		Tools:    []json.RawMessage{json.RawMessage(`{"type":"function","function":{"name":"query_prometheus"}}`)},
	}, httptest.NewRecorder())
	if err == nil {
		t.Fatal("expected tools to fail closed")
	}
	lower := strings.ToLower(err.Error())
	if !strings.Contains(lower, "400") || !strings.Contains(lower, "tool") {
		t.Fatalf("error must look like a 400 tools rejection, got %v", err)
	}
	if called {
		t.Fatal("must not call Anthropic when tools are present")
	}
}

func TestChat_OmitsEmptyAssistantAndToolTurns(t *testing.T) {
	var gotBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		if err := json.Unmarshal(raw, &gotBody); err != nil {
			t.Errorf("upstream body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"content":[{"type":"text","text":"ok"}]}`))
	}))
	defer server.Close()

	p, err := New(llm.LLMConfig{BaseURL: server.URL, APIKey: "k"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	err = p.Chat(context.Background(), llm.ChatRequest{
		Model: "claude-sonnet-5",
		Messages: []json.RawMessage{
			json.RawMessage(`{"role":"user","content":"hi"}`),
			json.RawMessage(`{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"query_prometheus","arguments":"{}"}}]}`),
			json.RawMessage(`{"role":"tool","tool_call_id":"call_1","content":"metric=1"}`),
			json.RawMessage(`{"role":"assistant","content":""}`),
			json.RawMessage(`{"role":"user","content":"what next"}`),
		},
	}, httptest.NewRecorder())
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}

	msgs, _ := gotBody["messages"].([]any)
	if len(msgs) != 1 {
		t.Fatalf("expected one merged user message, got %#v", gotBody["messages"])
	}
	first, _ := msgs[0].(map[string]any)
	if first["role"] != "user" {
		t.Errorf("first role: %#v", first["role"])
	}
	if first["content"] != "hi\n\nwhat next" {
		t.Errorf("content: %#v", first["content"])
	}
	for _, raw := range msgs {
		m, _ := raw.(map[string]any)
		content, _ := m["content"].(string)
		if content == "" {
			t.Errorf("empty content block: %#v", m)
		}
	}
}

func TestOpenAIToAnthropicMessages_DoesNotMergeOntoSkippedEmpty(t *testing.T) {
	_, messages, err := openaiToAnthropicMessages([]json.RawMessage{
		json.RawMessage(`{"role":"user","content":"one"}`),
		json.RawMessage(`{"role":"assistant","content":""}`),
		json.RawMessage(`{"role":"assistant","content":"two"}`),
		json.RawMessage(`{"role":"user","content":""}`),
		json.RawMessage(`{"role":"user","content":"three"}`),
	})
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	if len(messages) != 3 {
		t.Fatalf("expected user/assistant/user, got %#v", messages)
	}
	if messages[0] != (anthropicMessage{Role: "user", Content: "one"}) {
		t.Errorf("first: %#v", messages[0])
	}
	if messages[1] != (anthropicMessage{Role: "assistant", Content: "two"}) {
		t.Errorf("assistant must not start with skipped empty blob, got %#v", messages[1])
	}
	if messages[2] != (anthropicMessage{Role: "user", Content: "three"}) {
		t.Errorf("user must not start with skipped empty blob, got %#v", messages[2])
	}
}

func TestChat_FirstMessageMustBeUser(t *testing.T) {
	called := false
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
	}))
	defer server.Close()

	p, err := New(llm.LLMConfig{BaseURL: server.URL, APIKey: "k"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	err = p.Chat(context.Background(), llm.ChatRequest{
		Model:    "claude-sonnet-5",
		Messages: []json.RawMessage{json.RawMessage(`{"role":"assistant","content":"I started"}`)},
	}, httptest.NewRecorder())
	if err == nil {
		t.Fatal("expected first-message-must-be-user error")
	}
	if !strings.Contains(err.Error(), "first message") {
		t.Errorf("expected clear first-message error, got %v", err)
	}
	if called {
		t.Error("must not call Anthropic when the first message is not user")
	}
}

func TestChat_UpstreamError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"error":{"message":"invalid x-api-key"}}`))
	}))
	defer server.Close()

	p, err := New(llm.LLMConfig{BaseURL: server.URL, APIKey: "bad"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	err = p.Chat(context.Background(), llm.ChatRequest{
		Model:    "claude-sonnet-5",
		Messages: []json.RawMessage{json.RawMessage(`{"role":"user","content":"hi"}`)},
		Stream:   true,
	}, httptest.NewRecorder())
	if err == nil {
		t.Fatal("expected upstream error")
	}
	if !strings.Contains(err.Error(), "401") {
		t.Errorf("expected 401 in error, got %v", err)
	}
}
