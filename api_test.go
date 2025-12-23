package novelai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/wbrown/llmapi"
)

// mockCompletionResponse creates a mock OpenAI-compatible completions response
func mockCompletionResponse(text, finishReason string, promptTokens, completionTokens int) completionResponse {
	return completionResponse{
		ID:      "cmpl-123",
		Object:  "text_completion",
		Created: 1677652288,
		Model:   "glm-4-6",
		Choices: []struct {
			Index        int    `json:"index"`
			Text         string `json:"text"`
			FinishReason string `json:"finish_reason"`
		}{
			{
				Index:        0,
				Text:         text,
				FinishReason: finishReason,
			},
		},
		Usage: struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
			TotalTokens      int `json:"total_tokens"`
		}{
			PromptTokens:     promptTokens,
			CompletionTokens: completionTokens,
			TotalTokens:      promptTokens + completionTokens,
		},
	}
}

func TestNewConversation(t *testing.T) {
	system := "You are a helpful assistant."
	conv := NewConversation(system)

	if conv.System != system {
		t.Errorf("Expected system prompt %q, got %q", system, conv.System)
	}

	if len(conv.Messages) != 0 {
		t.Errorf("Expected empty messages, got %d", len(conv.Messages))
	}

	if conv.Settings.Model != DefaultSettings.Model {
		t.Errorf("Expected model %q, got %q", DefaultSettings.Model, conv.Settings.Model)
	}

	if conv.HttpClient == nil {
		t.Error("Expected HttpClient to be initialized")
	}
}

func TestSend(t *testing.T) {
	// Create mock server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify request
		if r.Method != "POST" {
			t.Errorf("Expected POST, got %s", r.Method)
		}

		if r.Header.Get("Content-Type") != "application/json" {
			t.Errorf("Expected Content-Type application/json, got %s", r.Header.Get("Content-Type"))
		}

		if r.Header.Get("Authorization") != "Bearer test-token" {
			t.Errorf("Expected Authorization header with test-token")
		}

		// Parse request body
		var req completionRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("Failed to decode request: %v", err)
		}

		// Verify request structure - should have a prompt string
		if req.Prompt == "" {
			t.Errorf("Expected non-empty prompt")
		}

		// Return mock response
		resp := mockCompletionResponse(" Hello! How can I help you?", "stop", 10, 8)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	// Create conversation pointing to mock server
	conv := NewConversation("You are helpful.")
	conv.ApiToken = "test-token"

	// Use a custom transport to redirect to our test server
	conv.HttpClient = &http.Client{
		Transport: &redirectTransport{server.URL},
	}

	reply, stopReason, inToks, outToks, err := conv.Send("Hello!", llmapi.Sampling{})
	if err != nil {
		t.Fatalf("Send failed: %v", err)
	}

	if reply != " Hello! How can I help you?" {
		t.Errorf("Unexpected reply: %q", reply)
	}

	if stopReason != "end_turn" {
		t.Errorf("Expected stop reason 'end_turn', got %q", stopReason)
	}

	if inToks != 10 {
		t.Errorf("Expected 10 input tokens, got %d", inToks)
	}

	if outToks != 8 {
		t.Errorf("Expected 8 output tokens, got %d", outToks)
	}

	// Verify message was added to history
	if len(conv.Messages) != 2 {
		t.Errorf("Expected 2 messages in history, got %d", len(conv.Messages))
	}

	if conv.Messages[0].Role != "user" {
		t.Errorf("Expected first message to be user, got %s", conv.Messages[0].Role)
	}

	if conv.Messages[1].Role != "assistant" {
		t.Errorf("Expected second message to be assistant, got %s", conv.Messages[1].Role)
	}
}

func TestNormalizeStopReason(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"stop", "end_turn"},
		{"length", "max_tokens"},
		{"tool_calls", "tool_use"},
		{"unknown", "unknown"},
		{"", ""},
	}

	for _, tc := range tests {
		result := normalizeStopReason(tc.input)
		if result != tc.expected {
			t.Errorf("normalizeStopReason(%q) = %q, expected %q", tc.input, result, tc.expected)
		}
	}
}

func TestAddMessage(t *testing.T) {
	conv := NewConversation("System")

	conv.AddMessage(llmapi.RoleUser, "Hello")
	conv.AddMessage(llmapi.RoleAssistant, "Hi there!")

	if len(conv.Messages) != 2 {
		t.Errorf("Expected 2 messages, got %d", len(conv.Messages))
	}

	if conv.Messages[0].Content != "Hello" {
		t.Errorf("Expected first message 'Hello', got %q", conv.Messages[0].Content)
	}

	if conv.Messages[1].Content != "Hi there!" {
		t.Errorf("Expected second message 'Hi there!', got %q", conv.Messages[1].Content)
	}
}

func TestGetters(t *testing.T) {
	conv := NewConversation("Test system")
	conv.AddMessage(llmapi.RoleUser, "Test")
	conv.Usage.InputTokens = 100
	conv.Usage.OutputTokens = 50

	if conv.GetSystem() != "Test system" {
		t.Errorf("GetSystem() = %q, expected 'Test system'", conv.GetSystem())
	}

	msgs := conv.GetMessages()
	if len(msgs) != 1 {
		t.Errorf("GetMessages() returned %d messages, expected 1", len(msgs))
	}

	usage := conv.GetUsage()
	if usage.InputTokens != 100 || usage.OutputTokens != 50 {
		t.Errorf("GetUsage() = %+v, expected {100, 50}", usage)
	}
}

func TestClear(t *testing.T) {
	conv := NewConversation("System")
	conv.AddMessage(llmapi.RoleUser, "Hello")
	conv.AddMessage(llmapi.RoleAssistant, "Hi")
	conv.Usage.InputTokens = 100
	conv.Usage.OutputTokens = 50

	conv.Clear()

	if len(conv.Messages) != 0 {
		t.Errorf("Expected empty messages after Clear, got %d", len(conv.Messages))
	}

	if conv.Usage.InputTokens != 0 || conv.Usage.OutputTokens != 0 {
		t.Errorf("Expected zero usage after Clear, got %+v", conv.Usage)
	}

	// System should be preserved
	if conv.System != "System" {
		t.Errorf("Expected system prompt to be preserved, got %q", conv.System)
	}
}

func TestMergeIfLastTwoAssistant(t *testing.T) {
	conv := NewConversation("System")

	// Add user message then two assistant messages
	conv.AddMessage(llmapi.RoleUser, "Hello")
	conv.AddMessage(llmapi.RoleAssistant, "First part")
	conv.AddMessage(llmapi.RoleAssistant, "second part")

	conv.MergeIfLastTwoAssistant()

	if len(conv.Messages) != 2 {
		t.Errorf("Expected 2 messages after merge, got %d", len(conv.Messages))
	}

	expected := "First partsecond part"
	if conv.Messages[1].Content != expected {
		t.Errorf("Expected merged content %q, got %q", expected, conv.Messages[1].Content)
	}
}

func TestMergeIfLastTwoAssistant_NoMerge(t *testing.T) {
	conv := NewConversation("System")

	// User, assistant, user - should not merge
	conv.AddMessage(llmapi.RoleUser, "Hello")
	conv.AddMessage(llmapi.RoleAssistant, "Hi")
	conv.AddMessage(llmapi.RoleUser, "How are you?")

	conv.MergeIfLastTwoAssistant()

	if len(conv.Messages) != 3 {
		t.Errorf("Expected 3 messages (no merge), got %d", len(conv.Messages))
	}
}

func TestSetModel(t *testing.T) {
	conv := NewConversation("System")

	conv.SetModel("llama-3-erato-v1")

	if conv.Settings.Model != "llama-3-erato-v1" {
		t.Errorf("Expected model 'llama-3-erato-v1', got %q", conv.Settings.Model)
	}
}

func TestSendWithoutToken(t *testing.T) {
	conv := NewConversation("System")
	conv.ApiToken = ""

	_, _, _, _, err := conv.Send("Hello", llmapi.Sampling{})
	if err == nil {
		t.Error("Expected error when API token is not set")
	}
}

func TestSendContinueWithoutAssistant(t *testing.T) {
	conv := NewConversation("System")
	conv.ApiToken = "test"
	conv.AddMessage(llmapi.RoleUser, "Hello")

	_, _, _, _, err := conv.Send("", llmapi.Sampling{})
	if err == nil {
		t.Error("Expected error when continuing without last assistant message")
	}
}

// redirectTransport redirects all requests to a test server
type redirectTransport struct {
	targetURL string
}

func (t *redirectTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	// Create new request to test server, preserving context
	newReq, err := http.NewRequestWithContext(req.Context(), req.Method, t.targetURL, req.Body)
	if err != nil {
		return nil, err
	}
	newReq.Header = req.Header
	return http.DefaultTransport.RoundTrip(newReq)
}

// TestContextCancellation tests that requests respect context cancellation.
func TestContextCancellation(t *testing.T) {
	// Create a server that delays its response
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(2 * time.Second)
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	// Create a context that will be cancelled quickly
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	conv := NewConversation("Test system prompt")
	conv.SetContext(ctx)
	conv.ApiToken = "test-token"
	conv.HttpClient = &http.Client{
		Transport: &redirectTransport{targetURL: server.URL},
	}

	_, _, _, _, err := conv.Send("Hello", llmapi.Sampling{})

	if err == nil {
		t.Fatal("Expected error due to context cancellation, got nil")
	}

	if !strings.Contains(err.Error(), "context deadline exceeded") &&
		!strings.Contains(err.Error(), "context canceled") {
		t.Errorf("Expected context cancellation error, got: %v", err)
	}
}

// TestContextCancellationImmediate tests immediate context cancellation.
func TestContextCancellationImmediate(t *testing.T) {
	// Create a server (won't actually be reached)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	// Create an already-cancelled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	conv := NewConversation("Test system prompt")
	conv.SetContext(ctx)
	conv.ApiToken = "test-token"
	conv.HttpClient = &http.Client{
		Transport: &redirectTransport{targetURL: server.URL},
	}

	_, _, _, _, err := conv.Send("Hello", llmapi.Sampling{})

	if err == nil {
		t.Fatal("Expected error due to context cancellation, got nil")
	}

	if !strings.Contains(err.Error(), "context canceled") {
		t.Errorf("Expected context canceled error, got: %v", err)
	}
}

// TestContextCancellationStreaming tests that streaming requests respect context cancellation.
func TestContextCancellationStreaming(t *testing.T) {
	// Create a server that delays its response
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(2 * time.Second)
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	// Create a context that will be cancelled quickly
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	conv := NewConversation("Test system prompt")
	conv.SetContext(ctx)
	conv.ApiToken = "test-token"
	conv.HttpClient = &http.Client{
		Transport: &redirectTransport{targetURL: server.URL},
	}

	_, _, _, _, err := conv.SendStreaming("Hello", llmapi.Sampling{}, nil)

	if err == nil {
		t.Fatal("Expected error due to context cancellation, got nil")
	}

	if !strings.Contains(err.Error(), "context deadline exceeded") &&
		!strings.Contains(err.Error(), "context canceled") {
		t.Errorf("Expected context cancellation error, got: %v", err)
	}
}

// =============================================================================
// Real-world context cancellation tests (no mock server)
// =============================================================================

// TestContextCancellationPreCancelled tests that an already-cancelled context
// results in a context cancellation error.
func TestContextCancellationPreCancelled(t *testing.T) {
	// Create an already-cancelled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	conv := NewConversation("Test")
	conv.SetContext(ctx)
	conv.ApiToken = "test-token" // Doesn't need to be valid

	_, _, _, _, err := conv.Send("Hello", llmapi.Sampling{})

	if err == nil {
		t.Fatal("Expected error due to pre-cancelled context")
	}

	// The error should indicate context cancellation
	if !strings.Contains(err.Error(), "context canceled") {
		t.Errorf("Expected context canceled error, got: %v", err)
	}
}

// TestContextCancellationTinyTimeout tests cancellation with a very short timeout
// against the real API endpoint. This tests the actual HTTP client behavior.
func TestContextCancellationTinyTimeout(t *testing.T) {
	// 1ms timeout - will fail during TCP/TLS handshake
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Millisecond)
	defer cancel()

	conv := NewConversation("Test")
	conv.SetContext(ctx)
	conv.ApiToken = "test-token" // Doesn't need to be valid

	_, _, _, _, err := conv.Send("Hello", llmapi.Sampling{})

	if err == nil {
		t.Fatal("Expected error due to timeout")
	}

	// The error should indicate context deadline exceeded
	if !strings.Contains(err.Error(), "context deadline exceeded") &&
		!strings.Contains(err.Error(), "context canceled") {
		t.Errorf("Expected context error, got: %v", err)
	}
}

// TestEndpointOverride tests the endpoint override functionality.
func TestEndpointOverride(t *testing.T) {
	conv := NewConversation("System")

	// Default should use DefaultCompletionsURL
	if conv.endpoint() != DefaultCompletionsURL {
		t.Errorf("Expected default endpoint %q, got %q", DefaultCompletionsURL, conv.endpoint())
	}

	// Set custom endpoint
	customEndpoint := "https://custom.api.example.com/v1/completions"
	conv.SetEndpoint(customEndpoint)

	if conv.Endpoint != customEndpoint {
		t.Errorf("Expected Endpoint field to be %q, got %q", customEndpoint, conv.Endpoint)
	}

	if conv.endpoint() != customEndpoint {
		t.Errorf("Expected endpoint() to return %q, got %q", customEndpoint, conv.endpoint())
	}

	// Clear endpoint should revert to default
	conv.SetEndpoint("")
	if conv.endpoint() != DefaultCompletionsURL {
		t.Errorf("Expected endpoint to revert to default, got %q", conv.endpoint())
	}
}

// TestEndpointOverrideWithMockServer tests that endpoint override works with actual HTTP requests.
func TestEndpointOverrideWithMockServer(t *testing.T) {
	// Create mock server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Return mock response
		resp := mockCompletionResponse("Hello from custom endpoint!", "stop", 10, 5)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	conv := NewConversation("System")
	conv.ApiToken = "test-token"
	conv.SetEndpoint(server.URL)

	reply, _, _, _, err := conv.Send("Hello", llmapi.Sampling{})
	if err != nil {
		t.Fatalf("Send failed: %v", err)
	}

	if reply != "Hello from custom endpoint!" {
		t.Errorf("Expected reply from custom endpoint, got %q", reply)
	}
}

// TestContextCancellationMidStream tests cancelling a real streaming request
// mid-generation. This is an integration test that requires valid API credentials.
func TestContextCancellationMidStream(t *testing.T) {
	if DefaultApiToken == "" {
		t.Skip("Skipping integration test: NAI_API_KEY not set")
	}

	ctx, cancel := context.WithCancel(context.Background())

	conv := NewConversation("You are a helpful assistant.")
	conv.SetContext(ctx)

	var tokensReceived int
	var cancelled bool

	// Cancel after receiving some tokens
	callback := func(text string, done bool) {
		if done {
			return
		}
		tokensReceived++
		// Cancel after receiving a few tokens
		if tokensReceived >= 5 && !cancelled {
			cancelled = true
			cancel()
		}
	}

	// Ask for a long response
	_, _, _, _, err := conv.SendStreaming(
		"Write a detailed 500 word essay about the history of computing.",
		llmapi.Sampling{},
		callback,
	)

	// Should get an error due to cancellation
	if err == nil {
		t.Log("Request completed without error (generation finished before cancellation)")
		return
	}

	// Rate limiting is not a test failure
	if strings.Contains(err.Error(), "429") || strings.Contains(err.Error(), "Concurrent generation") {
		t.Skip("Skipping: API rate limited")
	}

	if !strings.Contains(err.Error(), "context canceled") {
		t.Errorf("Expected context canceled error, got: %v", err)
	}

	t.Logf("Successfully cancelled after receiving %d tokens", tokensReceived)
}

// TestThinkFormatTypes tests the predefined ThinkFormat types.
func TestThinkFormatTypes(t *testing.T) {
	// Test GLM-4.6 format
	if ThinkFormatGLM46.UserSuffix != "/nothink" {
		t.Errorf("Expected GLM46 UserSuffix '/nothink', got %q", ThinkFormatGLM46.UserSuffix)
	}
	if ThinkFormatGLM46.AssistantPrefix != "<think></think>\n" {
		t.Errorf("Expected GLM46 AssistantPrefix '<think></think>\\n', got %q", ThinkFormatGLM46.AssistantPrefix)
	}

	// Test GLM-4.7 format
	if ThinkFormatGLM47.UserSuffix != "/nothink" {
		t.Errorf("Expected GLM47 UserSuffix '/nothink', got %q", ThinkFormatGLM47.UserSuffix)
	}
	if ThinkFormatGLM47.AssistantPrefix != "</think>" {
		t.Errorf("Expected GLM47 AssistantPrefix '</think>', got %q", ThinkFormatGLM47.AssistantPrefix)
	}

	// Test None format
	if ThinkFormatNone.UserSuffix != "" {
		t.Errorf("Expected None UserSuffix '', got %q", ThinkFormatNone.UserSuffix)
	}
	if ThinkFormatNone.AssistantPrefix != "" {
		t.Errorf("Expected None AssistantPrefix '', got %q", ThinkFormatNone.AssistantPrefix)
	}
}

// TestSetThinkFormat tests setting custom think formats.
func TestSetThinkFormat(t *testing.T) {
	conv := NewConversation("System")

	// Default should be GLM46
	tf := conv.thinkFormat()
	if tf.AssistantPrefix != ThinkFormatGLM46.AssistantPrefix {
		t.Errorf("Expected default to be GLM46 format")
	}

	// Set to GLM47
	conv.SetThinkFormat(&ThinkFormatGLM47)
	tf = conv.thinkFormat()
	if tf.AssistantPrefix != "</think>" {
		t.Errorf("Expected GLM47 AssistantPrefix '</think>', got %q", tf.AssistantPrefix)
	}

	// Set to None
	conv.SetThinkFormat(&ThinkFormatNone)
	tf = conv.thinkFormat()
	if tf.AssistantPrefix != "" {
		t.Errorf("Expected None AssistantPrefix '', got %q", tf.AssistantPrefix)
	}

	// Set to custom
	customFormat := &ThinkFormat{
		UserSuffix:      "/custom",
		AssistantPrefix: "<custom/>",
	}
	conv.SetThinkFormat(customFormat)
	tf = conv.thinkFormat()
	if tf.UserSuffix != "/custom" || tf.AssistantPrefix != "<custom/>" {
		t.Errorf("Expected custom format, got %+v", tf)
	}
}

// TestBuildPromptWithThinkFormats tests buildPrompt with different think formats.
func TestBuildPromptWithThinkFormats(t *testing.T) {
	tests := []struct {
		name            string
		format          *ThinkFormat
		thinking        bool
		expectSuffix    string
		expectPrefix    string
		notExpectSuffix string
		notExpectPrefix string
	}{
		{
			name:         "GLM46 thinking disabled",
			format:       &ThinkFormatGLM46,
			thinking:     false,
			expectSuffix: "/nothink",
			expectPrefix: "<think></think>\n",
		},
		{
			name:            "GLM46 thinking enabled",
			format:          &ThinkFormatGLM46,
			thinking:        true,
			notExpectSuffix: "/nothink",
			notExpectPrefix: "<think></think>",
		},
		{
			name:         "GLM47 thinking disabled",
			format:       &ThinkFormatGLM47,
			thinking:     false,
			expectSuffix: "/nothink",
			expectPrefix: "</think>",
		},
		{
			name:            "GLM47 thinking enabled",
			format:          &ThinkFormatGLM47,
			thinking:        true,
			notExpectSuffix: "/nothink",
			notExpectPrefix: "</think>",
		},
		{
			name:            "None format thinking disabled",
			format:          &ThinkFormatNone,
			thinking:        false,
			notExpectSuffix: "/nothink",
			notExpectPrefix: "</think>",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			conv := NewConversation("System prompt")
			conv.Settings.Thinking = tc.thinking
			conv.SetThinkFormat(tc.format)
			conv.AddMessage(llmapi.RoleUser, "Hello")

			prompt := conv.buildPrompt()

			if tc.expectSuffix != "" && !strings.Contains(prompt, tc.expectSuffix) {
				t.Errorf("Expected prompt to contain %q, got:\n%s", tc.expectSuffix, prompt)
			}
			if tc.expectPrefix != "" && !strings.Contains(prompt, tc.expectPrefix) {
				t.Errorf("Expected prompt to contain %q, got:\n%s", tc.expectPrefix, prompt)
			}
			if tc.notExpectSuffix != "" && strings.Contains(prompt, tc.notExpectSuffix) {
				t.Errorf("Expected prompt NOT to contain %q, got:\n%s", tc.notExpectSuffix, prompt)
			}
			if tc.notExpectPrefix != "" && strings.Contains(prompt, tc.notExpectPrefix) {
				t.Errorf("Expected prompt NOT to contain %q, got:\n%s", tc.notExpectPrefix, prompt)
			}
		})
	}
}

// TestDefaultSettingsThinkFormat tests that DefaultSettings includes ThinkFormat.
func TestDefaultSettingsThinkFormat(t *testing.T) {
	if DefaultSettings.ThinkFormat == nil {
		t.Error("Expected DefaultSettings.ThinkFormat to not be nil")
	}
	if DefaultSettings.ThinkFormat != &ThinkFormatGLM46 {
		t.Error("Expected DefaultSettings.ThinkFormat to be ThinkFormatGLM46")
	}
}

