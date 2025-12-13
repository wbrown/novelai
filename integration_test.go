//go:build integration

package novelai

import (
	"os"
	"strings"
	"testing"
)

// Integration tests that hit the real NovelAI API.
// Run with: NAI_API_KEY=... go test -v -tags=integration

func skipIfNoToken(t *testing.T) {
	if os.Getenv("NAI_API_KEY") == "" {
		t.Skip("NAI_API_KEY not set, skipping integration test")
	}
}

func skipIfServerError(t *testing.T, err error) {
	if err != nil && strings.Contains(err.Error(), "status 500") {
		t.Skipf("NovelAI API returned server error (may be temporary): %v", err)
	}
}

func TestRealAPI_Send(t *testing.T) {
	skipIfNoToken(t)

	conv := NewConversation("You are a helpful assistant. Be very brief.")
	conv.Settings.MaxTokens = 50

	reply, stopReason, inToks, outToks, err := conv.Send("Say hello in exactly 3 words.")
	skipIfServerError(t, err)
	if err != nil {
		t.Fatalf("Send failed: %v", err)
	}

	t.Logf("Reply: %q", reply)
	t.Logf("Stop reason: %s", stopReason)
	t.Logf("Tokens: %d in, %d out", inToks, outToks)

	if reply == "" {
		t.Error("Expected non-empty reply")
	}

	if stopReason != "end_turn" && stopReason != "max_tokens" {
		t.Errorf("Unexpected stop reason: %s", stopReason)
	}

	if inToks == 0 {
		t.Error("Expected non-zero input tokens")
	}

	if outToks == 0 {
		t.Error("Expected non-zero output tokens")
	}

	// Verify conversation history
	if len(conv.Messages) != 2 {
		t.Errorf("Expected 2 messages, got %d", len(conv.Messages))
	}
}

func TestRealAPI_MultiTurn(t *testing.T) {
	skipIfNoToken(t)

	conv := NewConversation("You are a helpful assistant. Be very brief.")
	conv.Settings.MaxTokens = 50
	conv.Settings.Thinking = false // Disable thinking for cleaner output

	// First turn
	reply1, _, _, _, err := conv.Send("My name is Alice.")
	skipIfServerError(t, err)
	if err != nil {
		t.Fatalf("First send failed: %v", err)
	}
	t.Logf("Turn 1: %q", reply1)

	// Second turn - should remember context
	reply2, _, _, _, err := conv.Send("What is my name?")
	if err != nil {
		t.Fatalf("Second send failed: %v", err)
	}
	t.Logf("Turn 2: %q", reply2)

	if !strings.Contains(strings.ToLower(reply2), "alice") {
		t.Errorf("Expected reply to contain 'Alice', got: %q", reply2)
	}

	// Verify 4 messages in history
	if len(conv.Messages) != 4 {
		t.Errorf("Expected 4 messages, got %d", len(conv.Messages))
	}
}

func TestRealAPI_Streaming(t *testing.T) {
	skipIfNoToken(t)

	conv := NewConversation("You are a helpful assistant. Be very brief.")
	conv.Settings.MaxTokens = 50

	var chunks []string
	callback := func(text string, done bool) {
		if text != "" {
			chunks = append(chunks, text)
		}
		if done {
			t.Log("Stream complete")
		}
	}

	reply, stopReason, _, _, err := conv.SendStreaming("Count from 1 to 5.", callback)
	skipIfServerError(t, err)
	if err != nil {
		t.Fatalf("SendStreaming failed: %v", err)
	}

	t.Logf("Full reply: %q", reply)
	t.Logf("Received %d chunks", len(chunks))
	t.Logf("Stop reason: %s", stopReason)

	if reply == "" {
		t.Error("Expected non-empty reply")
	}

	if len(chunks) == 0 {
		t.Error("Expected to receive streaming chunks")
	}

	// Verify chunks concatenate to full reply
	concatenated := strings.Join(chunks, "")
	if concatenated != reply {
		t.Errorf("Chunks don't match reply:\nChunks: %q\nReply: %q", concatenated, reply)
	}
}

func TestRealAPI_ThinkingEnabled(t *testing.T) {
	skipIfNoToken(t)

	conv := NewConversation("You are a helpful assistant.")
	conv.Settings.MaxTokens = 100
	conv.Settings.Thinking = true // Explicitly enabled (default)

	reply, _, _, _, err := conv.Send("What is 2+2?")
	skipIfServerError(t, err)
	if err != nil {
		t.Fatalf("Send failed: %v", err)
	}

	t.Logf("With thinking: %q", reply)

	// With thinking enabled, reply should contain <think> block
	if !strings.Contains(reply, "<think>") {
		t.Error("Expected <think> block when thinking is enabled")
	}
}

func TestRealAPI_ThinkingDisabled(t *testing.T) {
	skipIfNoToken(t)

	conv := NewConversation("You are a helpful assistant.")
	conv.Settings.MaxTokens = 100
	conv.Settings.Thinking = false // Disable thinking

	reply, _, _, _, err := conv.Send("What is 2+2?")
	skipIfServerError(t, err)
	if err != nil {
		t.Fatalf("Send failed: %v", err)
	}

	t.Logf("Without thinking: %q", reply)

	// With thinking disabled, reply should NOT contain <think> block
	if strings.Contains(reply, "<think>") {
		t.Error("Expected no <think> block when thinking is disabled")
	}
}

func TestRealAPI_StreamingNoThinking(t *testing.T) {
	skipIfNoToken(t)

	conv := NewConversation("You are a helpful assistant.")
	conv.Settings.MaxTokens = 50
	conv.Settings.Thinking = false // Disable thinking

	var chunks []string
	callback := func(text string, done bool) {
		if text != "" {
			chunks = append(chunks, text)
		}
	}

	reply, _, _, _, err := conv.SendStreaming("What is 3+3?", callback)
	skipIfServerError(t, err)
	if err != nil {
		t.Fatalf("SendStreaming failed: %v", err)
	}

	t.Logf("Streaming without thinking: %q", reply)
	t.Logf("Received %d chunks", len(chunks))

	// With thinking disabled, reply should NOT contain <think> block
	if strings.Contains(reply, "<think>") {
		t.Error("Expected no <think> block when thinking is disabled in streaming")
	}
}
