package novelai

import (
	"strings"
	"testing"

	"github.com/wbrown/llmapi"
)

// TestBuildPromptReasoningEffort pins novelai's reasoning control to the unified
// ReasoningEffort: ReasoningOff injects the /nothink suffix (disabling GLM thinking
// via ThinkFormat), and any non-off level leaves it out so the model reasons.
func TestBuildPromptReasoningEffort(t *testing.T) {
	mk := func() *Conversation {
		c := NewConversation("")
		c.AddMessage(llmapi.RoleUser, "hi")
		return c
	}

	off := mk().buildPrompt(llmapi.ReasoningOff)
	if !strings.Contains(off, "/nothink") {
		t.Errorf("ReasoningOff: prompt must inject /nothink, got:\n%s", off)
	}

	on := mk().buildPrompt(llmapi.ReasoningHigh)
	if strings.Contains(on, "/nothink") {
		t.Errorf("ReasoningHigh: prompt must NOT inject /nothink, got:\n%s", on)
	}
}
