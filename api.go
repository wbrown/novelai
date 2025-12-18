package novelai

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/wbrown/llmapi"
)

// Compile-time interface check
var _ llmapi.Conversation = (*Conversation)(nil)

// API endpoint for NovelAI's OpenAI-compatible completions.
const completionsURL = "https://text.novelai.net/oa/v1/completions"

// DefaultApiToken is set from NAI_API_KEY environment variable during init().
// It can be overridden by setting it directly or per-conversation.
var DefaultApiToken string

// HTTP retry configuration
var (
	retries    = 3
	retryDelay = 3 * time.Second
)

// GLM special tokens for conversation structure
const (
	glmPrefix    = "[gMASK]<sop>"
	glmSystem    = "<|system|>"
	glmUser      = "<|user|>"
	glmAssistant = "<|assistant|>"
	glmNoThink   = "/nothink"
)

// Conversation manages a chat session with NovelAI.
type Conversation struct {
	// System is the system prompt for the conversation.
	System string
	// Messages is the conversation history.
	Messages []Message
	// Usage tracks cumulative token consumption.
	Usage Usage
	// ApiToken is the NovelAI API token for this conversation.
	ApiToken string
	// Settings configures generation parameters.
	Settings Settings
	// HttpClient is used for API requests.
	HttpClient *http.Client
	// Tools stores tool definitions (not used by NovelAI API, but stored for interface compliance).
	Tools []llmapi.ToolDefinition
}

// NewConversation creates a new conversation with the given system prompt.
// It initializes with DefaultSettings and DefaultApiToken.
func NewConversation(system string) *Conversation {
	return &Conversation{
		System:     system,
		Messages:   make([]Message, 0),
		ApiToken:   DefaultApiToken,
		Settings:   DefaultSettings,
		HttpClient: &http.Client{Timeout: 120 * time.Second},
	}
}

// Send sends a user message and returns the assistant's reply.
// If text is empty, continues from the last assistant message (for max_tokens continuation).
//
// Returns:
//   - reply: The assistant's response text
//   - stopReason: Normalized stop reason ("end_turn", "max_tokens", "stop_sequence")
//   - inputTokens: Tokens used for this request's input
//   - outputTokens: Tokens generated in this response
//   - err: Any error that occurred
func (c *Conversation) Send(text string, sampling llmapi.Sampling) (
	reply string,
	stopReason string,
	inputTokens int,
	outputTokens int,
	err error,
) {
	if c.ApiToken == "" {
		return "", "", 0, 0, fmt.Errorf("API token not set")
	}

	// Add user message if provided
	if text != "" {
		c.Messages = append(c.Messages, Message{Role: "user", Content: text})
	} else if len(c.Messages) == 0 {
		// Can't generate with no messages
		return "", "", 0, 0, fmt.Errorf("cannot generate: no messages in conversation")
	}
	// Note: If text is empty and last message is "user", we generate a response to it.
	// If text is empty and last message is "assistant", we continue from that message.

	// Build prompt string from system + conversation history
	prompt := c.buildPrompt()

	// Use sampling overrides if provided, otherwise use conversation defaults
	temperature := c.Settings.Temperature
	if sampling.Temperature != 0 {
		temperature = sampling.Temperature
	}
	topP := c.Settings.TopP
	if sampling.TopP != 0 {
		topP = sampling.TopP
	}
	topK := c.Settings.TopK
	if sampling.TopK != 0 {
		topK = sampling.TopK
	}

	req := completionRequest{
		Model:             c.Settings.Model,
		Prompt:            prompt,
		MaxTokens:         c.Settings.MaxTokens,
		Temperature:       temperature,
		TopP:              topP,
		TopK:              topK,
		MinP:              c.Settings.MinP,
		FrequencyPenalty:  c.Settings.FrequencyPenalty,
		PresencePenalty:   c.Settings.PresencePenalty,
		RepetitionPenalty: c.Settings.RepetitionPenalty,
		Stop:              c.Settings.StopSequences,
	}

	// Marshal request to JSON
	jsonData, err := json.Marshal(req)
	if err != nil {
		return "", "", 0, 0, fmt.Errorf("error marshaling request: %w", err)
	}

	// Create HTTP request
	httpReq, err := http.NewRequest("POST", completionsURL, bytes.NewBuffer(jsonData))
	if err != nil {
		return "", "", 0, 0, fmt.Errorf("error creating request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+c.ApiToken)

	// Perform request with retries
	var resp *http.Response
	for attempt := 0; attempt <= retries; attempt++ {
		resp, err = c.HttpClient.Do(httpReq)
		if err == nil {
			break
		}
		if attempt < retries {
			time.Sleep(retryDelay)
			// Recreate request body for retry
			httpReq, _ = http.NewRequest("POST", completionsURL, bytes.NewBuffer(jsonData))
			httpReq.Header.Set("Content-Type", "application/json")
			httpReq.Header.Set("Authorization", "Bearer "+c.ApiToken)
		}
	}
	if err != nil {
		return "", "", 0, 0, fmt.Errorf("HTTP error after %d retries: %w", retries, err)
	}
	if resp == nil {
		return "", "", 0, 0, fmt.Errorf("HTTP response is nil")
	}
	defer resp.Body.Close()

	// Read response body
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", "", 0, 0, fmt.Errorf("error reading response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return "", "", 0, 0, fmt.Errorf("API error (status %d): %s", resp.StatusCode, body)
	}

	// Parse response
	var compResp completionResponse
	if err := json.Unmarshal(body, &compResp); err != nil {
		return string(body), "", 0, 0, fmt.Errorf("error parsing response: %w", err)
	}

	if len(compResp.Choices) == 0 {
		return "", "", 0, 0, fmt.Errorf("no choices in response")
	}

	choice := compResp.Choices[0]
	reply = choice.Text

	// Add assistant message to history
	c.Messages = append(c.Messages, Message{Role: "assistant", Content: reply})

	// Normalize stop reason from OpenAI format to common format
	stopReason = normalizeStopReason(choice.FinishReason)

	// Update usage
	inputTokens = compResp.Usage.PromptTokens
	outputTokens = compResp.Usage.CompletionTokens
	c.Usage.InputTokens += inputTokens
	c.Usage.OutputTokens += outputTokens

	return reply, stopReason, inputTokens, outputTokens, nil
}

// buildPrompt constructs a prompt string from the system prompt and conversation history.
// Uses GLM-4's special token format: [gMASK]<sop><|system|>...<|user|>...<|assistant|>
// When Settings.Thinking is false, appends /nothink to disable extended thinking.
func (c *Conversation) buildPrompt() string {
	var b strings.Builder

	// Start with GLM prefix
	b.WriteString(glmPrefix)

	// System prompt
	if c.System != "" {
		b.WriteString(glmSystem)
		b.WriteString("\n")
		b.WriteString(c.System)
		b.WriteString("\n")
	}

	// Conversation history
	for i, msg := range c.Messages {
		isLastMessage := i == len(c.Messages)-1

		switch msg.Role {
		case "user":
			b.WriteString(glmUser)
			b.WriteString("\n")
			b.WriteString(msg.Content)
			// Append /nothink to last user message if thinking is disabled
			if isLastMessage && !c.Settings.Thinking {
				b.WriteString(glmNoThink)
			}
			b.WriteString("\n")
		case "assistant":
			b.WriteString(glmAssistant)
			b.WriteString("\n")
			b.WriteString(msg.Content)
			b.WriteString("\n")
		case "system":
			// Additional system messages mid-conversation
			b.WriteString(glmSystem)
			b.WriteString("\n")
			b.WriteString(msg.Content)
			b.WriteString("\n")
		}
	}

	// End with assistant token to prompt for response
	b.WriteString(glmAssistant)
	b.WriteString("\n")

	// If thinking is disabled, prefill with empty think block
	if !c.Settings.Thinking {
		b.WriteString("<think></think>\n")
	}

	return b.String()
}

// normalizeStopReason converts OpenAI stop reasons to the common format
// used by the anthropic library.
func normalizeStopReason(reason string) string {
	switch reason {
	case "stop":
		return "end_turn"
	case "length":
		return "max_tokens"
	case "tool_calls":
		return "tool_use"
	default:
		return reason
	}
}

// SendUntilDone repeatedly calls Send until stopReason != "max_tokens".
// Returns the complete accumulated output.
func (c *Conversation) SendUntilDone(text string, sampling llmapi.Sampling) (
	reply string,
	stopReason string,
	inputTokens int,
	outputTokens int,
	err error,
) {
	var totalReply string
	input := text

	for {
		var partReply string
		var inToks, outToks int

		partReply, stopReason, inToks, outToks, err = c.Send(input, sampling)
		if err != nil {
			return totalReply, stopReason, inputTokens, outputTokens, err
		}

		totalReply += partReply
		inputTokens += inToks
		outputTokens += outToks

		// Merge consecutive assistant messages
		c.MergeIfLastTwoAssistant()

		if stopReason != "max_tokens" {
			break
		}

		// Continue with empty input
		input = ""
	}

	return totalReply, stopReason, inputTokens, outputTokens, nil
}

// MergeIfLastTwoAssistant merges the last two assistant messages if they are
// both from the assistant. This is useful for combining messages that are
// split due to token limits.
func (c *Conversation) MergeIfLastTwoAssistant() {
	if len(c.Messages) < 2 {
		return
	}

	lastIdx := len(c.Messages) - 1
	secondLastIdx := lastIdx - 1

	if c.Messages[lastIdx].Role != "assistant" ||
		c.Messages[secondLastIdx].Role != "assistant" {
		return
	}

	// Merge: trim trailing whitespace from second-last, append last
	merged := strings.TrimRight(c.Messages[secondLastIdx].Content, " \t\n\r")
	merged += strings.TrimSpace(c.Messages[lastIdx].Content)

	c.Messages[secondLastIdx].Content = merged
	c.Messages = c.Messages[:lastIdx]
}

// AddMessage manually adds a message to the conversation history.
func (c *Conversation) AddMessage(role, content string) {
	c.Messages = append(c.Messages, Message{Role: role, Content: content})
}

// GetMessages returns the current conversation history.
// Converts internal Message type to llmapi.Message for interface compliance.
func (c *Conversation) GetMessages() []llmapi.Message {
	result := make([]llmapi.Message, len(c.Messages))
	for i, m := range c.Messages {
		result[i] = llmapi.Message{Role: llmapi.Role(m.Role), Content: m.Content}
	}
	return result
}

// GetUsage returns cumulative token usage for this conversation.
// Converts internal Usage type to llmapi.Usage for interface compliance.
func (c *Conversation) GetUsage() llmapi.Usage {
	return llmapi.Usage{
		InputTokens:  c.Usage.InputTokens,
		OutputTokens: c.Usage.OutputTokens,
	}
}

// GetSystem returns the system prompt.
func (c *Conversation) GetSystem() string {
	return c.System
}

// Clear resets the conversation history but keeps the system prompt and settings.
func (c *Conversation) Clear() {
	c.Messages = make([]Message, 0)
	c.Usage = Usage{}
}

// SetModel changes the model for subsequent API calls.
func (c *Conversation) SetModel(model string) {
	c.Settings.Model = model
}

// init loads the API token from environment variable or token files.
// Priority: NAI_API_KEY env var > ~/.naitoken > ./.naitoken
func init() {
	// 1. Environment variable (highest priority)
	if token := os.Getenv("NAI_API_KEY"); token != "" {
		DefaultApiToken = token
		return
	}

	// 2. Home directory token file
	if home, err := os.UserHomeDir(); err == nil {
		if token := readTokenFile(home + "/.naitoken"); token != "" {
			DefaultApiToken = token
			return
		}
	}

	// 3. Current directory token file
	if token := readTokenFile(".naitoken"); token != "" {
		DefaultApiToken = token
	}
}

// readTokenFile reads a token from a file, returning empty string on error.
func readTokenFile(path string) string {
	data, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(data))
}

// ==========================================================================
// Rich Content Methods
// ==========================================================================

// SendRich sends a message with rich content blocks and returns a full response.
// NovelAI doesn't support rich content natively, so this extracts text from
// content blocks and delegates to Send.
//
// If content is nil or empty, continues from the last message.
func (c *Conversation) SendRich(content []llmapi.ContentBlock, sampling llmapi.Sampling) (*llmapi.RichResponse, error) {
	// Extract text from content blocks
	text := extractTextFromBlocks(content)

	reply, stopReason, inputTokens, outputTokens, err := c.Send(text, sampling)
	if err != nil {
		return nil, err
	}

	return &llmapi.RichResponse{
		Content: []llmapi.ContentBlock{
			llmapi.NewTextBlock(reply),
		},
		StopReason:   stopReason,
		InputTokens:  inputTokens,
		OutputTokens: outputTokens,
	}, nil
}

// SendRichStreaming sends rich content with streaming.
// NovelAI doesn't support rich content natively, so this extracts text and
// delegates to SendStreaming.
func (c *Conversation) SendRichStreaming(content []llmapi.ContentBlock, sampling llmapi.Sampling, callback llmapi.StreamCallback) (*llmapi.RichResponse, error) {
	// Extract text from content blocks
	text := extractTextFromBlocks(content)

	reply, stopReason, inputTokens, outputTokens, err := c.SendStreaming(text, sampling, callback)
	if err != nil {
		return nil, err
	}

	return &llmapi.RichResponse{
		Content: []llmapi.ContentBlock{
			llmapi.NewTextBlock(reply),
		},
		StopReason:   stopReason,
		InputTokens:  inputTokens,
		OutputTokens: outputTokens,
	}, nil
}

// AddRichMessage adds a message with multiple content blocks to the history.
// NovelAI doesn't support rich content, so this extracts text and adds a
// simple message.
func (c *Conversation) AddRichMessage(role string, content []llmapi.ContentBlock) {
	text := extractTextFromBlocks(content)
	c.Messages = append(c.Messages, Message{Role: role, Content: text})
}

// GetRichMessages returns the conversation history with full content blocks.
// Since NovelAI uses simple text messages, each message is wrapped in a
// single text content block.
func (c *Conversation) GetRichMessages() []llmapi.RichMessage {
	result := make([]llmapi.RichMessage, len(c.Messages))
	for i, msg := range c.Messages {
		result[i] = llmapi.RichMessage{
			Role: llmapi.Role(msg.Role),
			Content: []llmapi.ContentBlock{
				llmapi.NewTextBlock(msg.Content),
			},
		}
	}
	return result
}

// extractTextFromBlocks extracts text content from content blocks.
// For blocks with thinking content, it wraps them in <thinking> tags.
func extractTextFromBlocks(blocks []llmapi.ContentBlock) string {
	if len(blocks) == 0 {
		return ""
	}

	var text strings.Builder
	for _, block := range blocks {
		switch block.Type {
		case llmapi.ContentTypeText:
			text.WriteString(block.Text)
		case llmapi.ContentTypeThinking:
			if block.Thinking != nil {
				text.WriteString("<thinking>\n")
				text.WriteString(block.Thinking.Thinking)
				text.WriteString("\n</thinking>\n")
			}
		}
	}
	return text.String()
}

// ==========================================================================
// Tool Methods
// ==========================================================================

// SetTools configures the available tools for this conversation.
// NovelAI doesn't support tool use, so this is stored but not used in API calls.
func (c *Conversation) SetTools(tools []llmapi.ToolDefinition) {
	c.Tools = tools
}

// GetTools returns the currently configured tools.
func (c *Conversation) GetTools() []llmapi.ToolDefinition {
	return c.Tools
}

// ==========================================================================
// Capabilities
// ==========================================================================

// GetCapabilities returns NovelAI's supported features.
func (c *Conversation) GetCapabilities() llmapi.Capabilities {
	return llmapi.Capabilities{
		SupportsImages:      false,
		SupportsDocuments:   false,
		SupportsToolUse:     false,
		SupportsThinking:    true, // GLM-4 supports <think> blocks
		SupportsStreaming:   true,
		MaxImageSize:        0,
		SupportedImageTypes: nil,
	}
}
