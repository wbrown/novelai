package novelai

// ThinkFormat defines the prompt format for controlling thinking mode.
// Different model versions use different conventions for enabling/disabling
// extended thinking (<think> blocks).
type ThinkFormat struct {
	// UserSuffix is appended to the last user message when thinking is disabled.
	// Example: "/nothink"
	UserSuffix string

	// AssistantPrefix is prepended to the assistant response when thinking is disabled.
	// This "prefills" the response to skip the thinking phase.
	// Example: "<think></think>\n" (GLM-4.6) or "</think>" (GLM-4.7)
	AssistantPrefix string
}

// Predefined think formats for different model versions.
var (
	// ThinkFormatGLM46 is the format for GLM-4.6 models.
	// Disables thinking by appending /nothink and prefilling empty think block.
	ThinkFormatGLM46 = ThinkFormat{
		UserSuffix:      "/nothink",
		AssistantPrefix: "</think>\n",
	}

	// ThinkFormatGLM47 is the format for GLM-4.7 models.
	// Disables thinking by appending /nothink and prefilling bare closing tag.
	ThinkFormatGLM47 = ThinkFormat{
		UserSuffix:      "/nothink",
		AssistantPrefix: "</think>",
	}

	// ThinkFormatNone disables think formatting entirely.
	// Use this for models that don't support thinking mode.
	ThinkFormatNone = ThinkFormat{
		UserSuffix:      "",
		AssistantPrefix: "",
	}
)

// Settings configures generation parameters for NovelAI.
type Settings struct {
	// Model to use for generation (e.g., "glm-4-6", "llama-3-erato-v1")
	Model string
	// MaxTokens is the maximum number of tokens to generate.
	MaxTokens int
	// Temperature controls randomness. Range: 0.0 to 2.0.
	Temperature float64
	// TopP is nucleus sampling parameter.
	TopP float64
	// TopK limits vocabulary to top K tokens.
	TopK int
	// MinP is minimum probability threshold.
	MinP float64
	// FrequencyPenalty penalizes frequent tokens.
	FrequencyPenalty float64
	// PresencePenalty penalizes tokens that have appeared.
	PresencePenalty float64
	// RepetitionPenalty is an alternative repetition control.
	RepetitionPenalty float64
	// StopSequences are strings that stop generation.
	StopSequences []string
	// Thinking enables GLM's extended thinking mode (<think> blocks).
	// When false, uses ThinkFormat to disable reasoning output.
	Thinking bool
	// ThinkFormat specifies the prompt format for disabling thinking mode.
	// Different model versions require different formats.
	// If nil, defaults to ThinkFormatGLM46 for backwards compatibility.
	ThinkFormat *ThinkFormat
}

// DefaultSettings provides reasonable defaults for NovelAI GLM-4.
var DefaultSettings = Settings{
	Model:         "glm-4-6",
	MaxTokens:     2048,
	Temperature:   1.0,
	TopP:          0.9,
	StopSequences: []string{"<|user|>", "<|system|>"},
	Thinking:      false,             // Disable thinking by default for faster responses
	ThinkFormat:   &ThinkFormatGLM46, // Default to GLM-4.6 format
}

// Message represents a single message in a conversation.
// Unlike Anthropic's ContentBlock array format, NovelAI uses
// simple string content following the OpenAI chat format.
type Message struct {
	Role    string `json:"role"`    // "system", "user", "assistant"
	Content string `json:"content"` // The message text
}

// Usage tracks token consumption for a conversation.
type Usage struct {
	InputTokens  int
	OutputTokens int
}

// streamOptions controls streaming behavior options.
type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

// completionRequest is the OpenAI-compatible completions request format for NovelAI.
type completionRequest struct {
	Model             string         `json:"model"`
	Prompt            string         `json:"prompt"`
	MaxTokens         int            `json:"max_tokens,omitempty"`
	Temperature       float64        `json:"temperature,omitempty"`
	TopP              float64        `json:"top_p,omitempty"`
	TopK              int            `json:"top_k,omitempty"`
	MinP              float64        `json:"min_p,omitempty"`
	FrequencyPenalty  float64        `json:"frequency_penalty,omitempty"`
	PresencePenalty   float64        `json:"presence_penalty,omitempty"`
	RepetitionPenalty float64        `json:"repetition_penalty,omitempty"`
	Stream            bool           `json:"stream,omitempty"`
	StreamOptions     *streamOptions `json:"stream_options,omitempty"`
	Stop              []string       `json:"stop,omitempty"`
}

// completionResponse is the OpenAI-compatible completions response format from NovelAI.
type completionResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index        int    `json:"index"`
		Text         string `json:"text"`
		FinishReason string `json:"finish_reason"` // "stop", "length"
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

// streamChunk represents a single SSE chunk during streaming (completions format).
type streamChunk struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index        int     `json:"index"`
		Text         string  `json:"text"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage,omitempty"`
}
