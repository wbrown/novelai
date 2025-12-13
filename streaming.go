package novelai

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/wbrown/llmapi"
)

// StreamCallback is an alias for llmapi.StreamCallback for backwards compatibility.
type StreamCallback = llmapi.StreamCallback

// SendStreaming sends a message with real-time token streaming via SSE.
// The callback is invoked for each token received.
//
// Returns the same values as Send, but the callback receives tokens as they arrive.
func (c *Conversation) SendStreaming(text string, callback llmapi.StreamCallback) (
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
		return "", "", 0, 0, fmt.Errorf("cannot generate: no messages in conversation")
	}
	// Note: If text is empty and last message is "user", we generate a response to it.
	// If text is empty and last message is "assistant", we continue from that message.

	// Build prompt string from system + conversation history
	prompt := c.buildPrompt()

	req := completionRequest{
		Model:             c.Settings.Model,
		Prompt:            prompt,
		MaxTokens:         c.Settings.MaxTokens,
		Temperature:       c.Settings.Temperature,
		TopP:              c.Settings.TopP,
		TopK:              c.Settings.TopK,
		MinP:              c.Settings.MinP,
		FrequencyPenalty:  c.Settings.FrequencyPenalty,
		PresencePenalty:   c.Settings.PresencePenalty,
		RepetitionPenalty: c.Settings.RepetitionPenalty,
		Stop:              c.Settings.StopSequences,
		Stream:            true, // Enable streaming
	}

	jsonData, err := json.Marshal(req)
	if err != nil {
		return "", "", 0, 0, fmt.Errorf("error marshaling request: %w", err)
	}

	httpReq, err := http.NewRequest("POST", completionsURL, bytes.NewBuffer(jsonData))
	if err != nil {
		return "", "", 0, 0, fmt.Errorf("error creating request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+c.ApiToken)
	httpReq.Header.Set("Accept", "text/event-stream")

	// Use a client without timeout for streaming
	client := &http.Client{Timeout: 0}
	if c.HttpClient != nil && c.HttpClient.Transport != nil {
		client.Transport = c.HttpClient.Transport
	}

	// Perform request with retries
	var resp *http.Response
	for attempt := 0; attempt <= retries; attempt++ {
		resp, err = client.Do(httpReq)
		if err == nil {
			break
		}
		if attempt < retries {
			time.Sleep(retryDelay)
			httpReq, _ = http.NewRequest("POST", completionsURL, bytes.NewBuffer(jsonData))
			httpReq.Header.Set("Content-Type", "application/json")
			httpReq.Header.Set("Authorization", "Bearer "+c.ApiToken)
			httpReq.Header.Set("Accept", "text/event-stream")
		}
	}
	if err != nil {
		return "", "", 0, 0, fmt.Errorf("HTTP error after %d retries: %w", retries, err)
	}
	if resp == nil {
		return "", "", 0, 0, fmt.Errorf("HTTP response is nil")
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return "", "", 0, 0, fmt.Errorf("API error (status %d): %s", resp.StatusCode, body)
	}

	// Parse SSE stream
	reply, stopReason, err = c.parseSSEStream(resp.Body, callback)
	if err != nil {
		return reply, stopReason, 0, 0, err
	}

	// Add assistant message to history
	c.Messages = append(c.Messages, Message{Role: "assistant", Content: reply})

	// Normalize stop reason
	stopReason = normalizeStopReason(stopReason)

	// Note: Streaming responses may not include token counts in all implementations.
	// We estimate based on a rough 4 chars per token approximation.
	// Real token counts would need to be fetched from a separate endpoint or
	// accumulated from chunk metadata if provided.
	outputTokens = len(reply) / 4
	if outputTokens == 0 && len(reply) > 0 {
		outputTokens = 1
	}

	c.Usage.OutputTokens += outputTokens

	return reply, stopReason, inputTokens, outputTokens, nil
}

// parseSSEStream reads Server-Sent Events and calls the callback for each token.
func (c *Conversation) parseSSEStream(body io.Reader, callback StreamCallback) (
	fullText string,
	stopReason string,
	err error,
) {
	scanner := bufio.NewScanner(body)
	var accumulated strings.Builder

	for scanner.Scan() {
		line := scanner.Text()

		// SSE format: "data: {json}" or "data: [DONE]"
		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")

		// Check for stream end
		if data == "[DONE]" {
			if callback != nil {
				callback("", true)
			}
			break
		}

		// Parse chunk
		var chunk streamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			// Skip malformed chunks
			continue
		}

		if len(chunk.Choices) == 0 {
			continue
		}

		choice := chunk.Choices[0]

		// Extract text (completions format uses "text" not "delta.content")
		if choice.Text != "" {
			accumulated.WriteString(choice.Text)
			if callback != nil {
				callback(choice.Text, false)
			}
		}

		// Check for finish reason
		if choice.FinishReason != nil && *choice.FinishReason != "" {
			stopReason = *choice.FinishReason
		}
	}

	if err := scanner.Err(); err != nil {
		return accumulated.String(), stopReason, fmt.Errorf("error reading stream: %w", err)
	}

	return accumulated.String(), stopReason, nil
}

// SendStreamingUntilDone combines streaming with automatic continuation.
// It streams tokens via callback and continues until stopReason != "max_tokens".
func (c *Conversation) SendStreamingUntilDone(text string, callback llmapi.StreamCallback) (
	reply string,
	stopReason string,
	inputTokens int,
	outputTokens int,
	err error,
) {
	var totalReply strings.Builder
	input := text

	for {
		var partReply string
		var inToks, outToks int

		partReply, stopReason, inToks, outToks, err = c.SendStreaming(input, callback)
		if err != nil {
			return totalReply.String(), stopReason, inputTokens, outputTokens, err
		}

		totalReply.WriteString(partReply)
		inputTokens += inToks
		outputTokens += outToks

		c.MergeIfLastTwoAssistant()

		if stopReason != "max_tokens" {
			break
		}

		input = ""
	}

	return totalReply.String(), stopReason, inputTokens, outputTokens, nil
}
