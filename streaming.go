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
// Sampling parameters override conversation defaults for this call only.
//
// Returns the same values as Send, but the callback receives tokens as they arrive.
// cacheCreationTokens and cacheReadTokens are always 0 (NovelAI doesn't report cache stats).
// On a mid-stream read error the returned values carry what arrived before
// it, beside the error.
func (c *Conversation) SendStreaming(text string, sampling llmapi.Sampling, callback llmapi.StreamCallback) (
	reply string,
	stopReason string,
	inputTokens int,
	outputTokens int,
	cacheCreationTokens int,
	cacheReadTokens int,
	err error,
) {
	acct, err := c.streamExchange(text, sampling, callback)
	reply, stopReason, inputTokens, outputTokens, cacheCreationTokens, cacheReadTokens = acct.tuple()
	return reply, stopReason, inputTokens, outputTokens, cacheCreationTokens, cacheReadTokens, err
}

// streamExchange is the streaming twin of exchange: it adds the user text
// (when non-empty), sends the conversation as one SSE request, appends the
// assistant reply to history, accumulates usage, and returns the request's
// account. On a mid-stream read error the account carries what arrived before
// it, beside the error, and history and usage are left untouched.
func (c *Conversation) streamExchange(text string, sampling llmapi.Sampling, callback llmapi.StreamCallback) (requestAccount, error) {
	if c.ApiToken == "" {
		return requestAccount{}, fmt.Errorf("API token not set")
	}
	if text != "" {
		c.Messages = append(c.Messages, Message{Role: "user", Content: text})
	} else if len(c.Messages) == 0 {
		return requestAccount{}, fmt.Errorf("cannot generate: no messages in conversation")
	}

	req := c.buildRequest(sampling, true)
	jsonData, err := json.Marshal(req)
	if err != nil {
		return requestAccount{}, fmt.Errorf("error marshaling request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(c.context(), "POST", c.endpoint(), bytes.NewBuffer(jsonData))
	if err != nil {
		return requestAccount{}, fmt.Errorf("error creating request: %w", err)
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
			httpReq, _ = http.NewRequestWithContext(c.context(), "POST", c.endpoint(), bytes.NewBuffer(jsonData))
			httpReq.Header.Set("Content-Type", "application/json")
			httpReq.Header.Set("Authorization", "Bearer "+c.ApiToken)
			httpReq.Header.Set("Accept", "text/event-stream")
		}
	}
	if err != nil {
		return requestAccount{}, fmt.Errorf("HTTP error after %d retries: %w", retries, err)
	}
	if resp == nil {
		return requestAccount{}, fmt.Errorf("HTTP response is nil")
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return requestAccount{}, fmt.Errorf("API error (status %d): %s", resp.StatusCode, body)
	}

	reply, finishReason, inputTokens, outputTokens, err := c.parseSSEStream(resp.Body, callback)
	acct := requestAccount{
		text:             reply,
		finishReason:     finishReason,
		promptTokens:     inputTokens,
		completionTokens: outputTokens,
		budget:           req.MaxTokens,
	}
	if err != nil {
		return acct, err
	}

	c.Messages = append(c.Messages, Message{Role: "assistant", Content: reply})
	c.Usage.InputTokens += inputTokens
	c.Usage.OutputTokens += outputTokens
	return acct, nil
}

// parseSSEStream reads Server-Sent Events and calls the callback for each token.
func (c *Conversation) parseSSEStream(body io.Reader, callback StreamCallback) (
	fullText string,
	stopReason string,
	inputTokens int,
	outputTokens int,
	err error,
) {
	scanner := bufio.NewScanner(body)
	var accumulated strings.Builder
	var tokenCount int
	var inThink bool

	// emit routes one text segment by the current think state: reasoning is
	// surfaced to the callback tagged TokenReasoning but kept out of accumulated
	// (it is not part of the returned content); content is both accumulated and
	// emitted as TokenContent.
	emit := func(s string) {
		if s == "" {
			return
		}
		if inThink {
			if callback != nil {
				callback(llmapi.StreamDelta{Text: s, Kind: llmapi.TokenReasoning})
			}
		} else {
			accumulated.WriteString(s)
			if callback != nil {
				callback(llmapi.StreamDelta{Text: s, Kind: llmapi.TokenContent})
			}
		}
	}

	// held holds fragments whose concatenation is a partial <think>/</think> marker
	// spanning chunks; they are yielded individually (never joined) if they turn out
	// not to be a marker.
	var held []string

	// couldStartMarker reports whether s is a non-empty prefix of a marker — i.e. it
	// could still grow into <think> or </think>.
	couldStartMarker := func(s string) bool {
		return s != "" && (strings.HasPrefix("<think>", s) || strings.HasPrefix("</think>", s))
	}

	// flushHeld yields the held fragments individually under the current state.
	flushHeld := func() {
		for _, h := range held {
			emit(h)
		}
		held = nil
	}

	// routeText routes a fragment with no pending held prefix: it emits the text
	// around any complete markers within the fragment (flipping state and consuming
	// them) and holds a trailing partial marker for the next chunk.
	routeText := func(c string) {
		for c != "" {
			openIdx := strings.Index(c, "<think>")
			closeIdx := strings.Index(c, "</think>")
			switch {
			case openIdx >= 0 && (closeIdx < 0 || openIdx < closeIdx):
				emit(c[:openIdx])
				inThink = true
				c = c[openIdx+len("<think>"):]
			case closeIdx >= 0:
				emit(c[:closeIdx])
				inThink = false
				c = c[closeIdx+len("</think>"):]
			default:
				// No complete marker; hold the longest trailing partial marker.
				keep := 0
				maxK := len(c)
				if maxK > 7 { // longest proper marker prefix is "</think" (7)
					maxK = 7
				}
				for k := maxK; k >= 1; k-- {
					if couldStartMarker(c[len(c)-k:]) {
						keep = k
						break
					}
				}
				emit(c[:len(c)-keep])
				if keep > 0 {
					held = append(held, c[len(c)-keep:])
				}
				c = ""
			}
		}
	}

	// routeFragment routes one streamed fragment, reassembling a marker that may be
	// split across this and earlier chunks.
	routeFragment := func(c string) {
		if len(held) > 0 {
			combined := strings.Join(held, "") + c
			switch {
			case strings.HasPrefix(combined, "<think>"):
				inThink = true
				held = nil
				routeText(combined[len("<think>"):])
				return
			case strings.HasPrefix(combined, "</think>"):
				inThink = false
				held = nil
				routeText(combined[len("</think>"):])
				return
			case couldStartMarker(combined):
				held = append(held, c)
				return
			}
			flushHeld() // false alarm: held was not a marker — yield it individually
		}
		routeText(c)
	}

	for scanner.Scan() {
		line := scanner.Text()

		// SSE format: "data: {json}" or "data: [DONE]"
		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")

		// Check for stream end
		if data == "[DONE]" {
			flushHeld() // emit any held maybe-marker that never completed
			if callback != nil {
				callback(llmapi.StreamDelta{Done: true})
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

		// Completions format uses "text". GLM reasoning models wrap their
		// chain-of-thought in <think>/</think> markers, but NovelAI streams at
		// arbitrary byte boundaries, so a marker is often split across chunks.
		// routeFragment reassembles markers across chunks, routes reasoning vs
		// content, and yields held fragments individually if they aren't a marker.
		if choice.Text != "" {
			tokenCount++ // Count each chunk as a token
			routeFragment(choice.Text)
		}

		// Check for finish reason
		if choice.FinishReason != nil && *choice.FinishReason != "" {
			stopReason = *choice.FinishReason
		}

		// Capture usage data if API provides it (may override our count)
		if chunk.Usage != nil {
			inputTokens = chunk.Usage.PromptTokens
			outputTokens = chunk.Usage.CompletionTokens
		}
	}

	// Stream ended without [DONE]: flush any still-held maybe-marker bytes.
	flushHeld()

	if err := scanner.Err(); err != nil {
		return accumulated.String(), stopReason, inputTokens, outputTokens, fmt.Errorf("error reading stream: %w", err)
	}

	// Use our counted tokens if API didn't provide usage data
	if outputTokens == 0 {
		outputTokens = tokenCount
	}

	return accumulated.String(), stopReason, inputTokens, outputTokens, nil
}

// SendStreamingUntilDone combines streaming with automatic continuation.
// It streams tokens via callback and continues until stopReason != "max_tokens".
// Sampling parameters override conversation defaults for this call only.
// cacheCreationTokens and cacheReadTokens are always 0 (NovelAI doesn't report cache stats).
func (c *Conversation) SendStreamingUntilDone(text string, sampling llmapi.Sampling, callback llmapi.StreamCallback) (
	reply string,
	stopReason string,
	inputTokens int,
	outputTokens int,
	cacheCreationTokens int,
	cacheReadTokens int,
	err error,
) {
	var totalReply strings.Builder
	input := text

	for {
		var partReply string
		var inToks, outToks int

		partReply, stopReason, inToks, outToks, _, _, err = c.SendStreaming(input, sampling, callback)
		if err != nil {
			return totalReply.String(), stopReason, inputTokens, outputTokens, 0, 0, err
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

	return totalReply.String(), stopReason, inputTokens, outputTokens, 0, 0, nil
}
