# novelai

Go client library for NovelAI's OpenAI-compatible GLM-4 API.

Mirrors the API design of [github.com/wbrown/anthropic](https://github.com/wbrown/anthropic) for easy provider switching.

## Installation

```bash
go get github.com/wbrown/novelai
```

## Usage

```go
package main

import (
    "fmt"
    "github.com/wbrown/novelai"
)

func main() {
    // Create conversation (uses NAI_API_KEY env var by default)
    conv := novelai.NewConversation("You are a helpful assistant.")
    
    // Simple send
    reply, stopReason, inTok, outTok, err := conv.Send("Hello!")
    if err != nil {
        panic(err)
    }
    fmt.Printf("Reply: %s\nStop: %s, Tokens: %d/%d\n", reply, stopReason, inTok, outTok)
    
    // Disable thinking for faster responses
    conv.Settings.Thinking = false
    reply, _, _, _, _ = conv.Send("What is 2+2?")
    fmt.Println(reply) // "4" (no <think> block)
}
```

### Streaming

```go
conv := novelai.NewConversation("You are a helpful assistant.")

callback := func(text string, done bool) {
    fmt.Print(text)
    if done {
        fmt.Println()
    }
}

reply, stopReason, _, _, err := conv.SendStreaming("Tell me a story.", callback)
```

### Settings

```go
conv.Settings.Model = "glm-4-6"      // Model selection
conv.Settings.MaxTokens = 2048       // Max tokens to generate
conv.Settings.Temperature = 1.0      // Sampling temperature
conv.Settings.Thinking = true        // Enable/disable <think> blocks
conv.Settings.StopSequences = []string{"<|user|>"}
```

## Environment Variables

- `NAI_API_KEY` - Your NovelAI API token

## Testing

```bash
# Unit tests
go test -v ./...

# Integration tests (requires API key)
NAI_API_KEY=your_token go test -v -tags=integration
```

## License

MIT


