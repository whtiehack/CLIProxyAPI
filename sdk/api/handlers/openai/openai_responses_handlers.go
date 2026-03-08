// Package openai provides HTTP handlers for OpenAIResponses API endpoints.
// This package implements the OpenAIResponses-compatible API interface, including model listing
// and chat completion functionality. It supports both streaming and non-streaming responses,
// and manages a pool of clients to interact with backend services.
// The handlers translate OpenAIResponses API requests to the appropriate backend format and
// convert responses back to OpenAIResponses-compatible format.
package openai

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	. "github.com/router-for-me/CLIProxyAPI/v6/internal/constant"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/interfaces"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/registry"
	"github.com/router-for-me/CLIProxyAPI/v6/sdk/api/handlers"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// ---------- prompt_cache_key session affinity ----------

const (
	promptCachePinTTL = 30 * time.Minute
	pinStoreGCInterval = 5 * time.Minute
)

type promptCachePin struct {
	authID    string
	expiresAt time.Time
}

type promptCachePinStore struct {
	mu     sync.RWMutex
	pins   map[string]promptCachePin
	lastGC time.Time
}

var pinStore = &promptCachePinStore{
	pins: make(map[string]promptCachePin),
}

func (s *promptCachePinStore) get(key string) (string, bool) {
	s.mu.RLock()
	pin, ok := s.pins[key]
	s.mu.RUnlock()
	if !ok || time.Now().After(pin.expiresAt) {
		return "", false
	}
	return pin.authID, true
}

func (s *promptCachePinStore) set(key, authID string) {
	s.mu.Lock()
	s.pins[key] = promptCachePin{
		authID:    authID,
		expiresAt: time.Now().Add(promptCachePinTTL),
	}
	if now := time.Now(); now.Sub(s.lastGC) >= pinStoreGCInterval {
		for k, pin := range s.pins {
			if now.After(pin.expiresAt) {
				delete(s.pins, k)
			}
		}
		s.lastGC = now
	}
	s.mu.Unlock()
}

// computeFingerprint derives a fallback pin key from the client API key
// and the first message content hash for per-conversation granularity.
func computeFingerprint(ctx context.Context, rawJSON []byte) string {
	ginCtx, ok := ctx.Value("gin").(*gin.Context)
	if !ok || ginCtx == nil {
		return ""
	}
	var apiKey string
	if v, exists := ginCtx.Get("apiKey"); exists {
		if s, ok := v.(string); ok {
			apiKey = strings.TrimSpace(s)
		}
	}
	if apiKey == "" {
		return ""
	}

	// Try first message content for finer granularity
	firstMsg := gjson.GetBytes(rawJSON, "messages.0.content").String() // chat completions
	if firstMsg == "" {
		firstMsg = gjson.GetBytes(rawJSON, "input.0.content").String() // responses format
	}
	if firstMsg == "" {
		return "fp:" + apiKey
	}
	h := sha256.Sum256([]byte(firstMsg))
	return "fp:" + apiKey + ":" + hex.EncodeToString(h[:8])
}

// applyPromptCacheAuthPinning injects session-affinity into the context.
// Pin key priority: prompt_cache_key (body) → fingerprint (client apiKey).
// When prompt_cache_key first appears and a fingerprint pin already exists,
// the pin is migrated to prompt_cache_key so subsequent requests hit directly.
// A callback is always attached so that if the pinned auth fails and the
// conductor falls back to another auth, the pin is updated automatically.
func applyPromptCacheAuthPinning(cliCtx context.Context, rawJSON []byte) context.Context {
	promptCacheKey := strings.TrimSpace(gjson.GetBytes(rawJSON, "prompt_cache_key").String())
	fingerprint := computeFingerprint(cliCtx, rawJSON)

	// updateCallback returns a callback that updates the pin when a different auth is selected (fallback).
	updateCallback := func(storeKey, currentAuthID string) func(string) {
		return func(newAuthID string) {
			if newAuthID = strings.TrimSpace(newAuthID); newAuthID != "" && newAuthID != currentAuthID {
				pinStore.set(storeKey, newAuthID)
			}
		}
	}

	// 1. prompt_cache_key has a pin → use it
	if promptCacheKey != "" {
		if authID, ok := pinStore.get(promptCacheKey); ok {
			cliCtx = handlers.WithPinnedAuthID(cliCtx, authID)
			cliCtx = handlers.WithSelectedAuthIDCallback(cliCtx, updateCallback(promptCacheKey, authID))
			return cliCtx
		}
	}

	// 2. fingerprint has a pin → use it (migrate to prompt_cache_key if available)
	if fingerprint != "" {
		if authID, ok := pinStore.get(fingerprint); ok {
			storeKey := fingerprint
			if promptCacheKey != "" {
				pinStore.set(promptCacheKey, authID)
				storeKey = promptCacheKey
			}
			cliCtx = handlers.WithPinnedAuthID(cliCtx, authID)
			cliCtx = handlers.WithSelectedAuthIDCallback(cliCtx, updateCallback(storeKey, authID))
			return cliCtx
		}
	}

	// 3. No pin exists → first request, capture auth via callback
	storeKey := promptCacheKey
	if storeKey == "" {
		storeKey = fingerprint
	}
	if storeKey == "" {
		return cliCtx
	}
	return handlers.WithSelectedAuthIDCallback(cliCtx, func(authID string) {
		if authID = strings.TrimSpace(authID); authID != "" {
			pinStore.set(storeKey, authID)
		}
	})
}

// OpenAIResponsesAPIHandler contains the handlers for OpenAIResponses API endpoints.
// It holds a pool of clients to interact with the backend service.
type OpenAIResponsesAPIHandler struct {
	*handlers.BaseAPIHandler
}

// NewOpenAIResponsesAPIHandler creates a new OpenAIResponses API handlers instance.
// It takes an BaseAPIHandler instance as input and returns an OpenAIResponsesAPIHandler.
//
// Parameters:
//   - apiHandlers: The base API handlers instance
//
// Returns:
//   - *OpenAIResponsesAPIHandler: A new OpenAIResponses API handlers instance
func NewOpenAIResponsesAPIHandler(apiHandlers *handlers.BaseAPIHandler) *OpenAIResponsesAPIHandler {
	return &OpenAIResponsesAPIHandler{
		BaseAPIHandler: apiHandlers,
	}
}

// HandlerType returns the identifier for this handler implementation.
func (h *OpenAIResponsesAPIHandler) HandlerType() string {
	return OpenaiResponse
}

// Models returns the OpenAIResponses-compatible model metadata supported by this handler.
func (h *OpenAIResponsesAPIHandler) Models() []map[string]any {
	// Get dynamic models from the global registry
	modelRegistry := registry.GetGlobalRegistry()
	return modelRegistry.GetAvailableModels("openai")
}

// OpenAIResponsesModels handles the /v1/models endpoint.
// It returns a list of available AI models with their capabilities
// and specifications in OpenAIResponses-compatible format.
func (h *OpenAIResponsesAPIHandler) OpenAIResponsesModels(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"object": "list",
		"data":   h.Models(),
	})
}

// Responses handles the /v1/responses endpoint.
// It determines whether the request is for a streaming or non-streaming response
// and calls the appropriate handler based on the model provider.
//
// Parameters:
//   - c: The Gin context containing the HTTP request and response
func (h *OpenAIResponsesAPIHandler) Responses(c *gin.Context) {
	rawJSON, err := c.GetRawData()
	// If data retrieval fails, return a 400 Bad Request error.
	if err != nil {
		c.JSON(http.StatusBadRequest, handlers.ErrorResponse{
			Error: handlers.ErrorDetail{
				Message: fmt.Sprintf("Invalid request: %v", err),
				Type:    "invalid_request_error",
			},
		})
		return
	}

	// Check if the client requested a streaming response.
	streamResult := gjson.GetBytes(rawJSON, "stream")
	if streamResult.Type == gjson.True {
		h.handleStreamingResponse(c, rawJSON)
	} else {
		h.handleNonStreamingResponse(c, rawJSON)
	}

}

func (h *OpenAIResponsesAPIHandler) Compact(c *gin.Context) {
	rawJSON, err := c.GetRawData()
	if err != nil {
		c.JSON(http.StatusBadRequest, handlers.ErrorResponse{
			Error: handlers.ErrorDetail{
				Message: fmt.Sprintf("Invalid request: %v", err),
				Type:    "invalid_request_error",
			},
		})
		return
	}

	streamResult := gjson.GetBytes(rawJSON, "stream")
	if streamResult.Type == gjson.True {
		c.JSON(http.StatusBadRequest, handlers.ErrorResponse{
			Error: handlers.ErrorDetail{
				Message: "Streaming not supported for compact responses",
				Type:    "invalid_request_error",
			},
		})
		return
	}
	if streamResult.Exists() {
		if updated, err := sjson.DeleteBytes(rawJSON, "stream"); err == nil {
			rawJSON = updated
		}
	}

	c.Header("Content-Type", "application/json")
	modelName := gjson.GetBytes(rawJSON, "model").String()
	cliCtx, cliCancel := h.GetContextWithCancel(h, c, context.Background())
	cliCtx = applyPromptCacheAuthPinning(cliCtx, rawJSON)
	stopKeepAlive := h.StartNonStreamingKeepAlive(c, cliCtx)
	resp, upstreamHeaders, errMsg := h.ExecuteWithAuthManager(cliCtx, h.HandlerType(), modelName, rawJSON, "responses/compact")
	stopKeepAlive()
	if errMsg != nil {
		h.WriteErrorResponse(c, errMsg)
		cliCancel(errMsg.Error)
		return
	}
	handlers.WriteUpstreamHeaders(c.Writer.Header(), upstreamHeaders)
	_, _ = c.Writer.Write(resp)
	cliCancel()
}

// handleNonStreamingResponse handles non-streaming chat completion responses
// for Gemini models. It selects a client from the pool, sends the request, and
// aggregates the response before sending it back to the client in OpenAIResponses format.
//
// Parameters:
//   - c: The Gin context containing the HTTP request and response
//   - rawJSON: The raw JSON bytes of the OpenAIResponses-compatible request
func (h *OpenAIResponsesAPIHandler) handleNonStreamingResponse(c *gin.Context, rawJSON []byte) {
	c.Header("Content-Type", "application/json")

	modelName := gjson.GetBytes(rawJSON, "model").String()
	cliCtx, cliCancel := h.GetContextWithCancel(h, c, context.Background())
	cliCtx = applyPromptCacheAuthPinning(cliCtx, rawJSON)
	stopKeepAlive := h.StartNonStreamingKeepAlive(c, cliCtx)

	resp, upstreamHeaders, errMsg := h.ExecuteWithAuthManager(cliCtx, h.HandlerType(), modelName, rawJSON, "")
	stopKeepAlive()
	if errMsg != nil {
		h.WriteErrorResponse(c, errMsg)
		cliCancel(errMsg.Error)
		return
	}
	handlers.WriteUpstreamHeaders(c.Writer.Header(), upstreamHeaders)
	_, _ = c.Writer.Write(resp)
	cliCancel()
}

// handleStreamingResponse handles streaming responses for Gemini models.
// It establishes a streaming connection with the backend service and forwards
// the response chunks to the client in real-time using Server-Sent Events.
//
// Parameters:
//   - c: The Gin context containing the HTTP request and response
//   - rawJSON: The raw JSON bytes of the OpenAIResponses-compatible request
func (h *OpenAIResponsesAPIHandler) handleStreamingResponse(c *gin.Context, rawJSON []byte) {
	// Get the http.Flusher interface to manually flush the response.
	flusher, ok := c.Writer.(http.Flusher)
	if !ok {
		c.JSON(http.StatusInternalServerError, handlers.ErrorResponse{
			Error: handlers.ErrorDetail{
				Message: "Streaming not supported",
				Type:    "server_error",
			},
		})
		return
	}

	// New core execution path
	modelName := gjson.GetBytes(rawJSON, "model").String()
	cliCtx, cliCancel := h.GetContextWithCancel(h, c, context.Background())
	cliCtx = applyPromptCacheAuthPinning(cliCtx, rawJSON)
	dataChan, upstreamHeaders, errChan := h.ExecuteStreamWithAuthManager(cliCtx, h.HandlerType(), modelName, rawJSON, "")

	setSSEHeaders := func() {
		c.Header("Content-Type", "text/event-stream")
		c.Header("Cache-Control", "no-cache")
		c.Header("Connection", "keep-alive")
		c.Header("Access-Control-Allow-Origin", "*")
	}

	// Peek at the first chunk
	for {
		select {
		case <-c.Request.Context().Done():
			cliCancel(c.Request.Context().Err())
			return
		case errMsg, ok := <-errChan:
			if !ok {
				// Err channel closed cleanly; wait for data channel.
				errChan = nil
				continue
			}
			// Upstream failed immediately. Return proper error status and JSON.
			h.WriteErrorResponse(c, errMsg)
			if errMsg != nil {
				cliCancel(errMsg.Error)
			} else {
				cliCancel(nil)
			}
			return
		case chunk, ok := <-dataChan:
			if !ok {
				// Stream closed without data? Send headers and done.
				setSSEHeaders()
				handlers.WriteUpstreamHeaders(c.Writer.Header(), upstreamHeaders)
				_, _ = c.Writer.Write([]byte("\n"))
				flusher.Flush()
				cliCancel(nil)
				return
			}

			// Success! Set headers.
			setSSEHeaders()
			handlers.WriteUpstreamHeaders(c.Writer.Header(), upstreamHeaders)

			// Write first chunk logic (matching forwardResponsesStream)
			if bytes.HasPrefix(chunk, []byte("event:")) {
				_, _ = c.Writer.Write([]byte("\n"))
			}
			_, _ = c.Writer.Write(chunk)
			_, _ = c.Writer.Write([]byte("\n"))
			flusher.Flush()

			// Continue
			h.forwardResponsesStream(c, flusher, func(err error) { cliCancel(err) }, dataChan, errChan)
			return
		}
	}
}

func (h *OpenAIResponsesAPIHandler) forwardResponsesStream(c *gin.Context, flusher http.Flusher, cancel func(error), data <-chan []byte, errs <-chan *interfaces.ErrorMessage) {
	h.ForwardStream(c, flusher, cancel, data, errs, handlers.StreamForwardOptions{
		WriteChunk: func(chunk []byte) {
			if bytes.HasPrefix(chunk, []byte("event:")) {
				_, _ = c.Writer.Write([]byte("\n"))
			}
			_, _ = c.Writer.Write(chunk)
			_, _ = c.Writer.Write([]byte("\n"))
		},
		WriteTerminalError: func(errMsg *interfaces.ErrorMessage) {
			if errMsg == nil {
				return
			}
			status := http.StatusInternalServerError
			if errMsg.StatusCode > 0 {
				status = errMsg.StatusCode
			}
			errText := http.StatusText(status)
			if errMsg.Error != nil && errMsg.Error.Error() != "" {
				errText = errMsg.Error.Error()
			}
			chunk := handlers.BuildOpenAIResponsesStreamErrorChunk(status, errText, 0)
			_, _ = fmt.Fprintf(c.Writer, "\nevent: error\ndata: %s\n\n", string(chunk))
		},
		WriteDone: func() {
			_, _ = c.Writer.Write([]byte("\n"))
		},
	})
}
