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
	"encoding/json"
	"fmt"
	"io"
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
	promptCachePinTTL        = 30 * time.Minute
	promptCachePinTTLSlow    = 5 * time.Minute
	promptCacheSlowThreshold = 3 * time.Minute
	pinStoreGCInterval       = 5 * time.Minute
)

type pinContextKey struct{}

var pinCtxKey = pinContextKey{}

type pinContextValue struct {
	storeKey string
	authID   string
}

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
	s.setWithTTL(key, authID, promptCachePinTTL)
}

func (s *promptCachePinStore) setWithTTL(key, authID string, ttl time.Duration) {
	s.mu.Lock()
	s.pins[key] = promptCachePin{
		authID:    authID,
		expiresAt: time.Now().Add(ttl),
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

func (s *promptCachePinStore) delete(key string) {
	s.mu.Lock()
	delete(s.pins, key)
	s.mu.Unlock()
}

func (s *promptCachePinStore) updateAfterRequest(key, authID string, duration time.Duration) {
	if key == "" || authID == "" {
		return
	}
	if duration > promptCacheSlowThreshold {
		s.delete(key)
	} else if duration > 10*time.Second {
		s.setWithTTL(key, authID, promptCachePinTTLSlow)
	} else {
		s.set(key, authID)
	}
}

func modelAffinityKey(rawJSON []byte) string {
	return strings.TrimSpace(gjson.GetBytes(rawJSON, "model").String())
}

func scopedPromptCachePinKey(promptCacheKey string, rawJSON []byte) string {
	promptCacheKey = strings.TrimSpace(promptCacheKey)
	if promptCacheKey == "" {
		return ""
	}
	modelKey := modelAffinityKey(rawJSON)
	if modelKey == "" {
		return "pc:" + promptCacheKey
	}
	return "pc:" + modelKey + ":" + promptCacheKey
}

// computeFingerprint derives a fallback pin key from the client API key,
// requested model, and the first message content hash for per-conversation granularity.
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
	modelKey := modelAffinityKey(rawJSON)

	// Try first message content for finer granularity
	firstMsg := gjson.GetBytes(rawJSON, "messages.0.content").String() // chat completions
	if firstMsg == "" {
		firstMsg = gjson.GetBytes(rawJSON, "input.0.content").String() // responses format
	}
	if firstMsg == "" {
		if modelKey == "" {
			return "fp:" + apiKey
		}
		return "fp:" + apiKey + ":" + modelKey
	}
	h := sha256.Sum256([]byte(firstMsg))
	if modelKey == "" {
		return "fp:" + apiKey + ":" + hex.EncodeToString(h[:8])
	}
	return "fp:" + apiKey + ":" + modelKey + ":" + hex.EncodeToString(h[:8])
}

// applyPromptCacheAuthPinning injects session-affinity into the context.
func applyPromptCacheAuthPinning(cliCtx context.Context, rawJSON []byte) context.Context {
	promptCacheKey := scopedPromptCachePinKey(gjson.GetBytes(rawJSON, "prompt_cache_key").String(), rawJSON)
	fingerprint := computeFingerprint(cliCtx, rawJSON)

	updateCallback := func(storeKey, currentAuthID string) func(string) {
		return func(newAuthID string) {
			if newAuthID = strings.TrimSpace(newAuthID); newAuthID != "" && newAuthID != currentAuthID {
				pinStore.set(storeKey, newAuthID)
			}
		}
	}

	if promptCacheKey != "" {
		if authID, ok := pinStore.get(promptCacheKey); ok {
			cliCtx = handlers.WithPinnedAuthID(cliCtx, authID)
			cliCtx = handlers.WithSelectedAuthIDCallback(cliCtx, updateCallback(promptCacheKey, authID))
			cliCtx = context.WithValue(cliCtx, pinCtxKey, &pinContextValue{storeKey: promptCacheKey, authID: authID})
			return cliCtx
		}
	}

	if fingerprint != "" {
		if authID, ok := pinStore.get(fingerprint); ok {
			storeKey := fingerprint
			if promptCacheKey != "" {
				pinStore.set(promptCacheKey, authID)
				storeKey = promptCacheKey
			}
			cliCtx = handlers.WithPinnedAuthID(cliCtx, authID)
			cliCtx = handlers.WithSelectedAuthIDCallback(cliCtx, updateCallback(storeKey, authID))
			cliCtx = context.WithValue(cliCtx, pinCtxKey, &pinContextValue{storeKey: storeKey, authID: authID})
			return cliCtx
		}
	}

	storeKey := promptCacheKey
	if storeKey == "" {
		storeKey = fingerprint
	}
	if storeKey == "" {
		return cliCtx
	}
	cliCtx = handlers.WithSelectedAuthIDCallback(cliCtx, func(authID string) {
		if authID = strings.TrimSpace(authID); authID != "" {
			pinStore.set(storeKey, authID)
		}
	})
	cliCtx = context.WithValue(cliCtx, pinCtxKey, &pinContextValue{storeKey: storeKey})
	return cliCtx
}

func updatePinAfterRequest(ctx context.Context, duration time.Duration, failed bool) {
	val := ctx.Value(pinCtxKey)
	if val == nil {
		return
	}
	pinVal, ok := val.(*pinContextValue)
	if !ok || pinVal == nil || pinVal.storeKey == "" {
		return
	}
	if failed {
		pinStore.delete(pinVal.storeKey)
		return
	}
	authID := pinVal.authID
	if authID == "" {
		return
	}
	pinStore.updateAfterRequest(pinVal.storeKey, authID, duration)
}

// ---------- Responses SSE framing ----------

func writeResponsesSSEChunk(w io.Writer, chunk []byte) {
	if w == nil || len(chunk) == 0 {
		return
	}
	if _, err := w.Write(chunk); err != nil {
		return
	}
	if bytes.HasSuffix(chunk, []byte("\n\n")) || bytes.HasSuffix(chunk, []byte("\r\n\r\n")) {
		return
	}
	suffix := []byte("\n\n")
	if bytes.HasSuffix(chunk, []byte("\r\n")) {
		suffix = []byte("\r\n")
	} else if bytes.HasSuffix(chunk, []byte("\n")) {
		suffix = []byte("\n")
	}
	if _, err := w.Write(suffix); err != nil {
		return
	}
}

type responsesSSEFramer struct {
	pending []byte
}

func (f *responsesSSEFramer) WriteChunk(w io.Writer, chunk []byte) {
	if len(chunk) == 0 {
		return
	}
	if responsesSSENeedsLineBreak(f.pending, chunk) {
		f.pending = append(f.pending, '\n')
	}
	f.pending = append(f.pending, chunk...)
	for {
		frameLen := responsesSSEFrameLen(f.pending)
		if frameLen == 0 {
			break
		}
		writeResponsesSSEChunk(w, f.pending[:frameLen])
		copy(f.pending, f.pending[frameLen:])
		f.pending = f.pending[:len(f.pending)-frameLen]
	}
	if len(bytes.TrimSpace(f.pending)) == 0 {
		f.pending = f.pending[:0]
		return
	}
	if len(f.pending) == 0 || !responsesSSECanEmitWithoutDelimiter(f.pending) {
		return
	}
	writeResponsesSSEChunk(w, f.pending)
	f.pending = f.pending[:0]
}

func (f *responsesSSEFramer) Flush(w io.Writer) {
	if len(f.pending) == 0 {
		return
	}
	if len(bytes.TrimSpace(f.pending)) == 0 {
		f.pending = f.pending[:0]
		return
	}
	if !responsesSSECanEmitWithoutDelimiter(f.pending) {
		f.pending = f.pending[:0]
		return
	}
	writeResponsesSSEChunk(w, f.pending)
	f.pending = f.pending[:0]
}

func responsesSSEFrameLen(chunk []byte) int {
	if len(chunk) == 0 {
		return 0
	}
	lf := bytes.Index(chunk, []byte("\n\n"))
	crlf := bytes.Index(chunk, []byte("\r\n\r\n"))
	switch {
	case lf < 0:
		if crlf < 0 {
			return 0
		}
		return crlf + 4
	case crlf < 0:
		return lf + 2
	case lf < crlf:
		return lf + 2
	default:
		return crlf + 4
	}
}

func responsesSSENeedsMoreData(chunk []byte) bool {
	trimmed := bytes.TrimSpace(chunk)
	if len(trimmed) == 0 {
		return false
	}
	return responsesSSEHasField(trimmed, []byte("event:")) && !responsesSSEHasField(trimmed, []byte("data:"))
}

func responsesSSEHasField(chunk []byte, prefix []byte) bool {
	s := chunk
	for len(s) > 0 {
		line := s
		if i := bytes.IndexByte(s, '\n'); i >= 0 {
			line = s[:i]
			s = s[i+1:]
		} else {
			s = nil
		}
		line = bytes.TrimSpace(line)
		if bytes.HasPrefix(line, prefix) {
			return true
		}
	}
	return false
}

func responsesSSECanEmitWithoutDelimiter(chunk []byte) bool {
	trimmed := bytes.TrimSpace(chunk)
	if len(trimmed) == 0 || responsesSSENeedsMoreData(trimmed) || !responsesSSEHasField(trimmed, []byte("data:")) {
		return false
	}
	return responsesSSEDataLinesValid(trimmed)
}

func responsesSSEDataLinesValid(chunk []byte) bool {
	s := chunk
	for len(s) > 0 {
		line := s
		if i := bytes.IndexByte(s, '\n'); i >= 0 {
			line = s[:i]
			s = s[i+1:]
		} else {
			s = nil
		}
		line = bytes.TrimSpace(line)
		if len(line) == 0 || !bytes.HasPrefix(line, []byte("data:")) {
			continue
		}
		data := bytes.TrimSpace(line[len("data:"):])
		if len(data) == 0 || bytes.Equal(data, []byte("[DONE]")) {
			continue
		}
		if !json.Valid(data) {
			return false
		}
	}
	return true
}

func responsesSSENeedsLineBreak(pending, chunk []byte) bool {
	if len(pending) == 0 || len(chunk) == 0 {
		return false
	}
	if bytes.HasSuffix(pending, []byte("\n")) || bytes.HasSuffix(pending, []byte("\r")) {
		return false
	}
	if chunk[0] == '\n' || chunk[0] == '\r' {
		return false
	}
	trimmed := bytes.TrimLeft(chunk, " \t")
	if len(trimmed) == 0 {
		return false
	}
	for _, prefix := range [][]byte{[]byte("data:"), []byte("event:"), []byte("id:"), []byte("retry:"), []byte(":")} {
		if bytes.HasPrefix(trimmed, prefix) {
			return true
		}
	}
	return false
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
	startTime := time.Now()
	resp, upstreamHeaders, errMsg := h.ExecuteWithAuthManager(cliCtx, h.HandlerType(), modelName, rawJSON, "responses/compact")
	duration := time.Since(startTime)
	stopKeepAlive()
	updatePinAfterRequest(cliCtx, duration, errMsg != nil)
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

	startTime := time.Now()
	resp, upstreamHeaders, errMsg := h.ExecuteWithAuthManager(cliCtx, h.HandlerType(), modelName, rawJSON, "")
	duration := time.Since(startTime)
	stopKeepAlive()
	updatePinAfterRequest(cliCtx, duration, errMsg != nil)
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
	startTime := time.Now()
	streamFailed := false
	defer func() {
		updatePinAfterRequest(cliCtx, time.Since(startTime), streamFailed)
	}()
	dataChan, upstreamHeaders, errChan := h.ExecuteStreamWithAuthManager(cliCtx, h.HandlerType(), modelName, rawJSON, "")

	setSSEHeaders := func() {
		c.Header("Content-Type", "text/event-stream")
		c.Header("Cache-Control", "no-cache")
		c.Header("Connection", "keep-alive")
		c.Header("Access-Control-Allow-Origin", "*")
	}
	framer := &responsesSSEFramer{}

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
			streamFailed = true
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
			framer.WriteChunk(c.Writer, chunk)
			flusher.Flush()

			// Continue
			h.forwardResponsesStream(c, flusher, func(err error) { cliCancel(err) }, dataChan, errChan, framer)
			return
		}
	}
}

func (h *OpenAIResponsesAPIHandler) forwardResponsesStream(c *gin.Context, flusher http.Flusher, cancel func(error), data <-chan []byte, errs <-chan *interfaces.ErrorMessage, framer *responsesSSEFramer) {
	if framer == nil {
		framer = &responsesSSEFramer{}
	}
	h.ForwardStream(c, flusher, cancel, data, errs, handlers.StreamForwardOptions{
		WriteChunk: func(chunk []byte) {
			framer.WriteChunk(c.Writer, chunk)
		},
		WriteTerminalError: func(errMsg *interfaces.ErrorMessage) {
			framer.Flush(c.Writer)
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
			framer.Flush(c.Writer)
			_, _ = c.Writer.Write([]byte("\n"))
		},
	})
}
