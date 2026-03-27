// Package claude provides HTTP handlers for Claude API code-related functionality.
// This package implements Claude-compatible streaming chat completions with sophisticated
// client rotation and quota management systems to ensure high availability and optimal
// resource utilization across multiple backend clients. It handles request translation
// between Claude API format and the underlying Gemini backend, providing seamless
// API compatibility while maintaining robust error handling and connection management.
package claude

import (
	"bytes"
	"compress/gzip"
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
	"github.com/router-for-me/CLIProxyAPI/v6/internal/thinking"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/util"
	"github.com/router-for-me/CLIProxyAPI/v6/sdk/api/handlers"
	log "github.com/sirupsen/logrus"
	"github.com/tidwall/gjson"
)

// ClaudeCodeAPIHandler contains the handlers for Claude API endpoints.
// It holds a pool of clients to interact with the backend service.
const (
	claudeAuthPinTTL        = 30 * time.Minute
	claudeAuthPinTTLSlow    = 5 * time.Minute
	claudeAuthPinSlowThresh = 3 * time.Minute
	claudeAuthPinGCInterval = 5 * time.Minute
)

type claudePinContextKey struct{}

var claudePinCtxKey = claudePinContextKey{}

type claudePinContextValue struct {
	storeKey string
	authID   string
}

type claudeAuthPin struct {
	authID    string
	expiresAt time.Time
}

type claudeAuthPinStore struct {
	mu     sync.RWMutex
	pins   map[string]claudeAuthPin
	lastGC time.Time
}

var claudePinStore = &claudeAuthPinStore{
	pins: make(map[string]claudeAuthPin),
}

func (s *claudeAuthPinStore) get(key string) (string, bool) {
	s.mu.RLock()
	pin, ok := s.pins[key]
	s.mu.RUnlock()
	if !ok || time.Now().After(pin.expiresAt) {
		return "", false
	}
	return pin.authID, true
}

func (s *claudeAuthPinStore) set(key, authID string) {
	s.setWithTTL(key, authID, claudeAuthPinTTL)
}

func (s *claudeAuthPinStore) setWithTTL(key, authID string, ttl time.Duration) {
	s.mu.Lock()
	s.pins[key] = claudeAuthPin{
		authID:    authID,
		expiresAt: time.Now().Add(ttl),
	}
	if now := time.Now(); now.Sub(s.lastGC) >= claudeAuthPinGCInterval {
		for k, pin := range s.pins {
			if now.After(pin.expiresAt) {
				delete(s.pins, k)
			}
		}
		s.lastGC = now
	}
	s.mu.Unlock()
}

func (s *claudeAuthPinStore) delete(key string) {
	s.mu.Lock()
	delete(s.pins, key)
	s.mu.Unlock()
}

func (s *claudeAuthPinStore) updateAfterRequest(key, authID string, duration time.Duration) {
	if key == "" || authID == "" {
		return
	}
	if duration > claudeAuthPinSlowThresh {
		s.delete(key)
	} else if duration > 10*time.Second {
		s.setWithTTL(key, authID, claudeAuthPinTTLSlow)
	} else {
		s.set(key, authID)
	}
}

func claudeAuthPinKey(rawJSON []byte, normalizedModel string) string {
	userID := strings.TrimSpace(gjson.GetBytes(rawJSON, "metadata.user_id").String())
	if userID == "" {
		return ""
	}
	modelKey := strings.TrimSpace(normalizedModel)
	if modelKey == "" {
		modelKey = strings.TrimSpace(gjson.GetBytes(rawJSON, "model").String())
	}
	if modelKey == "" {
		return ""
	}
	firstMessage := strings.TrimSpace(gjson.GetBytes(rawJSON, "messages.0.content").String())
	if firstMessage == "" {
		return ""
	}
	h := sha256.Sum256([]byte(firstMessage))
	return "claude:" + modelKey + ":" + userID + ":" + hex.EncodeToString(h[:8])
}

func applyClaudeAuthPinning(cliCtx context.Context, rawJSON []byte, normalizedModel string) context.Context {
	storeKey := claudeAuthPinKey(rawJSON, normalizedModel)
	if storeKey == "" {
		return cliCtx
	}
	if authID, ok := claudePinStore.get(storeKey); ok {
		pinVal := &claudePinContextValue{storeKey: storeKey, authID: authID}
		cliCtx = handlers.WithPinnedAuthID(cliCtx, authID)
		cliCtx = handlers.WithSelectedAuthIDCallback(cliCtx, func(newAuthID string) {
			if newAuthID = strings.TrimSpace(newAuthID); newAuthID != "" {
				pinVal.authID = newAuthID
				if newAuthID != authID {
					claudePinStore.set(storeKey, newAuthID)
				}
			}
		})
		cliCtx = context.WithValue(cliCtx, claudePinCtxKey, pinVal)
		return cliCtx
	}
	pinVal := &claudePinContextValue{storeKey: storeKey}
	cliCtx = handlers.WithSelectedAuthIDCallback(cliCtx, func(authID string) {
		if authID = strings.TrimSpace(authID); authID != "" {
			pinVal.authID = authID
			claudePinStore.set(storeKey, authID)
		}
	})
	cliCtx = context.WithValue(cliCtx, claudePinCtxKey, pinVal)
	return cliCtx
}

func updateClaudePinAfterRequest(ctx context.Context, duration time.Duration) {
	val := ctx.Value(claudePinCtxKey)
	if val == nil {
		return
	}
	pinVal, ok := val.(*claudePinContextValue)
	if !ok || pinVal == nil || pinVal.storeKey == "" {
		return
	}
	authID := pinVal.authID
	if authID == "" {
		return
	}
	claudePinStore.updateAfterRequest(pinVal.storeKey, authID, duration)
}

func hasCodexProvider(providers []string) bool {
	for _, provider := range providers {
		if strings.EqualFold(strings.TrimSpace(provider), "codex") {
			return true
		}
	}
	return false
}

func resolveRequestDetails(modelName string) (providers []string, normalizedModel string, err *interfaces.ErrorMessage) {
	resolvedModelName := modelName
	initialSuffix := thinking.ParseSuffix(modelName)
	if initialSuffix.ModelName == "auto" {
		resolvedBase := util.ResolveAutoModel(initialSuffix.ModelName)
		if initialSuffix.HasSuffix {
			resolvedModelName = fmt.Sprintf("%s(%s)", resolvedBase, initialSuffix.RawSuffix)
		} else {
			resolvedModelName = resolvedBase
		}
	} else {
		resolvedModelName = util.ResolveAutoModel(modelName)
	}

	parsed := thinking.ParseSuffix(resolvedModelName)
	baseModel := strings.TrimSpace(parsed.ModelName)

	providers = util.GetProviderName(baseModel)
	if len(providers) == 0 && baseModel != resolvedModelName {
		providers = util.GetProviderName(resolvedModelName)
	}
	if len(providers) == 0 {
		return nil, "", &interfaces.ErrorMessage{StatusCode: http.StatusBadGateway, Error: fmt.Errorf("unknown provider for model %s", modelName)}
	}
	return providers, resolvedModelName, nil
}

type ClaudeCodeAPIHandler struct {
	*handlers.BaseAPIHandler
}

// NewClaudeCodeAPIHandler creates a new Claude API handlers instance.
// It takes an BaseAPIHandler instance as input and returns a ClaudeCodeAPIHandler.
//
// Parameters:
//   - apiHandlers: The base API handler instance.
//
// Returns:
//   - *ClaudeCodeAPIHandler: A new Claude code API handler instance.
func NewClaudeCodeAPIHandler(apiHandlers *handlers.BaseAPIHandler) *ClaudeCodeAPIHandler {
	return &ClaudeCodeAPIHandler{
		BaseAPIHandler: apiHandlers,
	}
}

// HandlerType returns the identifier for this handler implementation.
func (h *ClaudeCodeAPIHandler) HandlerType() string {
	return Claude
}

// Models returns a list of models supported by this handler.
func (h *ClaudeCodeAPIHandler) Models() []map[string]any {
	// Get dynamic models from the global registry
	modelRegistry := registry.GetGlobalRegistry()
	return modelRegistry.GetAvailableModels("claude")
}

// ClaudeMessages handles Claude-compatible streaming chat completions.
// This function implements a sophisticated client rotation and quota management system
// to ensure high availability and optimal resource utilization across multiple backend clients.
//
// Parameters:
//   - c: The Gin context for the request.
func (h *ClaudeCodeAPIHandler) ClaudeMessages(c *gin.Context) {
	// Extract raw JSON data from the incoming request
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
	if !streamResult.Exists() || streamResult.Type == gjson.False {
		h.handleNonStreamingResponse(c, rawJSON)
	} else {
		h.handleStreamingResponse(c, rawJSON)
	}
}

// ClaudeMessages handles Claude-compatible streaming chat completions.
// This function implements a sophisticated client rotation and quota management system
// to ensure high availability and optimal resource utilization across multiple backend clients.
//
// Parameters:
//   - c: The Gin context for the request.
func (h *ClaudeCodeAPIHandler) ClaudeCountTokens(c *gin.Context) {
	// Extract raw JSON data from the incoming request
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

	c.Header("Content-Type", "application/json")

	alt := h.GetAlt(c)
	cliCtx, cliCancel := h.GetContextWithCancel(h, c, context.Background())

	modelName := gjson.GetBytes(rawJSON, "model").String()

	resp, upstreamHeaders, errMsg := h.ExecuteCountWithAuthManager(cliCtx, h.HandlerType(), modelName, rawJSON, alt)
	if errMsg != nil {
		h.WriteErrorResponse(c, errMsg)
		cliCancel(errMsg.Error)
		return
	}
	handlers.WriteUpstreamHeaders(c.Writer.Header(), upstreamHeaders)
	_, _ = c.Writer.Write(resp)
	cliCancel()
}

// ClaudeModels handles the Claude models listing endpoint.
// It returns a JSON response containing available Claude models and their specifications.
//
// Parameters:
//   - c: The Gin context for the request.
func (h *ClaudeCodeAPIHandler) ClaudeModels(c *gin.Context) {
	models := h.Models()
	firstID := ""
	lastID := ""
	if len(models) > 0 {
		if id, ok := models[0]["id"].(string); ok {
			firstID = id
		}
		if id, ok := models[len(models)-1]["id"].(string); ok {
			lastID = id
		}
	}

	c.JSON(http.StatusOK, gin.H{
		"data":     models,
		"has_more": false,
		"first_id": firstID,
		"last_id":  lastID,
	})
}

// handleNonStreamingResponse handles non-streaming content generation requests for Claude models.
// This function processes the request synchronously and returns the complete generated
// response in a single API call. It supports various generation parameters and
// response formats.
//
// Parameters:
//   - c: The Gin context for the request
//   - modelName: The name of the Gemini model to use for content generation
//   - rawJSON: The raw JSON request body containing generation parameters and content
func (h *ClaudeCodeAPIHandler) handleNonStreamingResponse(c *gin.Context, rawJSON []byte) {
	c.Header("Content-Type", "application/json")
	alt := h.GetAlt(c)
	cliCtx, cliCancel := h.GetContextWithCancel(h, c, context.Background())

	modelName := gjson.GetBytes(rawJSON, "model").String()
	providers, normalizedModel, errMsg := resolveRequestDetails(modelName)
	if errMsg != nil {
		h.WriteErrorResponse(c, errMsg)
		cliCancel(errMsg.Error)
		return
	}
	if hasCodexProvider(providers) {
		cliCtx = applyClaudeAuthPinning(cliCtx, rawJSON, normalizedModel)
	}
	stopKeepAlive := h.StartNonStreamingKeepAlive(c, cliCtx)
	startTime := time.Now()

	resp, upstreamHeaders, errMsg := h.ExecuteWithAuthManager(cliCtx, h.HandlerType(), modelName, rawJSON, alt)
	duration := time.Since(startTime)
	stopKeepAlive()
	updateClaudePinAfterRequest(cliCtx, duration)
	if errMsg != nil {
		h.WriteErrorResponse(c, errMsg)
		cliCancel(errMsg.Error)
		return
	}

	// Decompress gzipped responses - Claude API sometimes returns gzip without Content-Encoding header
	// This fixes title generation and other non-streaming responses that arrive compressed
	if len(resp) >= 2 && resp[0] == 0x1f && resp[1] == 0x8b {
		gzReader, errGzip := gzip.NewReader(bytes.NewReader(resp))
		if errGzip != nil {
			log.Warnf("failed to decompress gzipped Claude response: %v", errGzip)
		} else {
			defer func() {
				if errClose := gzReader.Close(); errClose != nil {
					log.Warnf("failed to close Claude gzip reader: %v", errClose)
				}
			}()
			decompressed, errRead := io.ReadAll(gzReader)
			if errRead != nil {
				log.Warnf("failed to read decompressed Claude response: %v", errRead)
			} else {
				resp = decompressed
			}
		}
	}

	handlers.WriteUpstreamHeaders(c.Writer.Header(), upstreamHeaders)
	_, _ = c.Writer.Write(resp)
	cliCancel()
}

// handleStreamingResponse streams Claude-compatible responses backed by Gemini.
// It sets up SSE, selects a backend client with rotation/quota logic,
// forwards chunks, and translates them to Claude CLI format.
//
// Parameters:
//   - c: The Gin context for the request.
//   - rawJSON: The raw JSON request body.
func (h *ClaudeCodeAPIHandler) handleStreamingResponse(c *gin.Context, rawJSON []byte) {
	// Get the http.Flusher interface to manually flush the response.
	// This is crucial for streaming as it allows immediate sending of data chunks
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

	modelName := gjson.GetBytes(rawJSON, "model").String()
	providers, normalizedModel, errMsg := resolveRequestDetails(modelName)
	if errMsg != nil {
		h.WriteErrorResponse(c, errMsg)
		return
	}

	// Create a cancellable context for the backend client request
	// This allows proper cleanup and cancellation of ongoing requests
	cliCtx, cliCancel := h.GetContextWithCancel(h, c, context.Background())
	if hasCodexProvider(providers) {
		cliCtx = applyClaudeAuthPinning(cliCtx, rawJSON, normalizedModel)
	}
	startTime := time.Now()
	defer func() {
		updateClaudePinAfterRequest(cliCtx, time.Since(startTime))
	}()

	dataChan, upstreamHeaders, errChan := h.ExecuteStreamWithAuthManager(cliCtx, h.HandlerType(), modelName, rawJSON, "")
	setSSEHeaders := func() {
		c.Header("Content-Type", "text/event-stream")
		c.Header("Cache-Control", "no-cache")
		c.Header("Connection", "keep-alive")
		c.Header("Access-Control-Allow-Origin", "*")
	}

	// Peek at the first chunk to determine success or failure before setting headers
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
				// Stream closed without data? Send DONE or just headers.
				setSSEHeaders()
				handlers.WriteUpstreamHeaders(c.Writer.Header(), upstreamHeaders)
				flusher.Flush()
				cliCancel(nil)
				return
			}

			// Success! Set headers now.
			setSSEHeaders()
			handlers.WriteUpstreamHeaders(c.Writer.Header(), upstreamHeaders)

			// Write the first chunk
			if len(chunk) > 0 {
				_, _ = c.Writer.Write(chunk)
				flusher.Flush()
			}

			// Continue streaming the rest
			h.forwardClaudeStream(c, flusher, func(err error) { cliCancel(err) }, dataChan, errChan)
			return
		}
	}
}

func (h *ClaudeCodeAPIHandler) forwardClaudeStream(c *gin.Context, flusher http.Flusher, cancel func(error), data <-chan []byte, errs <-chan *interfaces.ErrorMessage) {
	h.ForwardStream(c, flusher, cancel, data, errs, handlers.StreamForwardOptions{
		WriteChunk: func(chunk []byte) {
			if len(chunk) == 0 {
				return
			}
			_, _ = c.Writer.Write(chunk)
		},
		WriteTerminalError: func(errMsg *interfaces.ErrorMessage) {
			if errMsg == nil {
				return
			}
			status := http.StatusInternalServerError
			if errMsg.StatusCode > 0 {
				status = errMsg.StatusCode
			}
			c.Status(status)

			errorBytes, _ := json.Marshal(h.toClaudeError(errMsg))
			_, _ = fmt.Fprintf(c.Writer, "event: error\ndata: %s\n\n", errorBytes)
		},
	})
}

type claudeErrorDetail struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

type claudeErrorResponse struct {
	Type  string            `json:"type"`
	Error claudeErrorDetail `json:"error"`
}

func (h *ClaudeCodeAPIHandler) toClaudeError(msg *interfaces.ErrorMessage) claudeErrorResponse {
	return claudeErrorResponse{
		Type: "error",
		Error: claudeErrorDetail{
			Type:    "api_error",
			Message: msg.Error.Error(),
		},
	}
}
