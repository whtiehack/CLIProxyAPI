package openai

import (
	"context"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
)

func newPinningTestContext(apiKey string) context.Context {
	recorder := httptest.NewRecorder()
	ginCtx, _ := gin.CreateTestContext(recorder)
	ginCtx.Set("apiKey", apiKey)
	return context.WithValue(context.Background(), "gin", ginCtx)
}

func resetPromptCachePinStore() {
	pinStore.mu.Lock()
	pinStore.pins = make(map[string]promptCachePin)
	pinStore.lastGC = time.Time{}
	pinStore.mu.Unlock()
}

func TestComputeFingerprint_IncludesModel(t *testing.T) {
	gin.SetMode(gin.TestMode)

	ctx := newPinningTestContext("client-key")
	rawA := []byte(`{"model":"gpt-5.3-codex","messages":[{"role":"user","content":"hello"}]}`)
	rawB := []byte(`{"model":"gpt-5.2","messages":[{"role":"user","content":"hello"}]}`)

	gotA := computeFingerprint(ctx, rawA)
	gotB := computeFingerprint(ctx, rawB)
	if gotA == "" || gotB == "" {
		t.Fatalf("expected non-empty fingerprints, got %q and %q", gotA, gotB)
	}
	if gotA == gotB {
		t.Fatalf("expected different fingerprints for different models, got %q", gotA)
	}
}

func TestScopedPromptCachePinKey_IncludesModel(t *testing.T) {
	rawA := []byte(`{"model":"gpt-5.3-codex","prompt_cache_key":"session-1"}`)
	rawB := []byte(`{"model":"gpt-5.2","prompt_cache_key":"session-1"}`)

	gotA := scopedPromptCachePinKey("session-1", rawA)
	gotB := scopedPromptCachePinKey("session-1", rawB)
	if gotA == "" || gotB == "" {
		t.Fatalf("expected non-empty scoped keys, got %q and %q", gotA, gotB)
	}
	if gotA == gotB {
		t.Fatalf("expected different scoped keys for different models, got %q", gotA)
	}
}

func TestPromptCachePinStore_UpdateAfterRequest_ExtendsTTLByDuration(t *testing.T) {
	resetPromptCachePinStore()
	t.Cleanup(resetPromptCachePinStore)

	key := "fp:test"
	authID := "auth1"

	pinStore.updateAfterRequest(key, authID, 60*time.Second)
	pinStore.mu.RLock()
	pinFast, ok := pinStore.pins[key]
	pinStore.mu.RUnlock()
	if !ok {
		t.Fatalf("expected fast request pin to be stored")
	}
	fastRemaining := time.Until(pinFast.expiresAt)
	if fastRemaining < 59*time.Minute || fastRemaining > 61*time.Minute {
		t.Fatalf("expected fast request TTL around 60 minutes, got %v", fastRemaining)
	}

	pinStore.updateAfterRequest(key, authID, 61*time.Second)
	pinStore.mu.RLock()
	pinLong, ok := pinStore.pins[key]
	pinStore.mu.RUnlock()
	if !ok {
		t.Fatalf("expected medium request pin to be stored")
	}
	longRemaining := time.Until(pinLong.expiresAt)
	if longRemaining < 29*time.Minute || longRemaining > 31*time.Minute {
		t.Fatalf("expected medium request TTL around 30 minutes, got %v", longRemaining)
	}
}

func TestPromptCachePinStore_UpdateAfterRequest_DeletesSlowPins(t *testing.T) {
	resetPromptCachePinStore()
	t.Cleanup(resetPromptCachePinStore)

	key := "fp:test"
	authID := "auth1"
	pinStore.set(key, authID)

	pinStore.updateAfterRequest(key, authID, 3*time.Minute+time.Second)
	if _, ok := pinStore.get(key); ok {
		t.Fatalf("expected slow request pin to be deleted")
	}
}
