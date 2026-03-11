package openai

import (
	"context"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
)

func newPinningTestContext(apiKey string) context.Context {
	recorder := httptest.NewRecorder()
	ginCtx, _ := gin.CreateTestContext(recorder)
	ginCtx.Set("apiKey", apiKey)
	return context.WithValue(context.Background(), "gin", ginCtx)
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
