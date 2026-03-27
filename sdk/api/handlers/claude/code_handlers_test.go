package claude

import (
	"context"
	"net/http"
	"sync"
	"testing"
	"time"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/registry"
	"github.com/router-for-me/CLIProxyAPI/v6/sdk/api/handlers"
	coreauth "github.com/router-for-me/CLIProxyAPI/v6/sdk/cliproxy/auth"
	coreexecutor "github.com/router-for-me/CLIProxyAPI/v6/sdk/cliproxy/executor"
	sdkconfig "github.com/router-for-me/CLIProxyAPI/v6/sdk/config"
)

type claudeAuthAwareExecutor struct {
	mu      sync.Mutex
	authIDs []string
}

func (e *claudeAuthAwareExecutor) Identifier() string { return "codex" }

func (e *claudeAuthAwareExecutor) Execute(ctx context.Context, auth *coreauth.Auth, req coreexecutor.Request, opts coreexecutor.Options) (coreexecutor.Response, error) {
	e.mu.Lock()
	if auth != nil {
		e.authIDs = append(e.authIDs, auth.ID)
	} else {
		e.authIDs = append(e.authIDs, "")
	}
	e.mu.Unlock()
	return coreexecutor.Response{Payload: []byte(`{"ok":true}`)}, nil
}

func (e *claudeAuthAwareExecutor) ExecuteStream(context.Context, *coreauth.Auth, coreexecutor.Request, coreexecutor.Options) (*coreexecutor.StreamResult, error) {
	return nil, &coreauth.Error{Code: "not_implemented", Message: "ExecuteStream not implemented"}
}

func (e *claudeAuthAwareExecutor) Refresh(ctx context.Context, auth *coreauth.Auth) (*coreauth.Auth, error) {
	return auth, nil
}

func (e *claudeAuthAwareExecutor) CountTokens(context.Context, *coreauth.Auth, coreexecutor.Request, coreexecutor.Options) (coreexecutor.Response, error) {
	return coreexecutor.Response{}, &coreauth.Error{Code: "not_implemented", Message: "CountTokens not implemented"}
}

func (e *claudeAuthAwareExecutor) HttpRequest(context.Context, *coreauth.Auth, *http.Request) (*http.Response, error) {
	return nil, &coreauth.Error{Code: "not_implemented", Message: "HttpRequest not implemented", HTTPStatus: http.StatusNotImplemented}
}

func (e *claudeAuthAwareExecutor) AuthIDs() []string {
	e.mu.Lock()
	defer e.mu.Unlock()
	return append([]string(nil), e.authIDs...)
}

func resetClaudePinStore() {
	claudePinStore.mu.Lock()
	claudePinStore.pins = make(map[string]claudeAuthPin)
	claudePinStore.lastGC = time.Time{}
	claudePinStore.mu.Unlock()
}

func TestClaudeAuthPinKey_IncludesModelUserIDAndFirstMessage(t *testing.T) {
	rawA := []byte(`{"model":"gpt-5.4","metadata":{"user_id":"user-1"},"messages":[{"role":"user","content":"hello"}]}`)
	rawB := []byte(`{"model":"gpt-5.4","metadata":{"user_id":"user-1"},"messages":[{"role":"user","content":"world"}]}`)
	rawC := []byte(`{"model":"gpt-5.4-mini","metadata":{"user_id":"user-1"},"messages":[{"role":"user","content":"hello"}]}`)
	rawD := []byte(`{"model":"gpt-5.4","metadata":{"user_id":"user-2"},"messages":[{"role":"user","content":"hello"}]}`)

	keyA := claudeAuthPinKey(rawA, "gpt-5.4")
	keyB := claudeAuthPinKey(rawB, "gpt-5.4")
	keyC := claudeAuthPinKey(rawC, "gpt-5.4-mini")
	keyD := claudeAuthPinKey(rawD, "gpt-5.4")

	if keyA == "" || keyB == "" || keyC == "" || keyD == "" {
		t.Fatalf("expected non-empty keys, got %q %q %q %q", keyA, keyB, keyC, keyD)
	}
	if keyA == keyB {
		t.Fatalf("expected different keys for different first messages, got %q", keyA)
	}
	if keyA == keyC {
		t.Fatalf("expected different keys for different models, got %q", keyA)
	}
	if keyA == keyD {
		t.Fatalf("expected different keys for different user IDs, got %q", keyA)
	}
}

func TestClaudeAuthPinning_ReusesPinnedAuthForMatchingConversation(t *testing.T) {
	resetClaudePinStore()

	executor := &claudeAuthAwareExecutor{}
	manager := coreauth.NewManager(nil, nil, nil)
	manager.RegisterExecutor(executor)

	auth1 := &coreauth.Auth{ID: "auth1", Provider: "codex", Status: coreauth.StatusActive}
	auth2 := &coreauth.Auth{ID: "auth2", Provider: "codex", Status: coreauth.StatusActive}
	if _, err := manager.Register(context.Background(), auth1); err != nil {
		t.Fatalf("manager.Register(auth1): %v", err)
	}
	if _, err := manager.Register(context.Background(), auth2); err != nil {
		t.Fatalf("manager.Register(auth2): %v", err)
	}

	registry.GetGlobalRegistry().RegisterClient(auth1.ID, auth1.Provider, []*registry.ModelInfo{{ID: "gpt-5.4"}})
	registry.GetGlobalRegistry().RegisterClient(auth2.ID, auth2.Provider, []*registry.ModelInfo{{ID: "gpt-5.4"}})
	t.Cleanup(func() {
		registry.GetGlobalRegistry().UnregisterClient(auth1.ID)
		registry.GetGlobalRegistry().UnregisterClient(auth2.ID)
		resetClaudePinStore()
	})

	base := handlers.NewBaseAPIHandlers(&sdkconfig.SDKConfig{}, manager)
	ctx := context.Background()
	raw := []byte(`{"model":"gpt-5.4","metadata":{"user_id":"user-1"},"messages":[{"role":"user","content":"hello"}]}`)
	ctx = applyClaudeAuthPinning(ctx, raw, "gpt-5.4")

	resp, _, errMsg := base.ExecuteWithAuthManager(ctx, "claude", "gpt-5.4", raw, "")
	if errMsg != nil {
		t.Fatalf("first ExecuteWithAuthManager() error = %+v", errMsg)
	}
	if string(resp) != `{"ok":true}` {
		t.Fatalf("unexpected response payload %q", string(resp))
	}
	updateClaudePinAfterRequest(ctx, time.Second)

	ctx2 := context.Background()
	ctx2 = applyClaudeAuthPinning(ctx2, raw, "gpt-5.4")
	resp, _, errMsg = base.ExecuteWithAuthManager(ctx2, "claude", "gpt-5.4", raw, "")
	if errMsg != nil {
		t.Fatalf("second ExecuteWithAuthManager() error = %+v", errMsg)
	}
	if string(resp) != `{"ok":true}` {
		t.Fatalf("unexpected response payload %q", string(resp))
	}

	authIDs := executor.AuthIDs()
	if len(authIDs) != 2 {
		t.Fatalf("expected 2 upstream attempts, got %v", authIDs)
	}
	if authIDs[0] == "" || authIDs[1] == "" {
		t.Fatalf("expected auth IDs to be recorded, got %v", authIDs)
	}
	if authIDs[0] != authIDs[1] {
		t.Fatalf("expected second request to reuse pinned auth, got %v", authIDs)
	}
}
