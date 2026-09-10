package auth

import (
	"context"
	"net/http"
	"testing"
	"time"
)

func newStallResultError() *Error {
	return &Error{
		Code:       ErrorCodeStreamStall,
		Message:    "upstream stream stalled: no response headers within 1m20s",
		HTTPStatus: http.StatusBadGateway,
	}
}

// OAuth credentials talk to first-party endpoints, so a stall is an upstream
// hiccup rather than a credential fault and must not take the account out of
// rotation.
func TestManager_MarkResult_StreamStallKeepsOAuthAuthAvailable(t *testing.T) {
	previous := quotaCooldownDisabled.Load()
	quotaCooldownDisabled.Store(false)
	t.Cleanup(func() { quotaCooldownDisabled.Store(previous) })

	m := NewManager(nil, nil, nil)
	auth := &Auth{
		ID:         "auth-oauth-stall",
		Provider:   "codex",
		Attributes: map[string]string{AttributeAuthKind: AuthKindOAuth},
	}
	if _, err := m.Register(context.Background(), auth); err != nil {
		t.Fatalf("register auth: %v", err)
	}

	model := "gpt-5.6-sol"
	m.MarkResult(context.Background(), Result{
		AuthID:   auth.ID,
		Provider: auth.Provider,
		Model:    model,
		Success:  false,
		Error:    newStallResultError(),
	})

	assertNoCooldown(t, m, auth.ID, model)

	updated, ok := m.GetByID(auth.ID)
	if !ok || updated == nil {
		t.Fatalf("expected auth to be present")
	}
	if blocked, reason, until := isAuthBlockedForModel(updated, model, time.Now()); blocked {
		t.Fatalf("expected oauth auth to stay schedulable after a stall, blocked by %v until %v", reason, until)
	}
	if state := updated.ModelStates[model]; state == nil || state.LastError == nil || state.LastError.Code != ErrorCodeStreamStall {
		t.Fatalf("expected the stall to still be recorded on the model state, got %#v", state)
	}
}

// API-key relays give no other health signal, so the 30 minute suspension stays.
func TestManager_MarkResult_StreamStallSuspendsAPIKeyAuth(t *testing.T) {
	previous := quotaCooldownDisabled.Load()
	quotaCooldownDisabled.Store(false)
	t.Cleanup(func() { quotaCooldownDisabled.Store(previous) })

	m := NewManager(nil, nil, nil)
	auth := &Auth{
		ID:       "auth-apikey-stall",
		Provider: "codex",
		Attributes: map[string]string{
			AttributeAuthKind: AuthKindAPIKey,
			AttributeAPIKey:   "sk-relay",
		},
	}
	if _, err := m.Register(context.Background(), auth); err != nil {
		t.Fatalf("register auth: %v", err)
	}

	model := "gpt-5.6-sol"
	before := time.Now()
	m.MarkResult(context.Background(), Result{
		AuthID:   auth.ID,
		Provider: auth.Provider,
		Model:    model,
		Success:  false,
		Error:    newStallResultError(),
	})

	updated, ok := m.GetByID(auth.ID)
	if !ok || updated == nil {
		t.Fatalf("expected auth to be present")
	}
	state := updated.ModelStates[model]
	if state == nil {
		t.Fatalf("expected a model state for %s", model)
	}
	if !state.Unavailable {
		t.Fatalf("expected api-key model state to be unavailable after a stall")
	}
	if got := state.NextRetryAfter.Sub(before); got < 29*time.Minute || got > 31*time.Minute {
		t.Fatalf("expected ~30m cooldown, got %v (next retry %v)", got, state.NextRetryAfter)
	}
	if blocked, _, _ := isAuthBlockedForModel(updated, model, time.Now()); !blocked {
		t.Fatalf("expected api-key auth to be blocked for %s after a stall", model)
	}
}

// Cooling overrides must not resurrect a stalled relay, which is why the stall
// branch bypasses disableCooling for api-key credentials.
func TestManager_MarkResult_StreamStallSuspendsAPIKeyAuthEvenWhenCoolingDisabled(t *testing.T) {
	previous := quotaCooldownDisabled.Load()
	quotaCooldownDisabled.Store(true)
	t.Cleanup(func() { quotaCooldownDisabled.Store(previous) })

	m := NewManager(nil, nil, nil)
	auth := &Auth{
		ID:       "auth-apikey-stall-nocool",
		Provider: "codex",
		Attributes: map[string]string{
			AttributeAuthKind: AuthKindAPIKey,
			AttributeAPIKey:   "sk-relay",
		},
	}
	if _, err := m.Register(context.Background(), auth); err != nil {
		t.Fatalf("register auth: %v", err)
	}

	model := "gpt-5.6-sol"
	m.MarkResult(context.Background(), Result{
		AuthID:   auth.ID,
		Provider: auth.Provider,
		Model:    model,
		Success:  false,
		Error:    newStallResultError(),
	})

	updated, ok := m.GetByID(auth.ID)
	if !ok || updated == nil {
		t.Fatalf("expected auth to be present")
	}
	if state := updated.ModelStates[model]; state == nil || !state.Unavailable || state.NextRetryAfter.IsZero() {
		t.Fatalf("expected stalled api-key credential to stay suspended, got %#v", state)
	}
}

// A stalled OAuth credential must recover its own model state, not the whole auth,
// so sibling models keep whatever cooldown they already had.
func TestManager_MarkResult_StreamStallOAuthLeavesSiblingModelCooldown(t *testing.T) {
	previous := quotaCooldownDisabled.Load()
	quotaCooldownDisabled.Store(false)
	t.Cleanup(func() { quotaCooldownDisabled.Store(previous) })

	m := NewManager(nil, nil, nil)
	auth := &Auth{
		ID:         "auth-oauth-stall-sibling",
		Provider:   "codex",
		Attributes: map[string]string{AttributeAuthKind: AuthKindOAuth},
	}
	if _, err := m.Register(context.Background(), auth); err != nil {
		t.Fatalf("register auth: %v", err)
	}

	sibling := "gpt-5.6-luna"
	m.MarkResult(context.Background(), Result{
		AuthID:   auth.ID,
		Provider: auth.Provider,
		Model:    sibling,
		Success:  false,
		Error:    &Error{Message: "forbidden", HTTPStatus: http.StatusForbidden},
	})

	stalled := "gpt-5.6-sol"
	m.MarkResult(context.Background(), Result{
		AuthID:   auth.ID,
		Provider: auth.Provider,
		Model:    stalled,
		Success:  false,
		Error:    newStallResultError(),
	})

	updated, ok := m.GetByID(auth.ID)
	if !ok || updated == nil {
		t.Fatalf("expected auth to be present")
	}
	if state := updated.ModelStates[sibling]; state == nil || state.NextRetryAfter.IsZero() {
		t.Fatalf("expected the 403 cooldown on %s to survive, got %#v", sibling, state)
	}
	if blocked, reason, until := isAuthBlockedForModel(updated, stalled, time.Now()); blocked {
		t.Fatalf("expected %s to stay schedulable, blocked by %v until %v", stalled, reason, until)
	}
}
