package helps

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
)

func streamGuardConfig(stall, idle int) *config.Config {
	cfg := &config.Config{}
	cfg.Streaming.StallTimeoutSeconds = stall
	cfg.Streaming.StreamIdleTimeoutSeconds = idle
	return cfg
}

func assertStreamStall(t *testing.T, err error) {
	t.Helper()
	var authErr *cliproxyauth.Error
	if !errors.As(err, &authErr) || authErr.Code != cliproxyauth.ErrorCodeStreamStall {
		t.Fatalf("expected stream_stall error, got %v", err)
	}
	if authErr.HTTPStatus != http.StatusBadGateway {
		t.Fatalf("expected 502, got %d", authErr.HTTPStatus)
	}
}

func TestApplyStreamGuard_DisabledLeavesClientUntouched(t *testing.T) {
	client := &http.Client{}
	if got := ApplyStreamGuard(client, streamGuardConfig(0, 0)); got.Transport != nil {
		t.Fatalf("expected transport untouched, got %T", got.Transport)
	}
}

func TestApplyStreamGuard_ResponseHeaderStall(t *testing.T) {
	release := make(chan struct{})
	defer close(release)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-release:
		case <-r.Context().Done():
		}
	}))
	defer srv.Close()

	client := ApplyStreamGuard(srv.Client(), streamGuardConfig(1, 0))
	req, _ := http.NewRequestWithContext(context.Background(), http.MethodGet, srv.URL, nil)
	start := time.Now()
	_, err := client.Do(req)
	assertStreamStall(t, err)
	if elapsed := time.Since(start); elapsed > 5*time.Second {
		t.Fatalf("stall took too long: %v", elapsed)
	}
}

func TestApplyStreamGuard_IdleStallAfterFirstChunk(t *testing.T) {
	release := make(chan struct{})
	defer close(release)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, "data: hello\n\n")
		w.(http.Flusher).Flush()
		select {
		case <-release:
		case <-r.Context().Done():
		}
	}))
	defer srv.Close()

	client := ApplyStreamGuard(srv.Client(), streamGuardConfig(0, 1))
	req, _ := http.NewRequestWithContext(context.Background(), http.MethodGet, srv.URL, nil)
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("unexpected request error: %v", err)
	}
	defer resp.Body.Close()

	buf := make([]byte, 64)
	n, err := resp.Body.Read(buf)
	if err != nil || string(buf[:n]) != "data: hello\n\n" {
		t.Fatalf("expected first chunk, got %q err=%v", buf[:n], err)
	}
	_, err = resp.Body.Read(buf)
	assertStreamStall(t, err)
}

func TestApplyStreamGuard_HealthyStreamPassesThrough(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, "ok")
	}))
	defer srv.Close()

	client := ApplyStreamGuard(srv.Client(), streamGuardConfig(1, 1))
	req, _ := http.NewRequestWithContext(context.Background(), http.MethodGet, srv.URL, nil)
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("unexpected request error: %v", err)
	}
	body, err := io.ReadAll(resp.Body)
	_ = resp.Body.Close()
	if err != nil || string(body) != "ok" {
		t.Fatalf("expected body ok, got %q err=%v", body, err)
	}
}
