package helps

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
	log "github.com/sirupsen/logrus"
)

func newStreamStallError(phase string, timeout time.Duration) *cliproxyauth.Error {
	return &cliproxyauth.Error{
		Code:       cliproxyauth.ErrorCodeStreamStall,
		Message:    fmt.Sprintf("upstream stream stalled: no %s within %v", phase, timeout),
		HTTPStatus: http.StatusBadGateway,
	}
}

// ApplyStreamGuard wraps the client's transport with the streaming stall guards from
// cfg.Streaming: stall-timeout-seconds bounds the wait for response headers and
// stream-idle-timeout-seconds bounds the gap between body reads. Both surface as a
// stream_stall *cliproxyauth.Error so the conductor can rotate credentials. Zero
// values leave the client untouched.
func ApplyStreamGuard(client *http.Client, cfg *config.Config) *http.Client {
	if client == nil || cfg == nil {
		return client
	}
	header := time.Duration(cfg.Streaming.StallTimeoutSeconds) * time.Second
	idle := time.Duration(cfg.Streaming.StreamIdleTimeoutSeconds) * time.Second
	if header <= 0 && idle <= 0 {
		return client
	}
	rt := client.Transport
	if rt == nil {
		rt = http.DefaultTransport
	}
	client.Transport = &streamGuardTransport{rt: rt, header: header, idle: idle}
	return client
}

type streamGuardTransport struct {
	rt     http.RoundTripper
	header time.Duration
	idle   time.Duration
}

func (t *streamGuardTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	ctx, cancel := context.WithCancelCause(req.Context())
	var headerTimer *time.Timer
	if t.header > 0 {
		stall := newStreamStallError("response headers", t.header)
		headerTimer = time.AfterFunc(t.header, func() {
			log.Warnf("[stream_stall] %s %s: %s", req.Method, req.URL.Host, stall.Message)
			cancel(stall)
		})
	}
	resp, err := t.rt.RoundTrip(req.WithContext(ctx))
	if headerTimer != nil {
		headerTimer.Stop()
	}
	if err != nil {
		var stall *cliproxyauth.Error
		if errors.As(context.Cause(ctx), &stall) && stall.Code == cliproxyauth.ErrorCodeStreamStall {
			err = stall
		}
		cancel(nil)
		return nil, err
	}
	// Cancelling now would abort body reads, so the derived context lives until Close.
	body := &idleGuardBody{ReadCloser: resp.Body, idle: t.idle, host: req.URL.Host, cancel: func() { cancel(nil) }}
	if t.idle > 0 {
		body.timer = time.AfterFunc(t.idle, body.onIdle)
	}
	resp.Body = body
	return resp, nil
}

// idleGuardBody closes the upstream body when no bytes arrive for idle, so a blocked
// scanner unwinds with a stream_stall error instead of hanging until the client leaves.
type idleGuardBody struct {
	io.ReadCloser
	idle    time.Duration
	host    string
	timer   *time.Timer
	cancel  func()
	stalled atomic.Bool
	once    sync.Once
}

func (b *idleGuardBody) onIdle() {
	b.stalled.Store(true)
	log.Warnf("[stream_idle_timeout] %s: closing upstream connection after %v idle", b.host, b.idle)
	_ = b.ReadCloser.Close()
}

func (b *idleGuardBody) Read(p []byte) (int, error) {
	n, err := b.ReadCloser.Read(p)
	if b.stalled.Load() {
		return n, newStreamStallError("stream data", b.idle)
	}
	if err == nil && b.timer != nil {
		b.timer.Reset(b.idle)
	}
	return n, err
}

func (b *idleGuardBody) Close() error {
	var err error
	b.once.Do(func() {
		if b.timer != nil {
			b.timer.Stop()
		}
		err = b.ReadCloser.Close()
		b.cancel()
	})
	return err
}
