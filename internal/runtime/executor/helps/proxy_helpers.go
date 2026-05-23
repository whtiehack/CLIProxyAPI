package helps

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
	"github.com/router-for-me/CLIProxyAPI/v7/sdk/proxyutil"
	log "github.com/sirupsen/logrus"
)

// NewProxyAwareHTTPClient creates an HTTP client with proper proxy configuration priority:
// 1. Use auth.ProxyURL if configured (highest priority)
// 2. Use cfg.ProxyURL if auth proxy is not configured
// 3. Use RoundTripper from context if neither are configured
//
// Parameters:
//   - ctx: The context containing optional RoundTripper
//   - cfg: The application configuration
//   - auth: The authentication information
//   - timeout: The client timeout (0 means no timeout)
//
// Returns:
//   - *http.Client: An HTTP client with configured proxy or transport
func NewProxyAwareHTTPClient(ctx context.Context, cfg *config.Config, auth *cliproxyauth.Auth, timeout time.Duration) *http.Client {
	httpClient := &http.Client{}
	if timeout > 0 {
		httpClient.Timeout = timeout
	}

	// Priority 1: Use auth.ProxyURL if configured
	var proxyURL string
	if auth != nil {
		proxyURL = strings.TrimSpace(auth.ProxyURL)
	}

	// Priority 2: Use cfg.ProxyURL if auth proxy is not configured
	if proxyURL == "" && cfg != nil {
		proxyURL = strings.TrimSpace(cfg.ProxyURL)
	}

	// If we have a proxy URL configured, set up the transport
	if proxyURL != "" {
		transport := buildProxyTransport(proxyURL)
		if transport != nil {
			httpClient.Transport = transport
			return httpClient
		}
		// If proxy setup failed, log and fall through to context RoundTripper
		log.Debugf("failed to setup proxy from URL: %s, falling back to context transport", proxyutil.Redact(proxyURL))
	}

	// Priority 3: Use RoundTripper from context (typically from RoundTripperFor)
	if rt, ok := ctx.Value("cliproxy.roundtripper").(http.RoundTripper); ok && rt != nil {
		httpClient.Transport = rt
	}

	return httpClient
}

// ApplyResponseHeaderTimeout sets ResponseHeaderTimeout on the client's transport so
// httpClient.Do returns a net.Error timeout if upstream does not return response headers
// within the given duration. Only affects waiting for response headers; body reads are unaffected.
// Used together with StallTimeoutSeconds to catch upstreams that accept the connection
// but never produce a response.
func ApplyResponseHeaderTimeout(client *http.Client, timeout time.Duration) {
	if client == nil || timeout <= 0 {
		return
	}
	switch t := client.Transport.(type) {
	case *http.Transport:
		cloned := t.Clone()
		cloned.ResponseHeaderTimeout = timeout
		client.Transport = cloned
	case nil:
		cloned := http.DefaultTransport.(*http.Transport).Clone()
		cloned.ResponseHeaderTimeout = timeout
		client.Transport = cloned
	default:
		// Custom RoundTripper (e.g. the utls fallback transport): wrap it with a
		// context-based response-header timeout so the guard still applies.
		client.Transport = &responseHeaderTimeoutTransport{rt: t, timeout: timeout}
	}
}

// errResponseHeaderTimeout is returned when upstream does not deliver response
// headers within the configured window. It implements net.Error with
// Timeout() == true so the conductor classifies it as a stream stall, matching
// the semantics of http.Transport.ResponseHeaderTimeout.
var errResponseHeaderTimeout error = headerTimeoutError{}

type headerTimeoutError struct{}

func (headerTimeoutError) Error() string   { return "timeout awaiting response headers" }
func (headerTimeoutError) Timeout() bool   { return true }
func (headerTimeoutError) Temporary() bool { return true }

// responseHeaderTimeoutTransport enforces a response-header timeout for
// transports that are not *http.Transport. It cancels the request context if
// headers do not arrive in time; once headers are received the derived context
// stays alive until the response body is closed.
type responseHeaderTimeoutTransport struct {
	rt      http.RoundTripper
	timeout time.Duration
}

func (t *responseHeaderTimeoutTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	ctx, cancel := context.WithCancelCause(req.Context())
	timer := time.AfterFunc(t.timeout, func() { cancel(errResponseHeaderTimeout) })
	resp, err := t.rt.RoundTrip(req.WithContext(ctx))
	timer.Stop()
	if err != nil {
		isHeaderTimeout := errors.Is(context.Cause(ctx), errResponseHeaderTimeout)
		cancel(nil)
		if isHeaderTimeout {
			return nil, errResponseHeaderTimeout
		}
		return nil, err
	}
	// Cancelling the derived context now would abort body reads, so defer it
	// to body close.
	resp.Body = &cancelOnCloseBody{ReadCloser: resp.Body, cancel: func() { cancel(nil) }}
	return resp, nil
}

type cancelOnCloseBody struct {
	io.ReadCloser
	cancel func()
}

func (b *cancelOnCloseBody) Close() error {
	err := b.ReadCloser.Close()
	b.cancel()
	return err
}

// buildProxyTransport creates an HTTP transport configured for the given proxy URL.
// It supports SOCKS5, HTTP, and HTTPS proxy protocols.
//
// Parameters:
//   - proxyURL: The proxy URL string (e.g., "socks5://user:pass@host:port", "http://host:port")
//
// Returns:
//   - *http.Transport: A configured transport, or nil if the proxy URL is invalid
func buildProxyTransport(proxyURL string) *http.Transport {
	transport, _, errBuild := proxyutil.BuildHTTPTransport(proxyURL)
	if errBuild != nil {
		log.Errorf("%v", errBuild)
		return nil
	}
	return transport
}
