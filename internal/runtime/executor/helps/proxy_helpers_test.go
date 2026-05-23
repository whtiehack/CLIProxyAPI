package helps

import (
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
	sdkconfig "github.com/router-for-me/CLIProxyAPI/v7/sdk/config"
)

func TestNewProxyAwareHTTPClientDirectBypassesGlobalProxy(t *testing.T) {
	t.Parallel()

	client := NewProxyAwareHTTPClient(
		context.Background(),
		&config.Config{SDKConfig: sdkconfig.SDKConfig{ProxyURL: "http://global-proxy.example.com:8080"}},
		&cliproxyauth.Auth{ProxyURL: "direct"},
		0,
	)

	transport, ok := client.Transport.(*http.Transport)
	if !ok {
		t.Fatalf("transport type = %T, want *http.Transport", client.Transport)
	}
	if transport.Proxy != nil {
		t.Fatal("expected direct transport to disable proxy function")
	}
}

// slowHeaderRoundTripper blocks until the request context is cancelled,
// simulating an upstream that never returns response headers.
type slowHeaderRoundTripper struct{}

func (slowHeaderRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	<-req.Context().Done()
	return nil, req.Context().Err()
}

// instantRoundTripper returns a canned response immediately.
type instantRoundTripper struct{}

func (instantRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	return &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader("data: hello\n")),
		Request:    req,
	}, nil
}

func TestApplyResponseHeaderTimeoutWrapsCustomRoundTripper(t *testing.T) {
	t.Parallel()

	client := &http.Client{Transport: slowHeaderRoundTripper{}}
	ApplyResponseHeaderTimeout(client, 50*time.Millisecond)

	req, err := http.NewRequest(http.MethodGet, "http://example.invalid/", nil)
	if err != nil {
		t.Fatal(err)
	}
	_, err = client.Do(req)
	if err == nil {
		t.Fatal("expected timeout error, got nil")
	}
	var netErr net.Error
	if !errors.As(err, &netErr) || !netErr.Timeout() {
		t.Fatalf("expected net.Error with Timeout()==true, got %T: %v", err, err)
	}
}

func TestApplyResponseHeaderTimeoutKeepsBodyReadableAfterHeaders(t *testing.T) {
	t.Parallel()

	client := &http.Client{Transport: instantRoundTripper{}}
	ApplyResponseHeaderTimeout(client, 20*time.Millisecond)

	req, err := http.NewRequest(http.MethodGet, "http://example.invalid/", nil)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Wait past the header timeout to ensure the guard does not cancel the
	// context after headers have been received.
	time.Sleep(60 * time.Millisecond)
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("body read failed after header timeout window: %v", err)
	}
	if errClose := resp.Body.Close(); errClose != nil {
		t.Fatalf("body close failed: %v", errClose)
	}
	if string(body) != "data: hello\n" {
		t.Fatalf("unexpected body: %q", string(body))
	}
}
