// Package config provides configuration management for the CLI Proxy API server.
// It handles loading and parsing YAML configuration files, and provides structured
// access to application settings including server port, authentication directory,
// debug settings, proxy configuration, and API keys.
package config

// SDKConfig represents the application's configuration, loaded from a YAML file.
type SDKConfig struct {
	// ProxyURL is the URL of an optional proxy server to use for outbound requests.
	ProxyURL string `yaml:"proxy-url" json:"proxy-url"`

	// EnableGeminiCLIEndpoint controls whether Gemini CLI internal endpoints (/v1internal:*) are enabled.
	// Default is false for safety; when false, /v1internal:* requests are rejected.
	EnableGeminiCLIEndpoint bool `yaml:"enable-gemini-cli-endpoint" json:"enable-gemini-cli-endpoint"`

	// ForceModelPrefix requires explicit model prefixes (e.g., "teamA/gemini-3-pro-preview")
	// to target prefixed credentials. When false, unprefixed model requests may use prefixed
	// credentials as well.
	ForceModelPrefix bool `yaml:"force-model-prefix" json:"force-model-prefix"`

	// RequestLog enables or disables detailed request logging functionality.
	RequestLog bool `yaml:"request-log" json:"request-log"`

	// APIKeys is a list of keys for authenticating clients to this proxy server.
	APIKeys []string `yaml:"api-keys" json:"api-keys"`

	// PassthroughHeaders controls whether upstream response headers are forwarded to downstream clients.
	// Default is false (disabled).
	PassthroughHeaders bool `yaml:"passthrough-headers" json:"passthrough-headers"`

	// Streaming configures server-side streaming behavior (keep-alives and safe bootstrap retries).
	Streaming StreamingConfig `yaml:"streaming" json:"streaming"`

	// NonStreamKeepAliveInterval controls how often blank lines are emitted for non-streaming responses.
	// <= 0 disables keep-alives. Value is in seconds.
	NonStreamKeepAliveInterval int `yaml:"nonstream-keepalive-interval,omitempty" json:"nonstream-keepalive-interval,omitempty"`
}

// StreamingConfig holds server streaming behavior configuration.
type StreamingConfig struct {
	// KeepAliveSeconds controls how often the server emits SSE heartbeats (": keep-alive\n\n").
	// <= 0 disables keep-alives. Default is 0.
	KeepAliveSeconds int `yaml:"keepalive-seconds,omitempty" json:"keepalive-seconds,omitempty"`

	// BootstrapRetries controls how many times the server may retry a streaming request before any bytes are sent,
	// to allow auth rotation / transient recovery.
	// <= 0 disables bootstrap retries. Default is 0.
	BootstrapRetries int `yaml:"bootstrap-retries,omitempty" json:"bootstrap-retries,omitempty"`

	// StallTimeoutSeconds sets the maximum time (in seconds) to wait for the first SSE payload
	// during stream bootstrap. If no payload arrives within this duration, the credential is
	// temporarily marked unavailable for 30 minutes.
	// <= 0 disables stall detection. Default is 0 (disabled).
	StallTimeoutSeconds int `yaml:"stall-timeout-seconds,omitempty" json:"stall-timeout-seconds,omitempty"`

	// StreamIdleTimeoutSeconds sets the maximum idle time (in seconds) during active streaming.
	// If no upstream data is received within this duration after the last chunk, the connection
	// is closed. This prevents requests from hanging indefinitely when upstream stops responding
	// mid-stream. <= 0 disables idle timeout. Default is 0 (disabled).
	StreamIdleTimeoutSeconds int `yaml:"stream-idle-timeout-seconds,omitempty" json:"stream-idle-timeout-seconds,omitempty"`

	// CodexAssembleOutput works around an upstream Codex API change where the
	// response.completed SSE event no longer contains the output array.
	// When true, the non-streaming Codex executor collects output items from
	// intermediate SSE events (response.output_item.done) and injects them
	// into the response.completed payload before translation.
	// Only affects requests originating from OpenAI chat-completions clients.
	// Default is false (disabled).
	CodexAssembleOutput bool `yaml:"codex-assemble-output,omitempty" json:"codex-assemble-output,omitempty"`
}
