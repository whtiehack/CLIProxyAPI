package cliproxy

import (
	"context"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/registry"
	coreauth "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
)

func TestRegisterModelsForAuth_CodexBuiltinsRespectConfiguredModels(t *testing.T) {
	const (
		imageAuthID = "codex-config-with-image"
		textAuthID  = "codex-config-without-image"
		imageModel  = "gpt-image-2"
	)

	service := &Service{cfg: &config.Config{CodexKey: []config.CodexKey{
		{
			APIKey:  "image-key",
			BaseURL: "https://image.example.com",
			Models: []config.CodexModel{
				{Name: "gpt-5.5"},
				{Name: imageModel},
			},
		},
		{
			APIKey:  "text-key",
			BaseURL: "https://text.example.com",
			Models: []config.CodexModel{
				{Name: "gpt-5.5"},
			},
		},
	}}}

	modelRegistry := registry.GetGlobalRegistry()
	modelRegistry.UnregisterClient(imageAuthID)
	modelRegistry.UnregisterClient(textAuthID)
	t.Cleanup(func() {
		modelRegistry.UnregisterClient(imageAuthID)
		modelRegistry.UnregisterClient(textAuthID)
	})

	register := func(id, apiKey, baseURL string) {
		t.Helper()
		service.registerModelsForAuth(context.Background(), &coreauth.Auth{
			ID:       id,
			Provider: "codex",
			Status:   coreauth.StatusActive,
			Attributes: map[string]string{
				"api_key":   apiKey,
				"auth_kind": "api_key",
				"base_url":  baseURL,
			},
		})
	}

	register(imageAuthID, "image-key", "https://image.example.com")
	register(textAuthID, "text-key", "https://text.example.com")

	if !modelRegistry.ClientSupportsModel(imageAuthID, imageModel) {
		t.Fatalf("image-enabled Codex provider does not support %s", imageModel)
	}
	if modelRegistry.ClientSupportsModel(textAuthID, imageModel) {
		t.Fatalf("text-only Codex provider unexpectedly supports %s", imageModel)
	}
}
