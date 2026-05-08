package chat_completions

import (
	"context"
	"testing"

	"github.com/tidwall/gjson"
)

func TestConvertCodexResponseToOpenAI_StreamSetsModelFromResponseCreated(t *testing.T) {
	ctx := context.Background()
	var param any

	modelName := "gpt-5.3-codex"

	out := ConvertCodexResponseToOpenAI(ctx, modelName, nil, nil, []byte(`data: {"type":"response.created","response":{"id":"resp_123","created_at":1700000000,"model":"gpt-5.3-codex"}}`), &param)
	if len(out) != 0 {
		t.Fatalf("expected no output for response.created, got %d chunks", len(out))
	}

	out = ConvertCodexResponseToOpenAI(ctx, modelName, nil, nil, []byte(`data: {"type":"response.output_text.delta","delta":"hello"}`), &param)
	if len(out) != 1 {
		t.Fatalf("expected 1 chunk, got %d", len(out))
	}

	gotModel := gjson.GetBytes(out[0], "model").String()
	if gotModel != modelName {
		t.Fatalf("expected model %q, got %q", modelName, gotModel)
	}
}

func TestConvertCodexResponseToOpenAI_FirstChunkUsesRequestModelName(t *testing.T) {
	ctx := context.Background()
	var param any

	modelName := "gpt-5.3-codex"

	out := ConvertCodexResponseToOpenAI(ctx, modelName, nil, nil, []byte(`data: {"type":"response.output_text.delta","delta":"hello"}`), &param)
	if len(out) != 1 {
		t.Fatalf("expected 1 chunk, got %d", len(out))
	}

	gotModel := gjson.GetBytes(out[0], "model").String()
	if gotModel != modelName {
		t.Fatalf("expected model %q, got %q", modelName, gotModel)
	}
}

func TestConvertCodexResponseToOpenAI_ToolCallChunkOmitsNullContentFields(t *testing.T) {
	ctx := context.Background()
	var param any

	out := ConvertCodexResponseToOpenAI(ctx, "gpt-5.4", nil, nil, []byte(`data: {"type":"response.output_item.added","item":{"type":"function_call","call_id":"call_123","name":"websearch"}}`), &param)
	if len(out) != 1 {
		t.Fatalf("expected 1 chunk, got %d", len(out))
	}

	if gjson.GetBytes(out[0], "choices.0.delta.content").Exists() {
		t.Fatalf("expected content to be omitted, got %s", string(out[0]))
	}
	if gjson.GetBytes(out[0], "choices.0.delta.reasoning_content").Exists() {
		t.Fatalf("expected reasoning_content to be omitted, got %s", string(out[0]))
	}
	if !gjson.GetBytes(out[0], "choices.0.delta.tool_calls").Exists() {
		t.Fatalf("expected tool_calls to exist, got %s", string(out[0]))
	}
}

func TestConvertCodexResponseToOpenAI_ToolCallArgumentsDeltaOmitsNullContentFields(t *testing.T) {
	ctx := context.Background()
	var param any

	out := ConvertCodexResponseToOpenAI(ctx, "gpt-5.4", nil, nil, []byte(`data: {"type":"response.output_item.added","item":{"type":"function_call","call_id":"call_123","name":"websearch"}}`), &param)
	if len(out) != 1 {
		t.Fatalf("expected tool call announcement chunk, got %d", len(out))
	}

	out = ConvertCodexResponseToOpenAI(ctx, "gpt-5.4", nil, nil, []byte(`data: {"type":"response.function_call_arguments.delta","delta":"{\"query\":\"OpenAI\"}"}`), &param)
	if len(out) != 1 {
		t.Fatalf("expected 1 chunk, got %d", len(out))
	}

	if gjson.GetBytes(out[0], "choices.0.delta.content").Exists() {
		t.Fatalf("expected content to be omitted, got %s", string(out[0]))
	}
	if gjson.GetBytes(out[0], "choices.0.delta.reasoning_content").Exists() {
		t.Fatalf("expected reasoning_content to be omitted, got %s", string(out[0]))
	}
	if !gjson.GetBytes(out[0], "choices.0.delta.tool_calls.0.function.arguments").Exists() {
		t.Fatalf("expected tool call arguments delta to exist, got %s", string(out[0]))
	}
}

func TestConvertCodexResponseToOpenAI_StreamPartialImageEmitsDeltaImages(t *testing.T) {
	ctx := context.Background()
	var param any

	chunk := []byte(`data: {"type":"response.image_generation_call.partial_image","item_id":"ig_123","output_format":"png","partial_image_b64":"aGVsbG8=","partial_image_index":0}`)

	out := ConvertCodexResponseToOpenAI(ctx, "gpt-5.4", nil, nil, chunk, &param)
	if len(out) != 1 {
		t.Fatalf("expected 1 chunk, got %d", len(out))
	}

	gotURL := gjson.GetBytes(out[0], "choices.0.delta.images.0.image_url.url").String()
	if gotURL != "data:image/png;base64,aGVsbG8=" {
		t.Fatalf("expected image url %q, got %q; chunk=%s", "data:image/png;base64,aGVsbG8=", gotURL, string(out[0]))
	}

	out = ConvertCodexResponseToOpenAI(ctx, "gpt-5.4", nil, nil, chunk, &param)
	if len(out) != 0 {
		t.Fatalf("expected duplicate image chunk to be suppressed, got %d", len(out))
	}
}

func TestConvertCodexResponseToOpenAI_StreamImageGenerationCallDoneEmitsDeltaImages(t *testing.T) {
	ctx := context.Background()
	var param any

	out := ConvertCodexResponseToOpenAI(ctx, "gpt-5.4", nil, nil, []byte(`data: {"type":"response.image_generation_call.partial_image","item_id":"ig_123","output_format":"png","partial_image_b64":"aGVsbG8=","partial_image_index":0}`), &param)
	if len(out) != 1 {
		t.Fatalf("expected 1 chunk, got %d", len(out))
	}

	out = ConvertCodexResponseToOpenAI(ctx, "gpt-5.4", nil, nil, []byte(`data: {"type":"response.output_item.done","item":{"id":"ig_123","type":"image_generation_call","output_format":"png","result":"aGVsbG8="}}`), &param)
	if len(out) != 0 {
		t.Fatalf("expected output_item.done to be suppressed when identical to last partial image, got %d", len(out))
	}

	out = ConvertCodexResponseToOpenAI(ctx, "gpt-5.4", nil, nil, []byte(`data: {"type":"response.output_item.done","item":{"id":"ig_123","type":"image_generation_call","output_format":"jpeg","result":"Ymll"}}`), &param)
	if len(out) != 1 {
		t.Fatalf("expected 1 chunk, got %d", len(out))
	}

	gotURL := gjson.GetBytes(out[0], "choices.0.delta.images.0.image_url.url").String()
	if gotURL != "data:image/jpeg;base64,Ymll" {
		t.Fatalf("expected image url %q, got %q; chunk=%s", "data:image/jpeg;base64,Ymll", gotURL, string(out[0]))
	}
}

func TestConvertCodexResponseToOpenAI_NonStreamImageGenerationCallAddsMessageImages(t *testing.T) {
	ctx := context.Background()

	raw := []byte(`{"type":"response.completed","response":{"id":"resp_123","created_at":1700000000,"model":"gpt-5.4","status":"completed","usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2},"output":[{"type":"message","content":[{"type":"output_text","text":"ok"}]},{"type":"image_generation_call","output_format":"png","result":"aGVsbG8="}]}}`)
	out := ConvertCodexResponseToOpenAINonStream(ctx, "gpt-5.4", nil, nil, raw, nil)

	gotURL := gjson.GetBytes(out, "choices.0.message.images.0.image_url.url").String()
	if gotURL != "data:image/png;base64,aGVsbG8=" {
		t.Fatalf("expected image url %q, got %q; chunk=%s", "data:image/png;base64,aGVsbG8=", gotURL, string(out))
	}
}

// Stream: reasoning output_item.done (item.type="reasoning") emits a single
// chunk with delta.reasoning_details containing both encrypted token + summary.
func TestConvertCodexResponseToOpenAI_StreamReasoningOutputItemDoneEmitsReasoningDetails(t *testing.T) {
	ctx := context.Background()
	var param any

	chunk := []byte(`data: {"type":"response.output_item.done","item":{"id":"rs_1","type":"reasoning","encrypted_content":"ENC_TOKEN","summary":[{"type":"summary_text","text":"thought one"},{"type":"summary_text","text":"thought two"}]}}`)
	out := ConvertCodexResponseToOpenAI(ctx, "gpt-5.4", nil, nil, chunk, &param)
	if len(out) != 1 {
		t.Fatalf("expected 1 chunk, got %d", len(out))
	}

	details := gjson.GetBytes(out[0], "choices.0.delta.reasoning_details").Array()
	if len(details) != 3 {
		t.Fatalf("expected 3 reasoning details (1 encrypted + 2 summaries), got %d: %s", len(details), string(out[0]))
	}
	if got := details[0].Get("type").String(); got != "reasoning.encrypted" {
		t.Errorf("detail 0: expected reasoning.encrypted, got %q", got)
	}
	if got := details[0].Get("data").String(); got != "ENC_TOKEN" {
		t.Errorf("detail 0: expected data=ENC_TOKEN, got %q", got)
	}
	if got := details[0].Get("format").String(); got != "openai-responses-v1" {
		t.Errorf("detail 0: expected format=openai-responses-v1, got %q", got)
	}
	if got := details[0].Get("id").String(); got != "rs_1" {
		t.Errorf("detail 0: expected id=rs_1, got %q", got)
	}
	if got := details[1].Get("type").String(); got != "reasoning.summary" {
		t.Errorf("detail 1: expected reasoning.summary, got %q", got)
	}
	if got := details[1].Get("summary").String(); got != "thought one" {
		t.Errorf("detail 1: expected summary=thought one, got %q", got)
	}
	if got := details[2].Get("summary").String(); got != "thought two" {
		t.Errorf("detail 2: expected summary=thought two, got %q", got)
	}
}

// Non-stream: codex output[type=reasoning] populates both reasoning_details
// (full structure) and reasoning_content (legacy first-summary fallback).
func TestConvertCodexResponseToOpenAINonStream_EmitsReasoningDetailsAndKeepsReasoningContent(t *testing.T) {
	ctx := context.Background()

	raw := []byte(`{"type":"response.completed","response":{"id":"resp_1","created_at":1700000000,"model":"gpt-5.4","status":"completed","output":[{"type":"reasoning","id":"rs_1","encrypted_content":"ENC_TOKEN","summary":[{"type":"summary_text","text":"first thought"},{"type":"summary_text","text":"second thought"}]},{"type":"message","content":[{"type":"output_text","text":"final answer"}]}]}}`)
	out := ConvertCodexResponseToOpenAINonStream(ctx, "gpt-5.4", nil, nil, raw, nil)

	if got := gjson.GetBytes(out, "choices.0.message.reasoning_content").String(); got != "first thought" {
		t.Errorf("expected reasoning_content=first thought, got %q", got)
	}
	if got := gjson.GetBytes(out, "choices.0.message.content").String(); got != "final answer" {
		t.Errorf("expected content=final answer, got %q", got)
	}

	details := gjson.GetBytes(out, "choices.0.message.reasoning_details").Array()
	if len(details) != 3 {
		t.Fatalf("expected 3 reasoning details, got %d: %s", len(details), string(out))
	}
	if details[0].Get("type").String() != "reasoning.encrypted" || details[0].Get("data").String() != "ENC_TOKEN" {
		t.Errorf("detail 0 wrong: %s", details[0].Raw)
	}
	if details[1].Get("type").String() != "reasoning.summary" || details[1].Get("summary").String() != "first thought" {
		t.Errorf("detail 1 wrong: %s", details[1].Raw)
	}
	if details[2].Get("type").String() != "reasoning.summary" || details[2].Get("summary").String() != "second thought" {
		t.Errorf("detail 2 wrong: %s", details[2].Raw)
	}
}

// Non-stream: reasoning item without encrypted_content still emits summary
// details and the legacy reasoning_content fallback.
func TestConvertCodexResponseToOpenAINonStream_ReasoningWithoutEncryptedStillEmitsSummary(t *testing.T) {
	ctx := context.Background()

	raw := []byte(`{"type":"response.completed","response":{"id":"resp_1","created_at":1700000000,"model":"gpt-5.4","status":"completed","output":[{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"thinking..."}]},{"type":"message","content":[{"type":"output_text","text":"hi"}]}]}}`)
	out := ConvertCodexResponseToOpenAINonStream(ctx, "gpt-5.4", nil, nil, raw, nil)

	details := gjson.GetBytes(out, "choices.0.message.reasoning_details").Array()
	if len(details) != 1 {
		t.Fatalf("expected 1 reasoning detail (summary only), got %d", len(details))
	}
	if details[0].Get("type").String() != "reasoning.summary" {
		t.Errorf("expected summary type, got %q", details[0].Get("type").String())
	}
	if got := gjson.GetBytes(out, "choices.0.message.reasoning_content").String(); got != "thinking..." {
		t.Errorf("expected reasoning_content fallback, got %q", got)
	}
}
