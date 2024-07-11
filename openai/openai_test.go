package openai

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/stretchr/testify/assert"
)

func TestMiddlewareRequests(t *testing.T) {
	type testCase struct {
		Name     string
		Method   string
		Path     string
		Handler  func() gin.HandlerFunc
		Setup    func(t *testing.T, req *http.Request)
		Expected func(t *testing.T, req *http.Request)
	}

	var capturedRequest *http.Request

	captureRequestMiddleware := func() gin.HandlerFunc {
		return func(c *gin.Context) {
			bodyBytes, _ := io.ReadAll(c.Request.Body)
			c.Request.Body = io.NopCloser(bytes.NewReader(bodyBytes))
			capturedRequest = c.Request
			c.Next()
		}
	}

	testCases := []testCase{
		{
			Name:    "chat handler",
			Method:  http.MethodPost,
			Path:    "/api/chat",
			Handler: ChatMiddleware,
			Setup: func(t *testing.T, req *http.Request) {
				body := ChatCompletionRequest{
					Model:    "test-model",
					Messages: []Message{{Role: "user", Content: "Hello"}},
				}

				bodyBytes, _ := json.Marshal(body)

				req.Body = io.NopCloser(bytes.NewReader(bodyBytes))
				req.Header.Set("Content-Type", "application/json")
			},
			Expected: func(t *testing.T, req *http.Request) {
				var chatReq api.ChatRequest
				if err := json.NewDecoder(req.Body).Decode(&chatReq); err != nil {
					t.Fatal(err)
				}

				if chatReq.Messages[0].Role != "user" {
					t.Fatalf("expected 'user', got %s", chatReq.Messages[0].Role)
				}

				if chatReq.Messages[0].Content != "Hello" {
					t.Fatalf("expected 'Hello', got %s", chatReq.Messages[0].Content)
				}
			},
		},
		{
			Name:    "completions handler",
			Method:  http.MethodPost,
			Path:    "/api/generate",
			Handler: CompletionsMiddleware,
			Setup: func(t *testing.T, req *http.Request) {
				temp := float32(0.8)
				body := CompletionRequest{
					Model:       "test-model",
					Prompt:      "Hello",
					Temperature: &temp,
					Stop:        []string{"\n", "stop"},
				}

				bodyBytes, _ := json.Marshal(body)

				req.Body = io.NopCloser(bytes.NewReader(bodyBytes))
				req.Header.Set("Content-Type", "application/json")
			},
			Expected: func(t *testing.T, req *http.Request) {
				var genReq api.GenerateRequest
				if err := json.NewDecoder(req.Body).Decode(&genReq); err != nil {
					t.Fatal(err)
				}

				if genReq.Prompt != "Hello" {
					t.Fatalf("expected 'Hello', got %s", genReq.Prompt)
				}

				if genReq.Options["temperature"] != 1.6 {
					t.Fatalf("expected 1.6, got %f", genReq.Options["temperature"])
				}

				stopTokens, ok := genReq.Options["stop"].([]any)

				if !ok {
					t.Fatalf("expected stop tokens to be a list")
				}

				if stopTokens[0] != "\n" || stopTokens[1] != "stop" {
					t.Fatalf("expected ['\\n', 'stop'], got %v", stopTokens)
				}
			},
		},
	}

	gin.SetMode(gin.TestMode)
	router := gin.New()

	endpoint := func(c *gin.Context) {
		c.Status(http.StatusOK)
	}

	for _, tc := range testCases {
		t.Run(tc.Name, func(t *testing.T) {
			router = gin.New()
			router.Use(captureRequestMiddleware())
			router.Use(tc.Handler())
			router.Handle(tc.Method, tc.Path, endpoint)
			req, _ := http.NewRequest(tc.Method, tc.Path, nil)

			if tc.Setup != nil {
				tc.Setup(t, req)
			}

			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, req)

			tc.Expected(t, capturedRequest)
		})
	}
}

func TestMiddlewareResponses(t *testing.T) {
	type testCase struct {
		Name     string
		Method   string
		Path     string
		TestPath string
		Handler  func() gin.HandlerFunc
		Endpoint func(c *gin.Context)
		Setup    func(t *testing.T, req *http.Request)
		Expected func(t *testing.T, resp *httptest.ResponseRecorder)
	}

	testCases := []testCase{
		{
			Name:     "completions handler error forwarding",
			Method:   http.MethodPost,
			Path:     "/api/generate",
			TestPath: "/api/generate",
			Handler:  CompletionsMiddleware,
			Endpoint: func(c *gin.Context) {
				c.JSON(http.StatusBadRequest, gin.H{"error": "invalid request"})
			},
			Setup: func(t *testing.T, req *http.Request) {
				body := CompletionRequest{
					Model:  "test-model",
					Prompt: "Hello",
				}

				bodyBytes, _ := json.Marshal(body)

				req.Body = io.NopCloser(bytes.NewReader(bodyBytes))
				req.Header.Set("Content-Type", "application/json")
			},
			Expected: func(t *testing.T, resp *httptest.ResponseRecorder) {
				if resp.Code != http.StatusBadRequest {
					t.Fatalf("expected 400, got %d", resp.Code)
				}

				if !strings.Contains(resp.Body.String(), `"invalid request"`) {
					t.Fatalf("error was not forwarded")
				}
			},
		},
		{
			Name:     "list handler",
			Method:   http.MethodGet,
			Path:     "/api/tags",
			TestPath: "/api/tags",
			Handler:  ListMiddleware,
			Endpoint: func(c *gin.Context) {
				c.JSON(http.StatusOK, api.ListResponse{
					Models: []api.ListModelResponse{
						{
							Name: "Test Model",
						},
					},
				})
			},
			Expected: func(t *testing.T, resp *httptest.ResponseRecorder) {
				assert.Equal(t, http.StatusOK, resp.Code)

				var listResp ListCompletion
				if err := json.NewDecoder(resp.Body).Decode(&listResp); err != nil {
					t.Fatal(err)
				}

				if listResp.Object != "list" {
					t.Fatalf("expected list, got %s", listResp.Object)
				}

				if len(listResp.Data) != 1 {
					t.Fatalf("expected 1, got %d", len(listResp.Data))
				}

				if listResp.Data[0].Id != "Test Model" {
					t.Fatalf("expected Test Model, got %s", listResp.Data[0].Id)
				}
			},
		},
		{
			Name:     "embedding handler (single embedding)",
			Method:   http.MethodPost,
			Path:     "/api/embeddings",
			TestPath: "/api/embeddings",
			Handler:  EmbedMiddleware,
			Endpoint: func(c *gin.Context) {
				c.JSON(http.StatusOK, api.EmbedResponse{
					Model:      "test-model",
					Embeddings: [][]float32{{0.1, 0.2, 0.3}},
				})
			},
			Setup: func(t *testing.T, req *http.Request) {
				body := EmbedRequest{
					Input: "Hello",
					Model: "test-model",
				}

				bodyBytes, _ := json.Marshal(body)

				req.Body = io.NopCloser(bytes.NewReader(bodyBytes))
				req.Header.Set("Content-Type", "application/json")
			},
			Expected: func(t *testing.T, resp *httptest.ResponseRecorder) {
				var embeddingResp EmbeddingList
				if err := json.NewDecoder(resp.Body).Decode(&embeddingResp); err != nil {
					t.Fatal(err)
				}

				if embeddingResp.Object != "list" {
					t.Fatalf("expected list, got %s", embeddingResp.Object)
				}

				if len(embeddingResp.Data) != 1 {
					t.Fatalf("expected 1 embedding, got %d", len(embeddingResp.Data))
				}

				if embeddingResp.Data[0].Object != "embedding" {
					t.Fatalf("expected embedding, got %s", embeddingResp.Data[0].Object)
				}

				if embeddingResp.Data[0].Embedding[0] != 0.1 {
					t.Fatalf("expected 0.1, got %f", embeddingResp.Data[0])
				}

				if embeddingResp.Model != "test-model" {
					t.Fatalf("expected test-model, got %s", embeddingResp.Model)
				}
			},
		},
		{
			Name:     "embedding handler (batch embedding)",
			Method:   http.MethodPost,
			Path:     "/api/embed",
			TestPath: "/api/embed",
			Handler:  EmbedMiddleware,
			Endpoint: func(c *gin.Context) {
				c.JSON(http.StatusOK, api.EmbedResponse{
					Model: "test-model",
					Embeddings: [][]float64{
						{0.1, 0.2, 0.3},
						{0.4, 0.5, 0.6},
						{0.7, 0.8, 0.9},
					},
				})
			},
			Setup: func(t *testing.T, req *http.Request) {
				body := EmbedRequest{
					Input: []string{"Hello", "World", "Ollama"},
					Model: "test-model",
				}

				bodyBytes, _ := json.Marshal(body)

				req.Body = io.NopCloser(bytes.NewReader(bodyBytes))
				req.Header.Set("Content-Type", "application/json")
			},
			Expected: func(t *testing.T, resp *httptest.ResponseRecorder) {
				var embeddingResp EmbeddingList
				if err := json.NewDecoder(resp.Body).Decode(&embeddingResp); err != nil {
					t.Fatal(err)
				}

				if embeddingResp.Object != "list" {
					t.Fatalf("expected list, got %s", embeddingResp.Object)
				}

				if len(embeddingResp.Data) != 3 {
					t.Fatalf("expected 3 embeddings, got %d", len(embeddingResp.Data))
				}

				if embeddingResp.Data[0].Object != "embedding" {
					t.Fatalf("expected embedding, got %s", embeddingResp.Data[0].Object)
				}

				if embeddingResp.Data[0].Embedding[0] != 0.1 {
					t.Fatalf("expected 0.1, got %f", embeddingResp.Data[0])
				}

				if embeddingResp.Model != "test-model" {
					t.Fatalf("expected test-model, got %s", embeddingResp.Model)
				}
			},
		},
		{
			Name:     "retrieve model",
			Method:   http.MethodGet,
			Path:     "/api/show/:model",
			TestPath: "/api/show/test-model",
			Handler:  RetrieveMiddleware,
			Endpoint: func(c *gin.Context) {
				c.JSON(http.StatusOK, api.ShowResponse{
					ModifiedAt: time.Date(2024, 6, 17, 13, 45, 0, 0, time.UTC),
				})
			},
			Expected: func(t *testing.T, resp *httptest.ResponseRecorder) {
				var retrieveResp Model
				if err := json.NewDecoder(resp.Body).Decode(&retrieveResp); err != nil {
					t.Fatal(err)
				}

				if retrieveResp.Object != "model" {
					t.Fatalf("Expected object to be model, got %s", retrieveResp.Object)
				}

				if retrieveResp.Id != "test-model" {
					t.Fatalf("Expected id to be test-model, got %s", retrieveResp.Id)
				}
			},
		},
	}

	gin.SetMode(gin.TestMode)
	router := gin.New()

	for _, tc := range testCases {
		t.Run(tc.Name, func(t *testing.T) {
			router = gin.New()
			router.Use(tc.Handler())
			router.Handle(tc.Method, tc.Path, tc.Endpoint)
			req, _ := http.NewRequest(tc.Method, tc.TestPath, nil)

			if tc.Setup != nil {
				tc.Setup(t, req)
			}

			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, req)

			tc.Expected(t, resp)
		})
	}
}
