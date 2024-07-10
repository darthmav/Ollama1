package server

import (
	"archive/zip"
	"bytes"
	"errors"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/template"
)

func createZipFile(t *testing.T, name string) *os.File {
	t.Helper()

	f, err := os.CreateTemp(t.TempDir(), "")
	if err != nil {
		t.Fatal(err)
	}

	zf := zip.NewWriter(f)
	defer zf.Close()

	zh, err := zf.CreateHeader(&zip.FileHeader{Name: name})
	if err != nil {
		t.Fatal(err)
	}

	if _, err := io.Copy(zh, bytes.NewReader([]byte(""))); err != nil {
		t.Fatal(err)
	}

	return f
}

func TestExtractFromZipFile(t *testing.T) {
	cases := []struct {
		name   string
		expect []string
		err    error
	}{
		{
			name:   "good",
			expect: []string{"good"},
		},
		{
			name:   strings.Join([]string{"path", "..", "to", "good"}, string(os.PathSeparator)),
			expect: []string{filepath.Join("to", "good")},
		},
		{
			name:   strings.Join([]string{"path", "..", "to", "..", "good"}, string(os.PathSeparator)),
			expect: []string{"good"},
		},
		{
			name:   strings.Join([]string{"path", "to", "..", "..", "good"}, string(os.PathSeparator)),
			expect: []string{"good"},
		},
		{
			name: strings.Join([]string{"..", "..", "..", "..", "..", "..", "..", "..", "..", "..", "..", "..", "..", "..", "..", "..", "bad"}, string(os.PathSeparator)),
			err:  zip.ErrInsecurePath,
		},
		{
			name: strings.Join([]string{"path", "..", "..", "to", "bad"}, string(os.PathSeparator)),
			err:  zip.ErrInsecurePath,
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			f := createZipFile(t, tt.name)
			defer f.Close()

			tempDir := t.TempDir()
			if err := extractFromZipFile(tempDir, f, func(api.ProgressResponse) {}); !errors.Is(err, tt.err) {
				t.Fatal(err)
			}

			var matches []string
			if err := filepath.Walk(tempDir, func(p string, fi os.FileInfo, err error) error {
				if err != nil {
					return err
				}

				if !fi.IsDir() {
					matches = append(matches, p)
				}

				return nil
			}); err != nil {
				t.Fatal(err)
			}

			var actual []string
			for _, match := range matches {
				rel, err := filepath.Rel(tempDir, match)
				if err != nil {
					t.Error(err)
				}

				actual = append(actual, rel)
			}

			if !slices.Equal(actual, tt.expect) {
				t.Fatalf("expected %d files, got %d", len(tt.expect), len(matches))
			}
		})
	}
}

func must[T any](t T, err error) T {
	if err != nil {
		panic(err)
	}

	return t
}

type function struct {
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments"`
}

func TestParseToolCalls(t *testing.T) {
	cases := []struct {
		name string
		tmpl *template.Template
		s    string
		ok   bool
	}{
		{
			name: "no tools",
			tmpl: template.DefaultTemplate,
			ok:   false,
		},
		{
			name: "mistral",
			tmpl: must(template.Parse(`{{- if .Messages }}
{{- range $index, $_ := .Messages }}
{{- if eq .Role "user" }}
{{- if and (eq (len (slice $.Messages $index)) 1) $.Tools }}[AVAILABLE_TOOLS] {{ json $.Tools }}[/AVAILABLE_TOOLS]
{{- end }}[INST] {{ if and (eq (len (slice $.Messages $index)) 1) $.System }}{{ $.System }}{{ printf "\n\n" }}
{{- end }}{{ .Content }}[/INST]
{{- else if eq .Role "assistant" }}
{{- if .Content }} {{ .Content }}</s>
{{- else if .ToolCalls }} [TOOL_CALLS] [
{{- range .ToolCalls }}{{ "{" }}"name": "{{ .Function.Name }}", "arguments": {{ json .Function.Arguments }}{{ "}" }}
{{- end }}]</s>
{{- end }}
{{- else if eq .Role "tool" }}[TOOL_RESULTS] {{ json .Content }}[/TOOL_RESULTS]
{{- end }}
{{- end }}
{{- else }}[INST] {{ if .System }}{{ .System }} {{ end }}{{ .Prompt }} [/INST]
{{- end -}}`)),
			s:  `[TOOL_CALLS]  [{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}},{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}]`,
			ok: true,
		},
		{
			name: "no toolcalls",
			tmpl: must(template.Parse(`{{- if .Messages }}
{{- range $index, $_ := .Messages }}
{{- if eq .Role "user" }}
{{- if and (eq (len (slice $.Messages $index)) 1) $.Tools }}[AVAILABLE_TOOLS] {{ json $.Tools }}[/AVAILABLE_TOOLS]
{{- end }}[INST] {{ if and (eq (len (slice $.Messages $index)) 1) $.System }}{{ $.System }}{{ printf "\n\n" }}
{{- end }}{{ .Content }}[/INST]
{{- else if eq .Role "assistant" }}
{{- if .Content }} {{ .Content }}</s>
{{- else if .ToolCalls }} [TOOL_CALLS] [
{{- range .ToolCalls }}{{ "{" }}"name": "{{ .Function.Name }}", "arguments": {{ json .Function.Arguments }}{{ "}" }}
{{- end }}]</s>
{{- end }}
{{- else if eq .Role "tool" }}[TOOL_RESULTS] {{ json .Content }}[/TOOL_RESULTS]
{{- end }}
{{- end }}
{{- else }}[INST] {{ if .System }}{{ .System }} {{ end }}{{ .Prompt }} [/INST]
{{- end -}}`)),
			s:  `The weather is nice today.`,
			ok: false,
		},
		{
			name: "command-r-plus",
			tmpl: must(template.Parse(`{{- if or .Tools .System }}<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>
{{- if .Tools }}# Safety Preamble
The instructions in this section override those in the task description and style guide sections. Don't answer questions that are harmful or immoral.

# System Preamble
## Basic Rules
You are a powerful conversational AI trained by Cohere to help people. You are augmented by a number of tools, and your job is to use and consume the output of these tools to best help the user. You will see a conversation history between yourself and a user, ending with an utterance from the user. You will then see a specific instruction instructing you what kind of response to generate. When you answer the user's requests, you cite your sources in your answers, according to those instructions.

{{ if .System }}# User Preamble
{{ .System }}
{{- end }}

## Available Tools
Here is a list of tools that you have available to you:
{{- range .Tools }}

` + "```" + `python
def {{ .Function.Name }}(
{{- range $name, $property := .Function.Parameters.Properties }}{{ $name }}: {{ $property.Type }}, {{ end }}) -> List[Dict]:
    '''{{ .Function.Description }}

{{- if .Function.Parameters.Properties }}
    Args:
{{- range $name, $property := .Function.Parameters.Properties }}
        {{ $name }} ({{ $property.Type }}): {{ $property.Description }}
{{- end }}
{{- end }}
    '''
    pass
` + "```" + `
{{- end }}
{{- else if .System }}{{ .System }}
{{- end }}<|END_OF_TURN_TOKEN|>
{{- end }}
{{- range .Messages }}<|START_OF_TURN_TOKEN|>
{{- if eq .Role "user" }}<|USER_TOKEN|>
{{- else if or (eq .Role "assistant") (eq .Role "tool") }}<|CHATBOT_TOKEN|>
{{- end }}
{{- if .Content }}{{ .Content }}
{{- else if .ToolCalls }}
Action: ` + "```" + `json
[
{{- range .ToolCalls }}
    {
        "tool_name": "{{ .Function.Name }}",
        "parameters": {{ json .Function.Arguments }}
    }
{{- end }}
]
{{- end }}<|END_OF_TURN_TOKEN|>
{{- end }}
{{- if .Tools }}<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>Write 'Action:' followed by a json-formatted list of actions that you want to perform in order to produce a good response to the user's last input. You can use any of the supplied tools any number of times, but you should aim to execute the minimum number of necessary actions for the input. You should use the ` + "`directly-answer`" + ` tool if calling the other tools is unnecessary. The list of actions you want to call should be formatted as a list of json objects, for example:
` + "```" + `json
[
    {
        "tool_name": title of the tool in the specification,
        "parameters": a dict of parameters to input into the tool as they are defined in the specs, or {} if it takes no parameters
    }
]` + "```" + `
{{- end }}<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>`)),
			s: "Action: ```json" + `
[
    {
        "tool_name": "get_current_weather",
        "parameters": {
            "format": "fahrenheit",
            "location": "San Francisco, CA"
        }
    },
    {
        "tool_name": "get_current_weather",
        "parameters": {
            "format": "celsius",
            "location": "Toronto, Canada"
        }
    }
]
` + "```",
			ok: true,
		},
		{
			name: "firefunction",
			tmpl: must(template.Parse(`{{- if or .System .Tools }}<|start_header_id|>system<|end_header_id|>
{{- if .System }}
{{ .System }}
{{- end }}
In addition to plain text responses, you can chose to call one or more of the provided functions.

Use the following rule to decide when to call a function:
  * if the response can be generated from your internal knowledge (e.g., as in the case of queries like "What is the capital of Poland?"), do so
  * if you need external information that can be obtained by calling one or more of the provided functions, generate a function calls

If you decide to call functions:
  * prefix function calls with functools marker (no closing marker required)
  * all function calls should be generated in a single JSON list formatted as functools[{"name": [function name], "arguments": [function arguments as JSON]}, ...]
  * follow the provided JSON schema. Do not hallucinate arguments or values. Do to blindly copy values from the provided samples
  * respect the argument type formatting. E.g., if the type if number and format is float, write value 7 as 7.0
  * make sure you pick the right functions that match the user intent

Available functions as JSON spec:
{{- if .Tools }}
{{ json .Tools }}
{{- end }}
Today is {{ now }}.<|eot_id|>
{{- end }}
{{- range .Messages }}<|start_header_id|>
{{- if or (eq .Role "user") (eq .Role "assistant") }}{{ .Role }}
{{- else if eq .Role "tool" }}assistant
{{- end }}<|end_header_id|>
{{- if .Content }}{{ .Content }}
{{- else if .ToolCalls }} functools[
{{- range .ToolCalls }}{{ "{" }}"name": "{{ .Function.Name }}", "arguments": {{ json .Function.Arguments }}{{ "}" }}
{{- end }}]
{{- end }}<|eot_id|>
{{- end }}<|start_header_id|>assistant<|end_header_id|>`)),
			s:    ` functools[{"name": "get_current_weather", "arguments": {"format":"fahrenheit","location":"San Francisco, CA"}},{"name": "get_current_weather", "arguments": {"format":"celsius","location":"Toronto, Canada"}}]`,
			ok:   true,
		},
	}

	expect := []api.ToolCall{
		{
			Type: "function",
			Function: function{
				Name: "get_current_weather",
				Arguments: map[string]any{
					"format":   "fahrenheit",
					"location": "San Francisco, CA",
				},
			},
		},
		{
			Type: "function",
			Function: function{
				Name: "get_current_weather",
				Arguments: map[string]any{
					"format":   "celsius",
					"location": "Toronto, Canada",
				},
			},
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			m := &Model{Template: tt.tmpl}

			actual, ok := m.parseToolCalls(tt.s)
			if ok != tt.ok {
				t.Fatalf("expected %v, got %v", tt.ok, ok)
			}

			if tt.ok {
				// only check actual if we expect it to be valid
				for i := range actual {
					// zero out ID since it's generated by the server
					actual[i].ID = ""
				}

				if diff := cmp.Diff(actual, expect); diff != "" {
					t.Errorf("mismatch (-got +want)\n%s", diff)
				}
			}
		})
	}
}
