# Template

Ollama provides a powerful templating engine backed by Go's built-in templating engine to construct prompts for your large language model. This feature is a valuable tool to get the most out of your models.

## Basic Template Structure

A basic Go template consists of three main parts:

* **Layout**: The overall structure of the template.
* **Variables**: Placeholders for dynamic data that will be replaced with actual values when the template is rendered.
* **Functions**: Custom functions or logic that can be used to manipulate the template's content.

Here's an example of a simple chat template:

```gotmpl
{{- range .Messages }}
{{ .Role }}: {{ .Content }}
{{- end }}
```

In this example, we have:

* A basic messages structure (layout)
* Three variables: `Messages`, `Role`, and `Content` (variables)
* A custom function (action) that iterates over an array of items (`range .Messages`) and displays each item

## Adding Templates to Your Model

By default, models imported into Ollama have a default template of `{{ .Prompt }}`, i.e. user inputs are sent verbatim to the LLM. This is appropriate for text or code completion models but lacks essential markers for chat or instruction models.

Omitting a template in these models puts the responsibility of correctly templating input onto the user. Adding a template allows users to get the best results from the model easily.

To add templates in your model, you'll need to add a `TEMPLATE` command to the Modelfile. Here's an example using Meta's Llama 3.

```dockerfile
FROM llama3

TEMPLATE """
{{- if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>
{{- end }}
{{- range .Messages }}<|start_header_id|>{{ .Role }}<|end_header_id|>

{{ .Content }}<|eot_id|>
{{- end }}<|start_header_id|>assistant<|end_header_id|>

"""
```

## Variables

`System` (string): system prompt

`Prompt` (string): user prompt

`Response` (string): assistant response

`Suffix` (string): text inserted after the assistant's response

`Messages` (list): list of messages (described below)

`Messages[].Role` (string): role which can be one of `system`, `user`, `assistant`, or `tool`

`Messages[].Content` (string):  message content

`Messages[].ToolCalls` (list): list of tools the model wants to call (described below)

`Messages[].ToolCalls[].ID` (string):

`Messages[].ToolCalls[].Type` (string): schema type. `type` is always `function`

`Messages[].ToolCalls[].Function.Name` (string): function name

`Messages[].ToolCalls[].Function.Arguments` (map): map of arguments where the map key is the property name and the map value is the value

`Tools` (list): list of tools the model can access (described below)

`Tools[].Type` (string): schema type. `type` is always `function`

`Tools[].Function`

`Tools[].Function.Name` (string): function name

`Tools[].Function.Description` (string): function description

`Tools[].Function.Parameters`

`Tools[].Function.Parameters.Type` (string): schema type. `type` is always `object`

`Tools[].Function.Parameters.Required` (optional list[string]): list of required properties

`Tools[].Function.Parameters.Properties` (map): map of properties where the map key is the property name and the map value is a property definition (described below)

`Tools[].Function.Parameters.Properties[].Type` (string): property type

`Tools[].Function.Parameters.Properties[].Description` (string): property description

`Tools[].Function.Parameters.Properties[].Enum` (optional list[string]): list of valid values

## Functions

In addition to the functions provided by [Go](https://pkg.go.dev/text/template#hdr-Functions), Ollama provides these additional functions:

- `json`: utility function to format JSON, primarily used for tool definitions

## Tips and Best Practices

Keep the following tips and best practices in mind when working with Go templates:

- **Be mindful of dot**: Control flow structures like `range` and `with` changes the value `.`
- **Out-of-scope variables**: Use `$.` to reference variables not currently in scope of `.`, starting from the root

## Examples

### Example Messages

**ChatML**

ChatML is a popular template format. It can be used for models such as Databrick's DBRX, Intel's Neural Chat, and Microsoft's Orca 2.

```gotmpl
{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}
{{- range .Messages }}<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ else }}
{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
```

### Example Tools

**Mistral**

Mistral v0.3 and Codestral v0.1 supports tool calling.

```gotmpl
{{- range $index, $_ := .Messages }}
{{- if eq .Role "user" }}
{{- if and (eq (len (slice $.Messages $index)) 1) $.Tools }}[AVAILABLE_TOOLS] {{ json $.Tools }}[/AVAILABLE_TOOLS]
{{- end }}[INST] {{ if and (eq (len (slice $.Messages $index)) 1) $.System }}{{ $.System }}

{{ end }}{{ .Content }}[/INST]
{{- else if eq .Role "assistant" }}
{{- if .Content }} {{ .Content }}</s>
{{- else if .ToolCalls }}[TOOL_CALLS] [
{{- range .ToolCalls }}{{ "{" }}"name": "{{ .Function.Name }}", "arguments": {{ json .Function.Arguments }}{{ "}" }}
{{- end }}]</s>
{{- end }}
{{- else if eq .Role "tool" }}[TOOL_RESULTS] {{ json . }}[/TOOL_RESULTS]
{{- end }}
{{- end }}
```

### Example Fill-in-Middle

**CodeLlama**

CodeLlama [7B](https://ollama.com/library/codellama:7b-code) and [13B](https://ollama.com/library/codellama:13b-code) code completion models support fill-in-middle.

```gotmpl
<PRE> {{ .Prompt }} <SUF>{{ .Suffix }} <MID>
```

> [!NOTE]
> CodeLlama 34B and 70B code completion and all instruct and Python fine-tuned models do not support fill-in-middle.

**Codestral**

Codestral [22B](https://ollama.com/library/codestral:22b) supports fill-in-middle.

```gotmpl
[SUFFIX]{{ .Suffix }}[PREFIX] {{ .Prompt }}
```
