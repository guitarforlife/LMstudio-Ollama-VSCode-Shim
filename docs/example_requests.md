## Example requests

### OpenAI text completion

```bash
curl http://localhost:11434/v1/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "my-model",
    "prompt": "Write a haiku about fog",
    "stream": false
  }'
```

### OpenAI chat completion

```bash
curl http://localhost:11434/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "my-model",
    "messages": [
      {"role": "system", "content": "You are concise."},
      {"role": "user", "content": "Summarize the plot of Dune in two sentences."}
    ],
    "stream": false
  }'
```

### Ollama-compatible chat

```bash
curl http://localhost:11434/api/chat \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "my-model",
    "messages": [{"role": "user", "content": "Hello from Ollama chat."}],
    "stream": false
  }'
```

### Ollama-compatible generate

```bash
curl http://localhost:11434/api/generate \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "my-model",
    "prompt": "Explain transformers in one paragraph.",
    "stream": false
  }'
```
