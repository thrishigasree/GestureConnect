# Signnect Backend

Express API that turns sign words into a proper sentence using Groq (Llama 3.1).

## Setup
```bash
npm install
cp .env.example .env     # then put your real GROQ_API_KEY inside
npm start
```

## API
`POST /generate-sentence`
```json
{ "words": "TODAY I SAD", "language": "English" }
```
`words` can also be an array: `["TODAY","I","SAD"]`. `language` is optional (default English).
Response: `{ "sentence": "Today I am sad" }`

`GET /health` -> `{ "status": "ok" }`

Rate limit: 60 requests/min per IP. Set `ALLOWED_ORIGINS` in production.
