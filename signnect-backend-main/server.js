require('dotenv').config();
const express = require('express');
const cors = require('cors');
const rateLimit = require('express-rate-limit');
const { Groq } = require('groq-sdk');

// ---------- Config ----------
const PORT = process.env.PORT || 3000;
const MODEL = process.env.GROQ_MODEL || 'llama-3.1-8b-instant';
const ALLOWED_ORIGINS = process.env.ALLOWED_ORIGINS; // e.g. "https://myapp.com,http://localhost:5173"
const MAX_WORDS = 30;
const MAX_WORD_LENGTH = 40;

if (!process.env.GROQ_API_KEY || process.env.GROQ_API_KEY.startsWith('YOUR')) {
    console.error('❌ GROQ_API_KEY is missing. Copy .env.example to .env and add your real key.');
    process.exit(1);
}

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY, timeout: 15000, maxRetries: 1 });

const app = express();
app.set('trust proxy', 1); // correct client IPs behind Render/Railway/Heroku proxies
app.use(cors(ALLOWED_ORIGINS ? { origin: ALLOWED_ORIGINS.split(',').map(o => o.trim()) } : undefined));
app.use(express.json({ limit: '10kb' }));
app.use(rateLimit({ windowMs: 60 * 1000, limit: 60, standardHeaders: true, legacyHeaders: false }));

// ---------- Helpers ----------
// Accepts "TODAY I SAD" or ["TODAY","I","SAD"]; returns a clean string or null.
function normalizeWords(input) {
    const list = Array.isArray(input) ? input : typeof input === 'string' ? input.split(/\s+/) : null;
    if (!list) return null;
    const cleaned = list
        .filter(w => typeof w === 'string')
        .map(w => w.replace(/[^\p{L}\p{N}'-]/gu, '').slice(0, MAX_WORD_LENGTH)) // strip anything odd (blocks prompt injection tricks)
        .filter(Boolean)
        .slice(0, MAX_WORDS);
    return cleaned.length ? cleaned.join(' ') : null;
}

// Language is interpolated into a prompt, so only allow plain language names.
function normalizeLanguage(input) {
    if (input === undefined || input === null || input === '') return 'English';
    if (typeof input !== 'string') return null;
    const lang = input.trim();
    return /^[\p{L}][\p{L} ()-]{0,29}$/u.test(lang) ? lang : null;
}

function cleanOutput(text, isEnglish) {
    // keep only the first line, drop wrapping quotes the model sometimes adds
    let out = (text || '').trim().split('\n')[0].trim().replace(/^["'“”]+|["'“”]+$/g, '');
    // Llama keeps starting English sentences with "The" even when told not to
    if (isEnglish) out = out.replace(/^the\s+/i, '');
    return out.charAt(0).toUpperCase() + out.slice(1);
}

// ---------- Routes ----------
app.get('/health', (req, res) => res.json({ status: 'ok' }));

app.post('/generate-sentence', async (req, res) => {
    const words = normalizeWords(req.body?.words);
    if (!words) {
        return res.status(400).json({ error: "'words' is required (a string like \"TODAY I SAD\" or an array of words)" });
    }
    const targetLanguage = normalizeLanguage(req.body?.language);
    if (!targetLanguage) {
        return res.status(400).json({ error: "'language' must be a plain language name, e.g. \"Tamil\"" });
    }
    const isEnglish = targetLanguage.toLowerCase() === 'english';

    try {
        let messages;
        if (isEnglish) {
            const system = 'You are a strict text formatter. Form a readable string by ONLY inserting "is", "am", "are", "a", or "an" strictly between the provided words. DO NOT add words to the beginning of the sentence. DO NOT use "the". DO NOT change the original words or tense. Output only the sentence.';
            messages = [
                { role: 'system', content: system },
                { role: 'user', content: 'Words: TODAY I SAD' },
                { role: 'assistant', content: 'Today I am sad' },
                { role: 'user', content: 'Words: HELLO I RICH' },
                { role: 'assistant', content: 'Hello I am rich' },
                { role: 'user', content: `Words: ${words}` },
            ];
        } else {
            const system = `You are a strict translator. Take the provided words and translate them into ${targetLanguage} as a simple, meaningful sentence. Return ONLY the translated sentence in ${targetLanguage}. Do NOT return the English version. Do NOT return any explanations or additional text.`;
            messages = [
                { role: 'system', content: system },
                { role: 'user', content: `Words: ${words}` },
            ];
        }

        const completion = await groq.chat.completions.create({
            messages,
            model: MODEL,
            temperature: 0.1,
            max_tokens: 100,
        });

        const sentence = cleanOutput(completion.choices[0]?.message?.content, isEnglish);
        if (!sentence) return res.status(502).json({ error: 'Model returned an empty response' });

        res.json({ sentence });
    } catch (error) {
        console.error('Error with Groq API:', error?.status || '', error?.message || error);
        const status = error?.status === 429 ? 429 : 502;
        res.status(status).json({ error: 'Failed to generate sentence' });
    }
});

// ---------- Error handling ----------
app.use((req, res) => res.status(404).json({ error: 'Not found' }));
app.use((err, req, res, next) => { // e.g. malformed JSON, body too large
    const status = err.status || 400;
    res.status(status).json({ error: status === 413 ? 'Request body too large' : 'Invalid request' });
});

app.listen(PORT, () => console.log(`✅ Server running on http://localhost:${PORT}`));
