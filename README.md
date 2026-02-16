# AI Tools Playground

A collection of AI-powered Python tools built with OpenAI APIs and Gradio.

## Tools

### 1. Voice Interview Simulator (`interview_bot.py`)
A voice-based interview simulator where an AI CTO interviews you for a Senior Vice President / Executive Director role in Technology Delivery.

**Flow:** Mic input → OpenAI Whisper (speech-to-text) → GPT-4o (CTO response) → OpenAI TTS (voice output)

**Run:**
```bash
python interview_bot.py
```
Open `http://localhost:7860` in your browser.

---

### 2. Flux2 Klein Image Generation UI (`flux2_klein_ui.py`)
A Gradio UI for generating images using the Flux2 Klein model via Ollama running locally.

**Requirements:** Ollama running with `x/flux2-klein:latest` model pulled.

**Run:**
```bash
python flux2_klein_ui.py
```

---

### 3. Multi-Model Competition (`multi_model_competition.py`)
Runs a prompt across multiple LLMs simultaneously and compares their responses. Outputs results as a styled HTML report with logs.

**Run:**
```bash
python multi_model_competition.py
```

---

## Setup

```bash
pip install openai gradio python-dotenv requests pillow
```

Create a `.env` file:
```
OPENAI_API_KEY=your_key_here
```

## Architecture Diagram

See `interview_bot_flow.md` for the Mermaid flow diagram of the interview bot.
