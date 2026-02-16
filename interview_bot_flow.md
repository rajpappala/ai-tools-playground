# Interview Bot — Application Flow

```mermaid
flowchart TD
    A([User opens browser\nlocalhost:7860]) --> B[Gradio UI loads]
    B --> C[Click ▶ Start Interview]

    C --> D[start_interview]
    D --> E[get_reply\nLet's begin the interview]
    E --> F[GPT-4o\nGenerates CTO welcome\n+ first question]
    F --> G[speak\nOpenAI TTS - onyx voice]
    G --> H[Audio plays in browser]
    H --> I[Transcript updated]

    I --> J[User clicks mic\nRecords answer]
    J --> K[Click Submit Answer]

    K --> L[interview]
    L --> M[transcribe\nOpenAI Whisper]
    M --> N[Candidate text extracted]
    N --> O[get_reply\nHistory + new message]
    O --> P[GPT-4o\nGenerates CTO follow-up\nprobing question]
    P --> Q[speak\nOpenAI TTS]
    Q --> R[Audio plays in browser]
    R --> S[Transcript updated\nwith both turns]

    S --> T{8-10 exchanges\ncomplete?}
    T -- No --> J
    T -- Yes --> U[CTO wraps up\nInvites candidate questions]
    U --> V([Interview ends])

    style A fill:#e0f2fe
    style V fill:#dcfce7
    style F fill:#fef3c7
    style M fill:#fef3c7
    style P fill:#fef3c7
    style G fill:#f3e8ff
    style Q fill:#f3e8ff
```

## Colour Legend

| Colour | Component |
|--------|-----------|
| Blue   | User action |
| Yellow | OpenAI — GPT-4o / Whisper |
| Purple | OpenAI TTS (voice output) |
| Green  | End state |

## Tech Stack

| Component | Technology |
|-----------|------------|
| UI | Gradio 6.x |
| Speech-to-Text | OpenAI Whisper (`whisper-1`) |
| Language Model | OpenAI GPT-4o |
| Text-to-Speech | OpenAI TTS (`tts-1`, voice: `onyx`) |
| Runtime | Python 3.13 |
