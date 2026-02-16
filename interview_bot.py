from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
import tempfile
import os

load_dotenv(override=True)
client = OpenAI()

# --- YOUR BACKGROUND ---
candidate_background = """
Applying for: Senior Vice President / Executive Director, Technology Delivery

Key expertise: Technology delivery leadership — Architecture solutioning, solution design,
managing development teams, planning, and end-to-end delivery across multiple programs.

Technical Skills & Expertise:

- Strong understanding of Java and its advanced concepts including design patterns,
  collections, concurrency, and JVM internals.

- Strong understanding of IAM concepts: authentication, authorization, identity federation,
  single sign-on, and multi-factor authentication.

- Extensive working experience in ForgeRock Identity Platform (AM, IDM, DS, IG),
  including product installation, configuration and customisation.

- Advanced technical skills in Spring Framework, Spring Boot, Spring MVC, Spring Security,
  AOP, RESTful APIs, Spring Data, Spring Cloud and related technologies.

- Experienced in implementing secure authentication and authorization mechanisms using
  Spring Security, OAuth, JWT, SAML, and OpenID Connect.

- Familiar with microservices architecture principles and cloud-native application development
  using OpenShift and Kubernetes.

- Strong knowledge of cryptography concepts, HSM core concepts, key management and integration.

- Experience working with directory services: LDAP and Active Directory.

- Frontend development experience using React/Redux, DHTML, DOM, and JavaScript.

- Deep understanding of databases and data modeling including NoSQL: MongoDB and Redis.

- Well-versed in DevOps practices: CI/CD, automated testing, deployment automation,
  and infrastructure as code.

- Good understanding of the banking domain, customer requirements, and industry trends,
  with the ability to make technical decisions aligned with business goals.

- Strong leadership skills: mentoring junior developers, leading development teams
  across geographies, and communicating effectively with stakeholders.
"""

# --- CTO INTERVIEWER SYSTEM PROMPT ---
system_prompt = """
You are a seasoned Chief Technology Officer at a major financial institution.

## Your persona:
- You have led large-scale technology transformations across banking and financial services
- You think in platforms, architecture, engineering culture, and business outcomes — not just delivery
- You value structured thinking, clear communication, and measurable delivery results
- You are direct and intellectually sharp — you do not accept vague or generic answers
- You ask one focused question at a time, then probe deeper based on the answer
- You speak in a measured, professional tone — authoritative but not aggressive
- When an answer is strong, you acknowledge it briefly and push deeper
- When an answer is weak or vague, you challenge it respectfully
- You benchmark candidates against the standards expected at SVP / Executive Director level

## The role being interviewed for:
Senior Vice President / Executive Director — Technology Delivery

This is a senior leadership role responsible for:
- Architecture solutioning and solution design governance
- Leading and managing development teams across geographies
- End-to-end planning and delivery of complex technology programs
- Stakeholder management and alignment with business goals
- Building engineering culture and delivery excellence

## Interview focus areas (cover these naturally across the conversation):
1. Architecture & Solution Design — how they approach solutioning, trade-offs, governance
2. Delivery — managing complex programs, dependencies, risks, and outcomes at scale
3. Team Leadership — building teams, handling underperformance, cross-geo management
4. Problem Solving — a real hard problem they faced and how they resolved it
5. Planning — roadmaps, ambiguity, stakeholder alignment, prioritisation
6. Business alignment — translating tech decisions into business value

## Rules:
- Ask ONE question at a time — never ask multiple questions in one turn
- Do NOT list topics upfront — let the interview flow naturally
- Keep your responses concise — you are asking questions, not giving speeches
- After 8-10 exchanges, wrap up professionally and invite the candidate to ask you questions
- This is a VOICE conversation — avoid bullet points, markdown, or lists in your responses.
  Write naturally as spoken language.

## Candidate background:
""" + candidate_background + """

Begin the interview with a brief professional welcome and your first question.
"""

# --- CORE FUNCTIONS ---
conversation_history = []


def transcribe(audio_path: str) -> str:
    with open(audio_path, "rb") as f:
        transcript = client.audio.transcriptions.create(model="whisper-1", file=f)
    return transcript.text


def get_reply(user_text: str) -> str:
    messages = [{"role": "system", "content": system_prompt}]
    messages += conversation_history
    messages += [{"role": "user", "content": user_text}]
    response = client.chat.completions.create(model="gpt-4o", messages=messages, temperature=0.7)
    return response.choices[0].message.content


def speak(text: str) -> str:
    response = client.audio.speech.create(model="tts-1", voice="onyx", input=text)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    response.stream_to_file(tmp.name)
    return tmp.name


def build_transcript():
    html = "<div style='font-family: sans-serif; line-height: 1.8;'>"
    for msg in conversation_history:
        if msg["role"] == "user":
            html += f"<p><b style='color:#2563eb;'>You:</b> {msg['content']}</p>"
        else:
            html += f"<p><b style='color:#dc2626;'>CTO:</b> {msg['content']}</p>"
    html += "</div>"
    return html


def start_interview():
    global conversation_history
    conversation_history = []
    reply = get_reply("Let's begin the interview.")
    conversation_history.append({"role": "user", "content": "Let's begin the interview."})
    conversation_history.append({"role": "assistant", "content": reply})
    return speak(reply), build_transcript()


def interview(audio):
    global conversation_history
    if audio is None:
        return None, build_transcript()
    candidate_text = transcribe(audio)
    print(f"You: {candidate_text}")
    reply = get_reply(candidate_text)
    print(f"CTO: {reply}")
    conversation_history.append({"role": "user", "content": candidate_text})
    conversation_history.append({"role": "assistant", "content": reply})
    return speak(reply), build_transcript()


# --- GRADIO UI ---
with gr.Blocks(title="SVP / ED Interview Simulator") as demo:

    gr.Markdown("""
    # SVP / Executive Director Interview Simulator
    ### Interviewer: Chief Technology Officer
    Click **Start Interview** first. Then record your answer and click **Submit**.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            start_btn = gr.Button("▶ Start Interview", variant="primary", size="lg")
            gr.Markdown("---")
            gr.Markdown("### Your Answer")
            audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Record your answer")
            submit_btn = gr.Button("Submit Answer", variant="secondary")

        with gr.Column(scale=2):
            gr.Markdown("### CTO")
            audio_output = gr.Audio(label="Listen")
            gr.Markdown("### Transcript")
            transcript = gr.HTML(value="<p><i>Click Start Interview to begin...</i></p>")

    start_btn.click(fn=start_interview, inputs=[], outputs=[audio_output, transcript])
    submit_btn.click(fn=interview, inputs=[audio_input], outputs=[audio_output, transcript])

demo.launch(server_name="0.0.0.0", server_port=7860)
