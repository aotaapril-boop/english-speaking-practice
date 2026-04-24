"""
English Speaking Practice App
- Two modes: Ophthalmology English / Daily & Childcare English
- Web Speech API for STT (continuous, trigger-word stop) and TTS
- Gemini API for cleansing + feedback
"""

import streamlit as st
import google.generativeai as genai
import json

st.set_page_config(page_title="English Speaking Practice", layout="centered")

# ─── Gemini Setup ───────────────────────────────────────────

GEMINI_API_KEY = st.secrets.get("gemini_api_key", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash")
else:
    model = None

# ─── Mode & State ───────────────────────────────────────────

if "mode" not in st.session_state:
    st.session_state.mode = "ophthalmology"
if "history" not in st.session_state:
    st.session_state.history = []
if "current_prompt" not in st.session_state:
    st.session_state.current_prompt = ""
if "step" not in st.session_state:
    st.session_state.step = "idle"  # idle -> prompt_shown -> waiting_speech -> processing -> feedback

# ─── System Prompts ─────────────────────────────────────────

SYSTEM_PROMPT_OPHTH = """You are an English tutor specializing in ophthalmology clinical English.
Generate practice sentences that an ophthalmologist would use in clinical settings:
- Explaining OCT findings to patients or colleagues
- Describing retinal pathology (IRF, SRF, PED, ERM, VMT, drusen, EZ disruption, SHRM, etc.)
- Discussing treatment plans (anti-VEGF injection, observation, laser, vitrectomy)
- Patient consultations and informed consent

Keep abbreviations as-is: IRF, SRF, PED, ERM, VMT, PVD, VH, SHRM, EZ, HRF, BRVO, CRVO, DME, CNV, OCTA.
Generate ONE Japanese sentence for the user to translate into English.
Respond with ONLY the Japanese sentence, nothing else."""

SYSTEM_PROMPT_DAILY = """You are an English tutor for daily conversation and childcare English.
Generate practice sentences that a parent would use:
- Talking to children (feeding, bath time, playing, bedtime)
- Daycare/school communication
- Pediatrician visits
- Daily errands and conversations

Generate ONE Japanese sentence for the user to translate into English.
Respond with ONLY the Japanese sentence, nothing else."""

CLEANSE_PROMPT = """You are a speech-to-text post-processor. The user is practicing English speaking.
The raw STT text below contains hesitations (um, ah, uh), false starts, and self-corrections.

Your task:
1. Infer what the user INTENDED to say as a single natural English sentence
2. Use context: the user was translating the Japanese prompt shown below
3. Preserve medical/ophthalmology terms exactly (IRF, SRF, PED, ERM, VMT, etc.)
4. Output ONLY the cleaned sentence, nothing else.

Japanese prompt: {prompt}
Raw STT text: {raw_text}"""

FEEDBACK_PROMPT = """You are an English tutor. The user tried to translate this Japanese sentence into English.

Japanese original: {prompt}
User's English (cleaned): {cleaned}

Provide feedback in this exact JSON format:
{{
  "grammar_score": <1-5>,
  "natural_score": <1-5>,
  "corrections": "<corrected version of user's sentence, or 'Perfect!' if no issues>",
  "explanation": "<brief explanation in Japanese of any grammar/usage issues>",
  "better_expression": "<a more natural/professional way to say it>",
  "model_answer": "<your ideal translation of the Japanese>"
}}

Respond with ONLY the JSON, no markdown formatting."""

# ─── CSS ────────────────────────────────────────────────────

st.markdown("""
<style>
.block-container { padding-top: 1rem; padding-bottom: 0; max-width: 700px; }
.mode-toggle { text-align: center; margin-bottom: 0.5rem; }
.prompt-box {
    background: #1a1a2e; border: 2px solid #e94560;
    border-radius: 12px; padding: 1.2rem;
    font-size: 1.3rem; text-align: center;
    margin: 0.8rem 0; color: #fff;
}
.feedback-box {
    background: #16213e; border-radius: 12px;
    padding: 1rem; margin: 0.5rem 0; color: #fff;
}
.score-badge {
    display: inline-block; padding: 4px 12px;
    border-radius: 20px; font-weight: bold;
    margin: 2px 4px; font-size: 0.9rem;
}
.score-good { background: #00b894; color: #fff; }
.score-ok { background: #fdcb6e; color: #333; }
.score-bad { background: #e17055; color: #fff; }
.user-text {
    background: #2d3436; border-radius: 8px;
    padding: 0.8rem; margin: 0.3rem 0;
    font-style: italic; color: #dfe6e9;
}
.correction { color: #55efc4; font-weight: bold; }
.better { color: #74b9ff; }

/* Mobile: large mic button at bottom */
@media (max-width: 767px) {
    .block-container { padding-top: 0.5rem; }
    .prompt-box { font-size: 1.1rem; padding: 0.8rem; }
}
</style>
""", unsafe_allow_html=True)

# ─── Helper Functions ───────────────────────────────────────

def generate_prompt(mode):
    if not model:
        return "Gemini API key not set. Add gemini_api_key to Streamlit secrets."
    sys_prompt = SYSTEM_PROMPT_OPHTH if mode == "ophthalmology" else SYSTEM_PROMPT_DAILY
    response = model.generate_content(sys_prompt)
    return response.text.strip()

def cleanse_speech(raw_text, prompt):
    if not model:
        return raw_text
    filled = CLEANSE_PROMPT.format(prompt=prompt, raw_text=raw_text)
    response = model.generate_content(filled)
    return response.text.strip()

def get_feedback(cleaned, prompt):
    if not model:
        return None
    filled = FEEDBACK_PROMPT.format(prompt=prompt, cleaned=cleaned)
    response = model.generate_content(filled)
    text = response.text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"grammar_score": 0, "natural_score": 0,
                "corrections": text, "explanation": "Parse error",
                "better_expression": "", "model_answer": ""}

def score_class(score):
    if score >= 4: return "score-good"
    if score >= 3: return "score-ok"
    return "score-bad"

# ─── UI ─────────────────────────────────────────────────────

# Mode toggle
col1, col2 = st.columns(2)
with col1:
    if st.button("Ophthalmology", use_container_width=True,
                 type="primary" if st.session_state.mode == "ophthalmology" else "secondary"):
        st.session_state.mode = "ophthalmology"
        st.session_state.step = "idle"
        st.rerun()
with col2:
    if st.button("Daily & Childcare", use_container_width=True,
                 type="primary" if st.session_state.mode == "daily" else "secondary"):
        st.session_state.mode = "daily"
        st.session_state.step = "idle"
        st.rerun()

mode_label = "Ophthalmology" if st.session_state.mode == "ophthalmology" else "Daily & Childcare"
st.markdown(f"<div style='text-align:center;color:#636e72;font-size:0.85rem;'>Mode: {mode_label}</div>",
            unsafe_allow_html=True)

st.markdown("---")

# Generate new prompt button
if st.button("New Sentence", use_container_width=True, type="primary"):
    with st.spinner("Generating..."):
        st.session_state.current_prompt = generate_prompt(st.session_state.mode)
        st.session_state.step = "prompt_shown"
    st.rerun()

# Show current prompt
if st.session_state.current_prompt:
    st.markdown(f'<div class="prompt-box">{st.session_state.current_prompt}</div>',
                unsafe_allow_html=True)

    # TTS button for prompt (read Japanese aloud)
    st.html(f"""
    <button onclick="
        var u = new SpeechSynthesisUtterance('{st.session_state.current_prompt.replace("'", "\\'")}');
        u.lang = 'ja-JP'; u.rate = 0.9;
        speechSynthesis.speak(u);
    " style="background:#6c5ce7;color:#fff;border:none;border-radius:8px;
             padding:8px 20px;cursor:pointer;font-size:0.9rem;margin:4px 0;">
        Read Aloud (Japanese)
    </button>
    """)

# ─── Speech Recognition Component ──────────────────────────

if st.session_state.current_prompt and st.session_state.step in ("prompt_shown", "waiting_speech"):
    st.markdown("### Speak your English translation")
    st.markdown("<p style='color:#b2bec3;font-size:0.85rem;'>Press the mic button and speak. "
                "Say <b>\"That's it\"</b>, <b>\"That's all\"</b>, or <b>\"Finished\"</b> to stop.</p>",
                unsafe_allow_html=True)

    # Web Speech API component
    st.html("""
    <div id="speech-area" style="text-align:center;">
        <button id="mic-btn" onclick="toggleMic()" style="
            width: 80px; height: 80px; border-radius: 50%;
            background: #e17055; color: white; border: none;
            font-size: 2rem; cursor: pointer; margin: 10px;
            box-shadow: 0 4px 15px rgba(225,112,85,0.4);
            transition: all 0.3s;
        ">🎤</button>
        <div id="status" style="color:#b2bec3;font-size:0.85rem;margin:5px;">Tap to start</div>
        <div id="live-text" style="
            background:#2d3436;border-radius:8px;padding:12px;
            margin:10px 0;min-height:60px;color:#dfe6e9;
            font-size:1rem;text-align:left;white-space:pre-wrap;
        "></div>
        <input type="hidden" id="final-result" />
    </div>

    <script>
    var recognition = null;
    var isListening = false;
    var fullTranscript = '';
    var interimText = '';
    var STOP_TRIGGERS = ["that's it", "that's all", "finished", "done", "owari", "おわり", "終わり"];

    function toggleMic() {
        if (isListening) {
            stopListening();
        } else {
            startListening();
        }
    }

    function startListening() {
        if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
            document.getElementById('status').textContent = 'Speech recognition not supported in this browser.';
            return;
        }
        var SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        recognition = new SpeechRecognition();
        recognition.lang = 'en-US';
        recognition.continuous = true;
        recognition.interimResults = true;

        recognition.onstart = function() {
            isListening = true;
            document.getElementById('mic-btn').style.background = '#d63031';
            document.getElementById('mic-btn').style.boxShadow = '0 0 20px rgba(214,48,49,0.6)';
            document.getElementById('status').textContent = 'Listening... (say "That\\'s it" to finish)';
            fullTranscript = '';
        };

        recognition.onresult = function(event) {
            interimText = '';
            for (var i = event.resultIndex; i < event.results.length; i++) {
                var transcript = event.results[i][0].transcript;
                if (event.results[i].isFinal) {
                    fullTranscript += transcript + ' ';
                    // Check stop triggers
                    var lower = transcript.toLowerCase().trim();
                    for (var j = 0; j < STOP_TRIGGERS.length; j++) {
                        if (lower.includes(STOP_TRIGGERS[j])) {
                            // Remove trigger word from transcript
                            fullTranscript = fullTranscript.replace(new RegExp(STOP_TRIGGERS[j], 'gi'), '').trim();
                            stopListening();
                            submitResult();
                            return;
                        }
                    }
                } else {
                    interimText += transcript;
                }
            }
            document.getElementById('live-text').innerHTML =
                '<span style="color:#55efc4;">' + fullTranscript + '</span>' +
                '<span style="color:#636e72;">' + interimText + '</span>';
        };

        recognition.onerror = function(event) {
            if (event.error === 'no-speech') {
                // Ignore no-speech errors - user is thinking
                return;
            }
            document.getElementById('status').textContent = 'Error: ' + event.error;
        };

        recognition.onend = function() {
            // Auto-restart if user didn't explicitly stop (continuous listening)
            if (isListening) {
                try { recognition.start(); } catch(e) {}
            }
        };

        try {
            recognition.start();
        } catch(e) {
            document.getElementById('status').textContent = 'Error starting: ' + e.message;
        }
    }

    function stopListening() {
        isListening = false;
        if (recognition) {
            recognition.abort();
        }
        document.getElementById('mic-btn').style.background = '#e17055';
        document.getElementById('mic-btn').style.boxShadow = '0 4px 15px rgba(225,112,85,0.4)';
        document.getElementById('status').textContent = 'Stopped';
    }

    function submitResult() {
        var text = fullTranscript.trim();
        if (!text) return;
        document.getElementById('status').textContent = 'Submitting...';
        // Send to Streamlit via query param
        var url = new URL(window.parent.location);
        url.searchParams.set('speech_result', encodeURIComponent(text));
        window.parent.location.href = url.toString();
    }
    </script>
    """)

    # Manual text input fallback
    with st.expander("Type instead"):
        manual = st.text_area("Your English:", key="manual_input", height=80)
        if st.button("Submit text"):
            if manual.strip():
                st.session_state.raw_speech = manual.strip()
                st.session_state.step = "processing"
                st.rerun()

# ─── Process speech result from URL param ───────────────────

query_params = st.query_params
if "speech_result" in query_params:
    import urllib.parse
    raw = urllib.parse.unquote(query_params["speech_result"])
    st.session_state.raw_speech = raw
    st.session_state.step = "processing"
    st.query_params.clear()
    st.rerun()

# ─── Processing & Feedback ──────────────────────────────────

if st.session_state.step == "processing" and hasattr(st.session_state, "raw_speech"):
    raw = st.session_state.raw_speech
    prompt = st.session_state.current_prompt

    st.markdown("### Processing...")
    st.markdown(f'<div class="user-text">Raw: {raw}</div>', unsafe_allow_html=True)

    with st.spinner("Cleansing speech..."):
        cleaned = cleanse_speech(raw, prompt)

    st.markdown(f'<div class="user-text">Cleaned: <b>{cleaned}</b></div>', unsafe_allow_html=True)

    with st.spinner("Getting feedback..."):
        feedback = get_feedback(cleaned, prompt)

    if feedback:
        gs = feedback.get("grammar_score", 0)
        ns = feedback.get("natural_score", 0)
        st.markdown(f"""
        <div class="feedback-box">
            <div style="margin-bottom:8px;">
                <span class="score-badge {score_class(gs)}">Grammar: {gs}/5</span>
                <span class="score-badge {score_class(ns)}">Naturalness: {ns}/5</span>
            </div>
            <p><b>Your sentence:</b> {cleaned}</p>
            <p class="correction"><b>Corrections:</b> {feedback.get('corrections', '')}</p>
            <p style="color:#ffeaa7;"><b>Explanation:</b> {feedback.get('explanation', '')}</p>
            <p class="better"><b>Better expression:</b> {feedback.get('better_expression', '')}</p>
            <p style="color:#a29bfe;"><b>Model answer:</b> {feedback.get('model_answer', '')}</p>
        </div>
        """, unsafe_allow_html=True)

        # TTS for model answer
        model_ans = feedback.get('model_answer', '').replace("'", "\\'")
        if model_ans:
            st.html(f"""
            <button onclick="
                var u = new SpeechSynthesisUtterance('{model_ans}');
                u.lang = 'en-US'; u.rate = 0.85;
                speechSynthesis.speak(u);
            " style="background:#0984e3;color:#fff;border:none;border-radius:8px;
                     padding:8px 20px;cursor:pointer;font-size:0.9rem;margin:4px 0;">
                Listen to Model Answer
            </button>
            """)

        # Save to history
        st.session_state.history.append({
            "prompt": prompt,
            "raw": raw,
            "cleaned": cleaned,
            "feedback": feedback,
            "mode": st.session_state.mode,
        })

    st.session_state.step = "feedback_shown"

# ─── History ────────────────────────────────────────────────

if st.session_state.history:
    with st.expander(f"History ({len(st.session_state.history)} sentences)"):
        for i, h in enumerate(reversed(st.session_state.history)):
            fb = h.get("feedback", {})
            st.markdown(f"""
            **#{len(st.session_state.history)-i}.** {h['prompt']}
            - You said: _{h['cleaned']}_
            - Score: Grammar {fb.get('grammar_score','-')}/5, Natural {fb.get('natural_score','-')}/5
            - Model: {fb.get('model_answer','')}
            """)
