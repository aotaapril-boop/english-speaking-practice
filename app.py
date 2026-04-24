"""
English Speaking Practice App
- Two modes: Ophthalmology English / Daily & Childcare English
- Web Speech API for STT (continuous, trigger-word stop) and TTS
- Groq API (Llama) for cleansing + feedback
"""

import streamlit as st
import streamlit.components.v1 as components
from groq import Groq
import json

st.set_page_config(page_title="English Speaking Practice", layout="centered")

# ─── Groq Setup ─────────────────────────────────────────────

GROQ_API_KEY = st.secrets.get("groq_api_key", "")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
MODEL = "llama-3.3-70b-versatile"

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

SYSTEM_PROMPT_OPHTH = """あなたは眼科医向けの英語教師です。
眼科の臨床現場で使う自然な日本語の文を1つ生成してください。ユーザーはこの文を英語に翻訳する練習をします。

場面の例（さまざまな眼科領域から出題してください）：
- 網膜・硝子体：OCT所見の説明、抗VEGF注射、硝子体手術、網膜剥離、糖尿病網膜症、加齢黄斑変性
- 角膜：角膜潰瘍、角膜移植（DSAEK/DMEK）、円錐角膜、角膜内皮細胞、ドライアイ、コンタクトレンズ合併症
- 結膜：アレルギー性結膜炎、翼状片、結膜弛緩症、ウイルス性結膜炎（アデノウイルス）
- 緑内障：眼圧測定、視野検査（ハンフリー/ゴールドマン）、点眼指導、SLT/ALT、濾過手術、MIGS
- 神経眼科：視神経炎、うっ血乳頭、外転神経麻痺、瞳孔不同、RAPD、巨細胞性動脈炎
- ぶどう膜炎：前部ぶどう膜炎、サルコイドーシス、ベーチェット病、原田病、CMV網膜炎
- 白内障：術前説明、IOL選択（単焦点/多焦点/トーリック）、術後合併症（PCO、CME）
- 眼形成・涙道：眼瞼下垂、涙道閉塞、DCR、眼窩骨折、甲状腺眼症
- 小児眼科・斜視：弱視訓練、斜視手術、ROP（未熟児網膜症）、先天白内障
- 患者への検査結果の説明
- 同僚・学会での症例報告
- 治療方針の相談・インフォームドコンセント

重要：
- 上記の領域からバランスよく、毎回異なる領域の文を出題してください
- 日本の眼科医が実際に使うような自然な日本語で書いてください
- 「〜が認められます」「〜を施行しました」のような医療日本語を使ってください
- 助詞（に/で/を/へ等）を正しく使い、文法的に正確な文にしてください
- 不自然な直訳調や、意味が通らない文は絶対に出力しないでください
- 出力前に、日本語ネイティブが読んで違和感がないか必ず確認してください
- 略語はそのまま使用：IRF, SRF, PED, ERM, VMT, PVD, SHRM, EZ, BRVO, CRVO, DME, CNV, IOL, PCO, CME, DSAEK, DMEK, SLT, MIGS, RAPD, ROP, DCR
- 日本語の文のみを出力してください。説明や英訳は不要です"""

SYSTEM_PROMPT_DAILY = """あなたは育児・日常英会話の英語教師です。
子育て中の親が日常で使う自然な日本語の文を1つ生成してください。ユーザーはこの文を英語に翻訳する練習をします。

場面の例：
- 子どもへの声かけ（ごはん、お風呂、遊び、寝かしつけ）
- 保育園・幼稚園の先生との会話
- 小児科の受診
- 買い物、公園、ママ友との会話

重要：
- 日本の親が実際に口にするような自然な日本語で書いてください
- 助詞（に/で/を/へ等）を正しく使い、文法的に正確な文にしてください
- 不自然な直訳調や、意味が通らない文は絶対に出力しないでください
- 出力前に、日本語ネイティブが読んで違和感がないか必ず確認してください
- 日本語の文のみを出力してください。説明や英訳は不要です"""

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

Important:
- Check prepositions (in/on/at/to/for/with etc.) carefully — these are the English equivalent of Japanese particles
- Check article usage (a/an/the) carefully
- Check verb tense consistency
- The "explanation" field MUST be written in natural Japanese (日本語ネイティブが読んで違和感のない日本語で書くこと)
- Use correct Japanese particles (助詞) in the explanation — 「に/で/を/へ/が/は」等を正確に使うこと
- Do NOT write unnatural machine-translated Japanese in the explanation

Provide feedback in this exact JSON format:
{{
  "grammar_score": <1-5>,
  "natural_score": <1-5>,
  "corrections": "<corrected version of user's sentence, or 'Perfect!' if no issues>",
  "explanation": "<日本語で文法・用法の問題点を簡潔に説明。自然な日本語で書くこと>",
  "better_expression": "<a more natural/professional way to say it>",
  "model_answer": "<your ideal translation of the Japanese>"
}}

Respond with ONLY the JSON, no markdown formatting."""

# ─── CSS ────────────────────────────────────────────────────

st.markdown("""
<style>
.block-container { padding-top: 3rem; padding-bottom: 0; max-width: 700px; }
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

def _chat(message):
    if not client:
        return "API key not set. Add groq_api_key to Streamlit secrets."
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": message}],
            temperature=0.7,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

def generate_prompt(mode):
    sys_prompt = SYSTEM_PROMPT_OPHTH if mode == "ophthalmology" else SYSTEM_PROMPT_DAILY
    return _chat(sys_prompt)

def cleanse_speech(raw_text, prompt):
    filled = CLEANSE_PROMPT.format(prompt=prompt, raw_text=raw_text)
    result = _chat(filled)
    return result if not result.startswith("Error") else raw_text

def get_feedback(cleaned, prompt):
    filled = FEEDBACK_PROMPT.format(prompt=prompt, cleaned=cleaned)
    text = _chat(filled)
    if text.startswith("Error"):
        return None
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

    # Web Speech API — runs in parent window via st.markdown (not iframe)
    st.markdown("""
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
        <button id="submit-btn" onclick="submitResult()" style="
            display:none; background:#0984e3; color:#fff; border:none;
            border-radius:8px; padding:10px 24px; font-size:1rem;
            cursor:pointer; margin:8px;
        ">Submit</button>
    </div>

    <script>
    var recognition = null;
    var isListening = false;
    var fullTranscript = '';
    var interimText = '';
    var STOP_TRIGGERS = ["that's it", "that's all", "finished", "done"];

    function toggleMic() {
        if (isListening) {
            stopListening();
            document.getElementById('submit-btn').style.display = 'inline-block';
        } else {
            startListening();
        }
    }

    function startListening() {
        var SR = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SR) {
            document.getElementById('status').textContent = 'Speech recognition not supported. Use Chrome.';
            return;
        }
        recognition = new SR();
        recognition.lang = 'en-US';
        recognition.continuous = true;
        recognition.interimResults = true;

        recognition.onstart = function() {
            isListening = true;
            document.getElementById('mic-btn').style.background = '#d63031';
            document.getElementById('mic-btn').style.boxShadow = '0 0 20px rgba(214,48,49,0.6)';
            document.getElementById('mic-btn').textContent = '⏹';
            document.getElementById('status').textContent = 'Listening... tap again to stop';
            document.getElementById('submit-btn').style.display = 'none';
            fullTranscript = '';
        };

        recognition.onresult = function(event) {
            interimText = '';
            for (var i = event.resultIndex; i < event.results.length; i++) {
                var t = event.results[i][0].transcript;
                if (event.results[i].isFinal) {
                    var lower = t.toLowerCase().trim();
                    var triggered = false;
                    for (var j = 0; j < STOP_TRIGGERS.length; j++) {
                        if (lower.includes(STOP_TRIGGERS[j])) {
                            fullTranscript = fullTranscript.replace(new RegExp(STOP_TRIGGERS[j], 'gi'), '').trim();
                            stopListening();
                            submitResult();
                            triggered = true;
                            break;
                        }
                    }
                    if (!triggered) fullTranscript += t + ' ';
                } else {
                    interimText += t;
                }
            }
            document.getElementById('live-text').innerHTML =
                '<span style="color:#55efc4;">' + fullTranscript + '</span>' +
                '<span style="color:#636e72;">' + interimText + '</span>';
        };

        recognition.onerror = function(event) {
            if (event.error === 'no-speech') return;
            document.getElementById('status').textContent = 'Error: ' + event.error;
        };

        recognition.onend = function() {
            if (isListening) {
                try { recognition.start(); } catch(e) {}
            }
        };

        try { recognition.start(); } catch(e) {
            document.getElementById('status').textContent = 'Error: ' + e.message;
        }
    }

    function stopListening() {
        isListening = false;
        if (recognition) recognition.abort();
        document.getElementById('mic-btn').style.background = '#e17055';
        document.getElementById('mic-btn').style.boxShadow = '0 4px 15px rgba(225,112,85,0.4)';
        document.getElementById('mic-btn').textContent = '🎤';
        document.getElementById('status').textContent = 'Stopped — tap Submit or record again';
        document.getElementById('submit-btn').style.display = 'inline-block';
    }

    function submitResult() {
        var text = fullTranscript.trim();
        if (!text) return;
        document.getElementById('status').textContent = 'Submitting...';
        document.getElementById('submit-btn').style.display = 'none';
        var url = new URL(window.location);
        url.searchParams.set('speech_result', encodeURIComponent(text));
        window.location.href = url.toString();
    }
    </script>
    """, unsafe_allow_html=True)

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
