"""
English Speaking Practice App
- Three modes: Ophthalmology / Daily & Childcare / Expression Lookup
- Feedback flow: Correction → Deep Dive → Conversation
- GPT-4o → Gemini → Groq fallback chain
"""

import streamlit as st
import json
import random
from datetime import datetime

st.set_page_config(page_title="English Speaking Practice", layout="centered")

# ─── API Setup (GPT-4o → Gemini → Groq) ────────────────────

OPENAI_API_KEY = st.secrets.get("openai_api_key", "")
GEMINI_API_KEY = st.secrets.get("gemini_api_key", "")
GROQ_API_KEY = st.secrets.get("groq_api_key", "")

_openai_client = None
_gemini_model = None
_groq_client = None

if OPENAI_API_KEY:
    from openai import OpenAI
    _openai_client = OpenAI(api_key=OPENAI_API_KEY)

if GEMINI_API_KEY:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    _gemini_model = genai.GenerativeModel("gemini-2.0-flash")

if GROQ_API_KEY:
    from groq import Groq
    _groq_client = Groq(api_key=GROQ_API_KEY)

# ─── State ──────────────────────────────────────────────────

for key, default in [
    ("mode", "ophthalmology"),
    ("history", []),
    ("current_prompt", ""),
    ("step", "idle"),
    ("feedback_data", None),
    ("deep_dive_result", None),
    ("conv_messages", []),
    ("history_loaded", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─── System Prompts ─────────────────────────────────────────

OPHTH_TOPICS = [
    ("網膜・硝子体", "OCT所見の説明、抗VEGF注射、硝子体手術、網膜剥離、糖尿病網膜症、加齢黄斑変性、IRF/SRF/PED/ERM"),
    ("角膜", "角膜潰瘍、角膜移植（DSAEK/DMEK）、円錐角膜、角膜内皮細胞、ドライアイ、コンタクトレンズ合併症"),
    ("結膜", "アレルギー性結膜炎、翼状片、結膜弛緩症、ウイルス性結膜炎（アデノウイルス）"),
    ("緑内障", "眼圧測定、視野検査（ハンフリー/ゴールドマン）、点眼指導、SLT/ALT、濾過手術、MIGS、視神経乳頭陥凹"),
    ("神経眼科", "視神経炎、うっ血乳頭、外転神経麻痺、瞳孔不同、RAPD、巨細胞性動脈炎、視野障害"),
    ("ぶどう膜炎", "前部ぶどう膜炎、サルコイドーシス、ベーチェット病、原田病、CMV網膜炎、免疫抑制剤"),
    ("白内障", "術前説明、IOL選択（単焦点/多焦点/トーリック）、術後合併症（PCO、CME）、phaco"),
    ("眼形成・涙道", "眼瞼下垂、涙道閉塞、DCR、眼窩骨折、甲状腺眼症、眼瞼痙攣"),
    ("小児眼科・斜視", "弱視訓練、斜視手術、ROP（未熟児網膜症）、先天白内障、遮閉法"),
]

def _ophth_prompt():
    topic_name, topic_detail = random.choice(OPHTH_TOPICS)
    return f"""あなたは眼科医向けの英語教師です。
今回は【{topic_name}】の領域から、眼科の臨床現場で使う自然な日本語の文を1つ生成してください。
ユーザーはこの文を英語に翻訳する練習をします。

【{topic_name}】のキーワード例：{topic_detail}

場面の例：
- 患者への検査結果の説明
- 同僚・学会での症例報告
- 治療方針の相談・インフォームドコンセント

重要：
- 必ず【{topic_name}】の領域に関する文を出してください（他の領域は不可）
- 日本の眼科医が実際に使うような自然な日本語で書いてください
- 「〜が認められます」「〜を施行しました」のような医療日本語を使ってください
- 助詞（に/で/を/へ等）を正しく使い、文法的に正確な文にしてください
- 文の意味が論理的・医学的に正しいこと（矛盾する記述はNG）
- 不自然な直訳調や、意味が通らない文は絶対に出力しないでください
- 出力前に、実際の診療や学会で使う文として違和感がないか必ず確認してください
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
- 文の意味が論理的に通ること（矛盾する行動を含めない。例：「寝ながら準備する」のような矛盾はNG）
- 1つの場面で1つの状況を描写する（複数の無関係な行動を詰め込まない）
- 助詞（に/で/を/へ等）を正しく使い、文法的に正確な文にしてください
- 不自然な直訳調や、意味が通らない文は絶対に出力しないでください
- 出力前に、その文を実際に子どもや周りの人に言うか想像して、違和感がないか必ず確認してください
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

FEEDBACK_PROMPT = """You are an expert English tutor. The user tried to translate this Japanese sentence into English.

Japanese original: {prompt}
User's English (cleaned): {cleaned}

Provide detailed feedback in this exact JSON format:
{{
  "grammar_score": <1-5>,
  "natural_score": <1-5>,
  "corrections": "<corrected version of user's sentence, or 'Perfect!' if no issues>",
  "why_corrections": "<日本語で、なぜその修正が必要かを具体的に説明。例：「inではなくatを使う理由は…」>",
  "advanced_expression": "<上級者向けの洗練された言い換え。学術・専門的な表現を使う>",
  "key_words": [
    {{"word": "<重要な英単語/フレーズ>", "meaning": "<日本語での意味>", "usage": "<使い方のポイント>"}},
    {{"word": "<重要な英単語/フレーズ>", "meaning": "<日本語での意味>", "usage": "<使い方のポイント>"}}
  ],
  "model_answer": "<your ideal translation of the Japanese>"
}}

Important:
- Check prepositions (in/on/at/to/for/with) carefully
- Check article usage (a/an/the) carefully
- Check verb tense consistency
- "why_corrections" must explain WHY each correction is needed, not just what changed
- "advanced_expression" should be a noticeably more sophisticated/professional version
- "key_words" should list 2-4 important words/phrases with Japanese explanation and usage tips
- All Japanese text must be natural (日本語ネイティブが読んで違和感のない日本語で書くこと)
- Respond with ONLY the JSON, no markdown formatting."""

DEEP_DIVE_PROMPT = """あなたは英語教師です。以下のフレーズ/単語を深掘りして学習させてください。

対象: {phrase}
元の文脈（日本語）: {context}

以下を提供してください：

**意味と使い方:**
このフレーズの正確な意味、ニュアンス、使われる場面を日本語で説明

**類似表現との違い:**
似た表現2〜3つとのニュアンスの違いを日本語で説明

**練習用の日本語文:**
このフレーズを使って英訳できる新しい日本語の文を3つ生成（難易度：易→中→難）
各文の後に模範英訳も記載

**よくある間違い:**
日本人がこのフレーズを使う時にやりがちな間違いを1〜2つ"""

CONVERSATION_SYSTEM = """You are a friendly English conversation partner.
The user is practicing English on this topic: {topic}
Key phrases from their recent practice: {phrases}

Rules:
- Speak naturally but clearly
- If the user makes a grammar mistake, gently correct it inline (e.g., "*corrected phrase*")
- Use the key phrases in your responses when natural
- Ask follow-up questions to keep the conversation going
- Keep responses to 2-3 sentences
- If the user writes in Japanese, respond in English and provide the translation"""

LOOKUP_PROMPT = """あなたは日英翻訳の専門家です。以下の日本語表現について、自然な英語表現を教えてください。

入力: {query}

以下の形式で回答してください（日本語で説明）：

**自然な英語表現:**
最も一般的で自然な英語表現を1つ

**上級者向けの表現:**
より洗練された/専門的な言い方を1〜2つ（ニュアンスの違いも説明）

**別の言い方:**
同じ意味の別表現があれば2〜3つ（ニュアンスの違いも簡潔に説明）

**例文:**
実際の使用例を3文（英語 + 日本語訳）
- 日常会話での例
- 医療・眼科の文脈での例（可能な場合）
- フォーマルな場面での例

**注意点:**
使い分けのポイントや、日本人が間違えやすいポイントがあれば簡潔に"""

# ─── CSS ────────────────────────────────────────────────────

st.markdown("""
<style>
.block-container { padding-top: 1rem; padding-bottom: 0; max-width: 700px; }
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
.keyword-card {
    background: #2d3436; border-left: 3px solid #6c5ce7;
    border-radius: 0 8px 8px 0; padding: 0.6rem 0.8rem;
    margin: 0.3rem 0; color: #dfe6e9;
}
.advanced-box {
    background: #2d3436; border: 1px solid #6c5ce7;
    border-radius: 8px; padding: 0.8rem; margin: 0.5rem 0;
    color: #a29bfe;
}
.conv-user { background: #2d3436; border-radius: 12px 12px 4px 12px;
    padding: 0.6rem 1rem; margin: 0.3rem 0; color: #dfe6e9; text-align: right; }
.conv-ai { background: #16213e; border-radius: 12px 12px 12px 4px;
    padding: 0.6rem 1rem; margin: 0.3rem 0; color: #fff; }
.date-header {
    color: #636e72; font-size: 0.8rem; font-weight: bold;
    margin: 0.8rem 0 0.3rem 0; padding-bottom: 0.2rem;
    border-bottom: 1px solid #2d3436;
}

header[data-testid="stHeader"] { display: none !important; }
.stAppDeployButton { display: none !important; }
div[data-testid="stDecoration"] { display: none !important; }
div[data-testid="stToolbar"] { display: none !important; }
iframe[title="streamlit_app_toolbar"] { display: none !important; }
#stStreamlitMainMenu { display: none !important; }

@media (max-width: 767px) {
    .block-container { padding-top: 0.2rem; }
    .prompt-box { font-size: 1.1rem; padding: 0.8rem; }
    section[data-testid="stSidebar"] { display: none !important; }
}
</style>
""", unsafe_allow_html=True)

# ─── Helper Functions ───────────────────────────────────────

def _chat(message):
    if _openai_client:
        try:
            resp = _openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": message}],
                temperature=0.7,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            pass
    if _gemini_model:
        try:
            resp = _gemini_model.generate_content(message)
            return resp.text.strip()
        except Exception:
            pass
    if _groq_client:
        try:
            resp = _groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": message}],
                temperature=0.7,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {e}"
    return "API key not set. Add openai_api_key, gemini_api_key, or groq_api_key to Streamlit secrets."

def generate_prompt(mode):
    sys_prompt = _ophth_prompt() if mode == "ophthalmology" else SYSTEM_PROMPT_DAILY
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
                "corrections": text, "why_corrections": "Parse error",
                "advanced_expression": "", "key_words": [],
                "model_answer": ""}

def score_class(score):
    if score >= 4: return "score-good"
    if score >= 3: return "score-ok"
    return "score-bad"

def _save_history_js():
    compact = []
    for h in st.session_state.history[-50:]:
        fb = h.get("feedback", {})
        compact.append({
            "prompt": h.get("prompt", ""),
            "cleaned": h.get("cleaned", ""),
            "mode": h.get("mode", ""),
            "timestamp": h.get("timestamp", ""),
            "feedback": {
                "grammar_score": fb.get("grammar_score", 0),
                "natural_score": fb.get("natural_score", 0),
                "corrections": fb.get("corrections", ""),
                "model_answer": fb.get("model_answer", ""),
            }
        })
    data = json.dumps(compact, ensure_ascii=False)
    escaped = data.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
    st.html(f"""
    <script>
    try {{
        localStorage.setItem('esp_history', `{escaped}`);
    }} catch(e) {{}}
    </script>
    """)

def _load_history_component():
    st.html("""
    <script>
    try {
        const data = localStorage.getItem('esp_history');
        if (data) {
            const parsed = JSON.parse(data);
            if (parsed && parsed.length > 0) {
                const recent = parsed.slice(-50);
                const encoded = btoa(unescape(encodeURIComponent(JSON.stringify(recent))));
                if (encoded.length < 8000) {
                    const params = new URLSearchParams(window.location.search);
                    if (!params.has('h_loaded')) {
                        params.set('h_loaded', '1');
                        params.set('h_data', encoded);
                        window.location.search = params.toString();
                    }
                }
            }
        }
    } catch(e) {}
    </script>
    """)

if not st.session_state.history_loaded:
    st.session_state.history_loaded = True
    params = st.query_params
    if "h_data" in params:
        try:
            decoded = json.loads(
                __import__('base64').b64decode(params["h_data"]).decode('utf-8')
            )
            if decoded and not st.session_state.history:
                st.session_state.history = decoded
        except Exception:
            pass
    else:
        _load_history_component()


# ─── UI ─────────────────────────────────────────────────────

col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("Ophth", use_container_width=True,
                 type="primary" if st.session_state.mode == "ophthalmology" else "secondary"):
        st.session_state.mode = "ophthalmology"
        st.session_state.step = "idle"
        st.session_state.conv_messages = []
        st.rerun()
with col2:
    if st.button("Daily", use_container_width=True,
                 type="primary" if st.session_state.mode == "daily" else "secondary"):
        st.session_state.mode = "daily"
        st.session_state.step = "idle"
        st.session_state.conv_messages = []
        st.rerun()
with col3:
    if st.button("Lookup", use_container_width=True,
                 type="primary" if st.session_state.mode == "lookup" else "secondary"):
        st.session_state.mode = "lookup"
        st.session_state.step = "idle"
        st.session_state.conv_messages = []
        st.rerun()
with col4:
    if st.button("History", use_container_width=True,
                 type="primary" if st.session_state.mode == "history" else "secondary"):
        st.session_state.mode = "history"
        st.session_state.step = "idle"
        st.session_state.conv_messages = []
        st.rerun()

api_name = "GPT-4o" if _openai_client else ("Gemini" if _gemini_model else ("Groq" if _groq_client else "None"))
st.markdown(f"<div style='text-align:center;color:#636e72;font-size:0.75rem;margin:4px 0;'>AI: {api_name}</div>",
            unsafe_allow_html=True)

st.markdown("---")

# ═══════════════════════════════════════════════════════════
# LOOKUP MODE
# ═══════════════════════════════════════════════════════════

if st.session_state.mode == "lookup":
    st.markdown("### Expression Lookup")
    lookup_input = st.text_area("日本語を入力:", key="lookup_input", height=80,
                                placeholder="例: 経過観察する、所見を認める...")
    if st.button("Search", use_container_width=True, type="primary"):
        if lookup_input.strip():
            with st.spinner("Searching..."):
                lookup_result = _chat(LOOKUP_PROMPT.format(query=lookup_input.strip()))
            st.markdown(f'<div class="feedback-box">{lookup_result}</div>',
                        unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# SPEAKING PRACTICE MODE
# ═══════════════════════════════════════════════════════════

if st.session_state.mode in ("ophthalmology", "daily"):

    # ─── New Sentence ──────────────────────────────────────
    if st.session_state.step in ("idle", "feedback_shown", "deep_dive", "conversation"):
        if st.button("New Sentence", use_container_width=True, type="primary"):
            with st.spinner("Generating..."):
                st.session_state.current_prompt = generate_prompt(st.session_state.mode)
                st.session_state.step = "prompt_shown"
                st.session_state.feedback_data = None
                st.session_state.deep_dive_result = None
                st.session_state.conv_messages = []
            st.rerun()

    # ─── Show Prompt ───────────────────────────────────────
    if st.session_state.current_prompt:
        st.markdown(f'<div class="prompt-box">{st.session_state.current_prompt}</div>',
                    unsafe_allow_html=True)

    # ─── Speech Input ──────────────────────────────────────
    if st.session_state.current_prompt and st.session_state.step in ("prompt_shown", "waiting_speech"):
        st.markdown("### Speak your English translation")
        st.markdown("<p style='color:#b2bec3;font-size:0.85rem;'>"
                    "Type or use your keyboard's mic icon for voice input. "
                    "Typos and hesitations will be auto-corrected.</p>",
                    unsafe_allow_html=True)

        user_input = st.text_area("Your English:", key="speech_result", height=100,
                                  label_visibility="collapsed",
                                  placeholder="Type or voice-input here...")
        if st.button("Send", use_container_width=True, type="primary"):
            if user_input.strip():
                st.session_state.raw_speech = user_input.strip()
                st.session_state.step = "processing"
                del st.session_state["speech_result"]
                st.rerun()

    # ─── Processing & Feedback ─────────────────────────────
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
            st.session_state.feedback_data = feedback
            st.session_state.feedback_data["cleaned"] = cleaned
            st.session_state.feedback_data["raw"] = raw
            st.session_state.history.append({
                "prompt": prompt, "raw": raw, "cleaned": cleaned,
                "feedback": feedback, "mode": st.session_state.mode,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            })

        st.session_state.step = "feedback_shown"
        st.rerun()

    # ─── Show Feedback ─────────────────────────────────────
    if st.session_state.step in ("feedback_shown", "deep_dive", "conversation") and st.session_state.feedback_data:
        _save_history_js()
        fb = st.session_state.feedback_data
        gs = fb.get("grammar_score", 0)
        ns = fb.get("natural_score", 0)
        cleaned = fb.get("cleaned", "")

        st.markdown(f"""
        <div class="feedback-box">
            <div style="margin-bottom:8px;">
                <span class="score-badge {score_class(gs)}">Grammar: {gs}/5</span>
                <span class="score-badge {score_class(ns)}">Naturalness: {ns}/5</span>
            </div>
            <p><b>You said:</b> {cleaned}</p>
            <p style="color:#55efc4;"><b>Correction:</b> {fb.get('corrections', '')}</p>
            <p style="color:#ffeaa7;"><b>Why:</b> {fb.get('why_corrections', '')}</p>
            <p style="color:#a29bfe;"><b>Model answer:</b> {fb.get('model_answer', '')}</p>
        </div>
        """, unsafe_allow_html=True)

        # Advanced expression
        adv = fb.get("advanced_expression", "")
        if adv:
            st.markdown(f'<div class="advanced-box"><b>Advanced:</b> {adv}</div>',
                        unsafe_allow_html=True)

        # Key words
        key_words = fb.get("key_words", [])
        if key_words:
            st.markdown("**Key Words:**")
            for kw in key_words:
                if isinstance(kw, dict):
                    st.markdown(f"""<div class="keyword-card">
                        <b>{kw.get('word','')}</b> — {kw.get('meaning','')}
                        <br><span style="color:#b2bec3;font-size:0.85rem;">{kw.get('usage','')}</span>
                    </div>""", unsafe_allow_html=True)

        # TTS for model answer
        model_ans = fb.get('model_answer', '').replace("'", "\\'")
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

        # ─── Deep Dive & Conversation Buttons ──────────────
        if st.session_state.step == "feedback_shown":
            st.markdown("---")
            st.markdown("**Next step:**")
            dc1, dc2 = st.columns(2)

            # Build keyword options for deep dive
            dive_options = []
            for kw in key_words:
                if isinstance(kw, dict) and kw.get("word"):
                    dive_options.append(kw["word"])
            corr = fb.get("corrections", "")
            if corr and corr != "Perfect!":
                dive_options.insert(0, corr)
            if adv:
                dive_options.insert(0, adv)

            with dc1:
                if st.button("Deep Dive", use_container_width=True, type="primary"):
                    st.session_state.step = "deep_dive"
                    st.rerun()
            with dc2:
                if st.button("Conversation", use_container_width=True, type="secondary"):
                    st.session_state.step = "conversation"
                    st.session_state.conv_messages = []
                    st.rerun()

    # ─── Deep Dive Mode ────────────────────────────────────
    if st.session_state.step == "deep_dive" and st.session_state.feedback_data:
        st.markdown("---")
        st.markdown("### Deep Dive")

        fb = st.session_state.feedback_data
        key_words = fb.get("key_words", [])
        dive_options = []
        for kw in key_words:
            if isinstance(kw, dict) and kw.get("word"):
                dive_options.append(kw["word"])
        adv = fb.get("advanced_expression", "")
        if adv:
            dive_options.insert(0, f"[Advanced] {adv[:50]}")
        model_ans = fb.get("model_answer", "")
        if model_ans:
            dive_options.insert(0, f"[Model] {model_ans[:50]}")

        if dive_options:
            selected = st.selectbox("Pick a phrase to deep dive:", dive_options)
            phrase = selected.replace("[Advanced] ", "").replace("[Model] ", "")
        else:
            phrase = st.text_input("Enter a phrase to deep dive:")

        if st.button("Dive In", use_container_width=True, type="primary"):
            if phrase.strip():
                with st.spinner("Diving deep..."):
                    result = _chat(DEEP_DIVE_PROMPT.format(
                        phrase=phrase.strip(),
                        context=st.session_state.current_prompt
                    ))
                st.session_state.deep_dive_result = result

        if st.session_state.deep_dive_result:
            st.markdown(f'<div class="feedback-box">{st.session_state.deep_dive_result}</div>',
                        unsafe_allow_html=True)

        if st.button("Start Conversation", use_container_width=True, type="secondary"):
            st.session_state.step = "conversation"
            st.session_state.conv_messages = []
            st.rerun()

    # ─── Conversation Mode ─────────────────────────────────
    if st.session_state.step == "conversation" and st.session_state.feedback_data:
        st.markdown("---")
        st.markdown("### Free Conversation")
        st.markdown(f"<p style='color:#b2bec3;font-size:0.85rem;'>Topic: {st.session_state.current_prompt[:60]}...</p>",
                    unsafe_allow_html=True)

        fb = st.session_state.feedback_data
        key_phrases = ", ".join([kw.get("word", "") for kw in fb.get("key_words", []) if isinstance(kw, dict)])
        if not key_phrases:
            key_phrases = fb.get("model_answer", "")

        # Show conversation history
        for msg in st.session_state.conv_messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="conv-user">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="conv-ai">{msg["content"]}</div>', unsafe_allow_html=True)

        conv_input = st.text_area("Your message (in English):", key="conv_input", height=80,
                                  label_visibility="collapsed",
                                  placeholder="Type or voice-input here...")
        if st.button("Send message", use_container_width=True):
            if conv_input.strip():
                raw_conv = conv_input.strip()
                with st.spinner("Cleaning up..."):
                    cleaned_conv = _chat(f"Clean up this English speech input. Fix typos, hesitations, and incomplete words. Output ONLY the cleaned sentence:\n{raw_conv}")
                    if cleaned_conv.startswith("Error"):
                        cleaned_conv = raw_conv
                st.session_state.conv_messages.append({"role": "user", "content": cleaned_conv})

                sys_msg = CONVERSATION_SYSTEM.format(
                    topic=st.session_state.current_prompt,
                    phrases=key_phrases
                )
                conv_history = sys_msg + "\n\n"
                for msg in st.session_state.conv_messages:
                    role_label = "User" if msg["role"] == "user" else "Assistant"
                    conv_history += f"{role_label}: {msg['content']}\n"
                conv_history += "Assistant:"

                with st.spinner("..."):
                    reply = _chat(conv_history)

                st.session_state.conv_messages.append({"role": "assistant", "content": reply})
                del st.session_state["conv_input"]
                st.rerun()

        # TTS for last AI message
        if st.session_state.conv_messages and st.session_state.conv_messages[-1]["role"] == "assistant":
            last_msg = st.session_state.conv_messages[-1]["content"].replace("'", "\\'")
            st.html(f"""
            <button onclick="
                var u = new SpeechSynthesisUtterance('{last_msg}');
                u.lang = 'en-US'; u.rate = 0.85;
                speechSynthesis.speak(u);
            " style="background:#0984e3;color:#fff;border:none;border-radius:8px;
                     padding:8px 16px;cursor:pointer;font-size:0.85rem;margin:4px 0;">
                Listen
            </button>
            """)

# ═══════════════════════════════════════════════════════════
# HISTORY MODE
# ═══════════════════════════════════════════════════════════

if st.session_state.mode == "history":
    st.markdown("### Practice History")
    if st.session_state.history:
        _save_history_js()

    if not st.session_state.history:
        st.markdown("<p style='color:#636e72;'>No history yet. Start practicing to build your history!</p>",
                    unsafe_allow_html=True)
    else:
        total = len(st.session_state.history)
        avg_g = sum(h.get("feedback", {}).get("grammar_score", 0) for h in st.session_state.history) / total
        avg_n = sum(h.get("feedback", {}).get("natural_score", 0) for h in st.session_state.history) / total
        st.markdown(f"""<div style="display:flex;gap:12px;margin-bottom:12px;">
            <div style="flex:1;background:#16213e;border-radius:8px;padding:10px;text-align:center;">
                <div style="color:#636e72;font-size:0.75rem;">Total</div>
                <div style="color:#fff;font-size:1.3rem;font-weight:bold;">{total}</div>
            </div>
            <div style="flex:1;background:#16213e;border-radius:8px;padding:10px;text-align:center;">
                <div style="color:#636e72;font-size:0.75rem;">Avg Grammar</div>
                <div style="color:#55efc4;font-size:1.3rem;font-weight:bold;">{avg_g:.1f}/5</div>
            </div>
            <div style="flex:1;background:#16213e;border-radius:8px;padding:10px;text-align:center;">
                <div style="color:#636e72;font-size:0.75rem;">Avg Natural</div>
                <div style="color:#a29bfe;font-size:1.3rem;font-weight:bold;">{avg_n:.1f}/5</div>
            </div>
        </div>""", unsafe_allow_html=True)

        dates = {}
        for h in st.session_state.history:
            ts = h.get("timestamp", "")
            d = ts[:10] if ts else "Unknown"
            dates.setdefault(d, []).append(h)

        for date_key in sorted(dates.keys(), reverse=True):
            entries = dates[date_key]
            with st.expander(f"{date_key} ({len(entries)} sentences)", expanded=(date_key == sorted(dates.keys(), reverse=True)[0])):
                for h in reversed(entries):
                    hfb = h.get("feedback", {})
                    ts = h.get("timestamp", "")
                    time_str = ts[11:] if len(ts) > 11 else ""
                    mode_label = "Ophth" if h.get("mode") == "ophthalmology" else "Daily"
                    gs = hfb.get('grammar_score', 0)
                    ns = hfb.get('natural_score', 0)
                    st.markdown(f"""<div style="background:#16213e;border-radius:8px;padding:10px;margin:6px 0;">
                        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
                            <span style="color:#e94560;font-size:0.75rem;font-weight:bold;">{mode_label}</span>
                            <span style="color:#636e72;font-size:0.7rem;">{time_str}</span>
                        </div>
                        <div style="color:#fff;font-size:0.9rem;margin-bottom:6px;">{h.get('prompt','')}</div>
                        <div style="color:#dfe6e9;font-size:0.85rem;font-style:italic;">You: {h.get('cleaned','')}</div>
                        <div style="display:flex;gap:6px;margin-top:4px;">
                            <span class="score-badge {score_class(gs)}" style="font-size:0.75rem;padding:2px 8px;">G:{gs}/5</span>
                            <span class="score-badge {score_class(ns)}" style="font-size:0.75rem;padding:2px 8px;">N:{ns}/5</span>
                        </div>
                        <div style="color:#55efc4;font-size:0.8rem;margin-top:4px;">Model: {hfb.get('model_answer','')}</div>
                    </div>""", unsafe_allow_html=True)

        if st.button("Clear All History", use_container_width=True, type="secondary"):
            st.session_state.history = []
            st.html("""
            <script>
            try { localStorage.removeItem('esp_history'); } catch(e) {}
            </script>
            """)
            st.rerun()
