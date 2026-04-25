"""
English Speaking Practice App
- Two modes: Ophthalmology English / Daily & Childcare English
- Web Speech API for STT (continuous, trigger-word stop) and TTS
- GPT-4o > Gemini > Groq fallback chain
"""

import streamlit as st
import streamlit.components.v1 as components
import json
import random

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
- 不自然な直訳調や、意味が通らない文は絶対に出力しないでください
- 出力前に、日本語ネイティブが読んで違和感がないか必ず確認してください
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

LOOKUP_PROMPT = """あなたは日英翻訳の専門家です。以下の日本語表現について、自然な英語表現を教えてください。

入力: {query}

以下の形式で回答してください（日本語で説明）：

**自然な英語表現:**
最も一般的で自然な英語表現を1つ

**別の言い方:**
同じ意味の別表現があれば2〜3つ（ニュアンスの違いも簡潔に説明）

**例文:**
実際の使用例を2〜3文（英語 + 日本語訳）
医療・眼科の文脈で使える場合はそちらも含める

**注意点:**
使い分けのポイントや、日本人が間違えやすいポイントがあれば簡潔に"""

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
    # 1) GPT-4o
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
    # 2) Gemini
    if _gemini_model:
        try:
            resp = _gemini_model.generate_content(message)
            return resp.text.strip()
        except Exception:
            pass
    # 3) Groq
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
                "corrections": text, "explanation": "Parse error",
                "better_expression": "", "model_answer": ""}

def score_class(score):
    if score >= 4: return "score-good"
    if score >= 3: return "score-ok"
    return "score-bad"

# ─── UI ─────────────────────────────────────────────────────

# Mode toggle (3 modes)
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Ophthalmology", use_container_width=True,
                 type="primary" if st.session_state.mode == "ophthalmology" else "secondary"):
        st.session_state.mode = "ophthalmology"
        st.session_state.step = "idle"
        st.rerun()
with col2:
    if st.button("Daily", use_container_width=True,
                 type="primary" if st.session_state.mode == "daily" else "secondary"):
        st.session_state.mode = "daily"
        st.session_state.step = "idle"
        st.rerun()
with col3:
    if st.button("Lookup", use_container_width=True,
                 type="primary" if st.session_state.mode == "lookup" else "secondary"):
        st.session_state.mode = "lookup"
        st.session_state.step = "idle"
        st.rerun()

api_name = "GPT-4o" if _openai_client else ("Gemini" if _gemini_model else ("Groq" if _groq_client else "None"))
st.markdown(f"<div style='text-align:center;color:#636e72;font-size:0.75rem;margin:4px 0;'>AI: {api_name}</div>",
            unsafe_allow_html=True)

st.markdown("---")

# ─── Lookup Mode ───────────────────────────────────────────

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

# ─── Speaking Practice Mode ────────────────────────────────

if st.session_state.mode in ("ophthalmology", "daily"):
    if st.button("New Sentence", use_container_width=True, type="primary"):
        with st.spinner("Generating..."):
            st.session_state.current_prompt = generate_prompt(st.session_state.mode)
            st.session_state.step = "prompt_shown"
        st.rerun()

    if st.session_state.current_prompt:
        st.markdown(f'<div class="prompt-box">{st.session_state.current_prompt}</div>',
                    unsafe_allow_html=True)

# ─── Speech Recognition Component ──────────────────────────

if st.session_state.mode in ("ophthalmology", "daily") and st.session_state.current_prompt and st.session_state.step in ("prompt_shown", "waiting_speech"):
    st.markdown("### Speak your English translation")
    st.markdown("<p style='color:#b2bec3;font-size:0.85rem;'>Tap the mic, speak in English, then tap Stop and Submit. "
                "Or type below.</p>",
                unsafe_allow_html=True)

    speech_html = """
    <div id="speech-area" style="text-align:center;">
        <button id="mic-btn" onclick="toggleMic()" style="
            width:80px;height:80px;border-radius:50%;
            background:#e17055;color:white;border:none;
            font-size:2rem;cursor:pointer;margin:10px;
            box-shadow:0 4px 15px rgba(225,112,85,0.4);
        ">&#x1F3A4;</button>
        <div id="status" style="color:#b2bec3;font-size:0.85rem;margin:5px;">Tap to start</div>
        <div id="live-text" style="
            background:#2d3436;border-radius:8px;padding:12px;
            margin:10px 0;min-height:50px;color:#dfe6e9;
            font-size:1rem;text-align:left;
        "></div>
        <button id="submit-btn" onclick="submitToStreamlit()" style="
            display:none;background:#0984e3;color:#fff;border:none;
            border-radius:8px;padding:10px 24px;font-size:1rem;
            cursor:pointer;margin:8px;
        ">Submit</button>
    </div>
    <script>
    var recognition=null, isListening=false, fullTranscript='', interimText='';
    function toggleMic(){
        if(isListening){stopListening();}else{startListening();}
    }
    function startListening(){
        var SR=window.SpeechRecognition||window.webkitSpeechRecognition;
        if(!SR){document.getElementById('status').textContent='Not supported. Use Chrome.';return;}
        recognition=new SR();
        recognition.lang='en-US';recognition.continuous=true;recognition.interimResults=true;
        recognition.onstart=function(){
            isListening=true;fullTranscript='';
            document.getElementById('mic-btn').style.background='#d63031';
            document.getElementById('mic-btn').innerHTML='&#x23F9;';
            document.getElementById('status').textContent='Listening...';
            document.getElementById('submit-btn').style.display='none';
        };
        recognition.onresult=function(e){
            interimText='';
            for(var i=e.resultIndex;i<e.results.length;i++){
                var t=e.results[i][0].transcript;
                if(e.results[i].isFinal){fullTranscript+=t+' ';}
                else{interimText+=t;}
            }
            document.getElementById('live-text').innerHTML=
                '<span style="color:#55efc4;">'+fullTranscript+'</span>'+
                '<span style="color:#636e72;">'+interimText+'</span>';
        };
        recognition.onerror=function(e){
            if(e.error==='no-speech')return;
            document.getElementById('status').textContent='Error: '+e.error;
        };
        recognition.onend=function(){
            if(isListening){try{recognition.start();}catch(ex){}}
        };
        try{recognition.start();}catch(ex){
            document.getElementById('status').textContent='Error: '+ex.message;
        }
    }
    function stopListening(){
        isListening=false;
        if(recognition)recognition.stop();
        document.getElementById('mic-btn').style.background='#e17055';
        document.getElementById('mic-btn').innerHTML='&#x1F3A4;';
        document.getElementById('status').textContent='Tap Submit to send';
        document.getElementById('submit-btn').style.display='inline-block';
    }
    function submitToStreamlit(){
        var text=fullTranscript.trim();
        if(!text)return;
        document.getElementById('status').textContent='Submitting...';
        document.getElementById('submit-btn').style.display='none';
        // Write result into Streamlit's text_area widget
        var textareas=window.parent.document.querySelectorAll('textarea');
        for(var i=0;i<textareas.length;i++){
            if(textareas[i].getAttribute('aria-label')==='speech_result'){
                var nativeSetter=Object.getOwnPropertyDescriptor(
                    window.HTMLTextAreaElement.prototype,'value').set;
                nativeSetter.call(textareas[i],text);
                textareas[i].dispatchEvent(new Event('input',{bubbles:true}));
                break;
            }
        }
        // Click the submit button
        setTimeout(function(){
            var buttons=window.parent.document.querySelectorAll('button');
            for(var i=0;i<buttons.length;i++){
                if(buttons[i].textContent.trim()==='Send'){
                    buttons[i].click();break;
                }
            }
        },300);
    }
    </script>
    """
    components.html(speech_html, height=250)

    # Text input (also receives speech result via JS injection)
    user_input = st.text_area("Your English:", key="speech_result", height=80,
                              label_visibility="collapsed",
                              placeholder="Type or use mic above...")
    if st.button("Send", use_container_width=True):
        if user_input.strip():
            st.session_state.raw_speech = user_input.strip()
            st.session_state.step = "processing"
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
