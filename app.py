import numpy as np
import joblib
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Digit Recognizer", page_icon="🔢", layout="centered")

# ---------------------------------------------------------------
# Style: clean light UI with a colorful gradient identity
# ---------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #f4f6fb;
    --card: #ffffff;
    --text: #1e293b;
    --muted: #64748b;
    --indigo: #4f46e5;
    --violet: #7c3aed;
    --pink: #ec4899;
    --orange: #f97316;
    --line: #e2e8f0;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background: var(--bg) !important;
}
[data-testid="stAppViewContainer"] * {
    font-family: 'Poppins', 'Segoe UI', sans-serif;
    color: var(--text);
}
#MainMenu, footer { visibility: hidden; }
.block-container { max-width: 800px; padding-top: 1.6rem; padding-bottom: 3rem; }

/* Hero banner */
.hero {
    background: linear-gradient(120deg, #4f46e5 0%, #7c3aed 45%, #ec4899 80%, #f97316 100%);
    border-radius: 22px;
    padding: 2rem 2.2rem;
    margin-bottom: 1.4rem;
    box-shadow: 0 14px 30px rgba(124, 58, 237, 0.28);
}
.hero h1 { margin: 0; font-size: 2.2rem; font-weight: 800; color: #fff !important; line-height: 1.15; }
.hero p  { margin: .5rem 0 0 0; color: rgba(255,255,255,.92) !important; font-size: 1rem; }
.hero .author { margin-top: .7rem; font-size: .95rem; }
.hero .author b { color: #fff !important; font-weight: 700; }
.chips { margin-top: 1rem; display: flex; gap: .5rem; flex-wrap: wrap; }
.chip {
    background: rgba(255,255,255,.2);
    color: #fff !important;
    padding: .25rem .8rem;
    border-radius: 999px;
    font-size: .82rem;
    font-weight: 500;
}

/* Tabs as pills */
div[data-baseweb="tab-list"] { gap: .5rem; border-bottom: none !important; }
button[data-baseweb="tab"] {
    background: #fff !important;
    border: 1px solid var(--line) !important;
    border-radius: 999px !important;
    padding: .45rem 1.3rem !important;
    font-weight: 600;
}
button[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(120deg, var(--indigo), var(--violet)) !important;
    border-color: transparent !important;
}
button[data-baseweb="tab"][aria-selected="true"] * { color: #fff !important; }
div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] { display: none !important; }

/* Canvas */
iframe {
    border: 4px solid var(--indigo) !important;
    border-radius: 16px;
    box-shadow: 0 10px 24px rgba(79, 70, 229, .22);
}

/* Upload box */
[data-testid="stFileUploaderDropzone"] {
    background: #fff !important;
    border: 2px dashed var(--violet) !important;
    border-radius: 16px;
}
[data-testid="stFileUploaderDropzone"] button {
    background: linear-gradient(120deg, var(--indigo), var(--violet)) !important;
    border: none !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploaderDropzone"] button * { color: #fff !important; font-weight: 600; }

/* Tips card */
.tips {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 1rem 1.2rem;
    box-shadow: 0 4px 14px rgba(15, 23, 42, .05);
}
.tips b { font-size: 1rem; }
.tips ul { margin: .5rem 0 0 1.1rem; padding: 0; color: var(--muted); font-size: .9rem; line-height: 1.7; }
.tips li, .tips li * { color: var(--muted) !important; }

/* Result card */
.result {
    display: flex;
    align-items: center;
    gap: 1.6rem;
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 20px;
    padding: 1.2rem 1.8rem;
    margin: 1.4rem 0 1rem 0;
    box-shadow: 0 10px 28px rgba(15, 23, 42, .08);
}
.result .digit {
    font-size: 5.6rem;
    font-weight: 800;
    line-height: 1;
    min-width: 1.3ch;
    text-align: center;
    -webkit-text-fill-color: transparent;
}
.result .label { color: var(--muted) !important; font-size: .9rem; }
.result .conf  { font-size: 1.8rem; font-weight: 700; }
.badge {
    display: inline-block;
    margin-top: .3rem;
    padding: .15rem .7rem;
    border-radius: 999px;
    font-size: .78rem;
    font-weight: 600;
}
.badge.high { background: #dcfce7; color: #166534 !important; }
.badge.mid  { background: #fef3c7; color: #92400e !important; }
.badge.low  { background: #fee2e2; color: #991b1b !important; }

/* Probability bars */
.panel {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 1rem 1.4rem;
    box-shadow: 0 4px 14px rgba(15, 23, 42, .05);
}
.panel h4 { margin: 0 0 .6rem 0; font-size: 1rem; font-weight: 700; }
.bar-row { display: flex; align-items: center; gap: .8rem; margin: .5rem 0; }
.bar-row .d { width: 1.4rem; text-align: center; font-weight: 700; font-size: 1.1rem; }
.bar-row .track { flex: 1; height: 12px; background: #eef2f7; border-radius: 999px; overflow: hidden; }
.bar-row .fill  { height: 100%; border-radius: 999px; }
.bar-row .p { width: 3.4rem; text-align: right; color: var(--muted) !important; font-size: .9rem; font-weight: 500; }

.empty {
    background: var(--card);
    border: 2px dashed #cbd5e1;
    border-radius: 18px;
    padding: 1.6rem;
    text-align: center;
    color: var(--muted) !important;
    margin-top: 1.2rem;
}

[data-testid="stExpander"] {
    background: var(--card) !important;
    border: 1px solid var(--line) !important;
    border-radius: 14px;
    margin-top: 1rem;
}

.foot { text-align: center; color: var(--muted) !important; font-size: .82rem; margin-top: 2rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# One color per digit 0-9
COLORS = ["#ef4444", "#f97316", "#f59e0b", "#84cc16", "#10b981",
          "#06b6d4", "#3b82f6", "#6366f1", "#a855f7", "#ec4899"]


# ---------------------------------------------------------------
# 1. Load the trained model only once
# ---------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("mnist_model.pkl")


model = load_model()


# ---------------------------------------------------------------
# 2. Prepare digit image for MNIST
# ---------------------------------------------------------------
def prepare_digit(gray):
    gray = np.asarray(gray, dtype=np.float32)

    if gray.max() < 30:
        return None

    rows = np.where(gray.max(axis=1) > 30)[0]
    cols = np.where(gray.max(axis=0) > 30)[0]
    digit = gray[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

    h, w = digit.shape
    scale = 20.0 / max(h, w)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))

    digit = np.array(
        Image.fromarray(digit.astype(np.uint8)).resize((new_w, new_h), Image.LANCZOS),
        dtype=np.float32,
    )

    box = np.zeros((28, 28), dtype=np.float32)
    top = (28 - new_h) // 2
    left = (28 - new_w) // 2
    box[top:top + new_h, left:left + new_w] = digit

    total = box.sum()
    if total > 0:
        r, c = np.indices(box.shape)
        dy = int(round(14 - (r * box).sum() / total))
        dx = int(round(14 - (c * box).sum() / total))
        box = np.roll(np.roll(box, dy, axis=0), dx, axis=1)

    return (box / 255.0).reshape(1, 784)


# ---------------------------------------------------------------
# 3. Header
# ---------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>Handwritten Digit Recognizer</h1>
        <p>Write a number from 0 to 9 and the model reads it instantly.</p>
        <p class="author">Developed by <b>Muntazir Jumani</b></p>
        <div class="chips">
            <span class="chip">Python</span>
            <span class="chip">Machine Learning</span>
            <span class="chip">MNIST dataset</span>
            <span class="chip">Streamlit</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_draw, tab_upload = st.tabs(["✏️ Draw a digit", "📤 Upload an image"])
gray = None

# ---------------------------------------------------------------
# 4. Draw
# ---------------------------------------------------------------
with tab_draw:
    left_col, right_col = st.columns([3, 2])

    with left_col:
        canvas = st_canvas(
            stroke_width=18,
            stroke_color="#FFFFFF",
            background_color="#000000",
            width=300,
            height=300,
            drawing_mode="freedraw",
            return_image_data=True,
            key="canvas",
        )

    with right_col:
        st.markdown(
            """
            <div class="tips">
                <b>Tips for best results</b>
                <ul>
                    <li>Draw one digit only</li>
                    <li>Make it big, in the center</li>
                    <li>Use the 🗑️ icon to clear the board</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if canvas.image_data is not None:
        gray = np.array(canvas.image_data)[:, :, 0]

# ---------------------------------------------------------------
# 5. Upload
# ---------------------------------------------------------------
with tab_upload:
    file = st.file_uploader("Choose a PNG or JPG image of one digit", type=["png", "jpg", "jpeg"])

    if file is not None:
        img = np.array(Image.open(file).convert("L"), dtype=np.float32)
        if img.mean() > 127:  # dark ink on white paper -> white on black
            img = 255 - img
        gray = img
        st.image(file, caption="Your image", width=200)

# ---------------------------------------------------------------
# 6. Prediction
# ---------------------------------------------------------------
x = prepare_digit(gray) if gray is not None else None

if x is None:
    st.markdown(
        '<div class="empty">No digit yet. Draw one on the board or upload an image.</div>',
        unsafe_allow_html=True,
    )
else:
    probs = model.predict_proba(x)[0]
    classes = [int(c) for c in model.classes_]
    order = np.argsort(probs)[::-1]
    best = order[0]
    digit = classes[best]
    conf = float(probs[best])
    color = COLORS[digit % 10]

    if conf >= 0.85:
        badge = '<span class="badge high">High confidence</span>'
    elif conf >= 0.5:
        badge = '<span class="badge mid">Medium confidence</span>'
    else:
        badge = '<span class="badge low">Low confidence: draw bigger and centered</span>'

    st.markdown(
        f"""
        <div class="result">
            <div class="digit" style="background: linear-gradient(135deg, {color}, #4f46e5); -webkit-background-clip: text; background-clip: text;">{digit}</div>
            <div>
                <div class="label">Predicted digit</div>
                <div class="conf">{conf:.0%} sure</div>
                {badge}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    rows_html = ""
    for i in order[:3]:
        p = float(probs[i])
        c = COLORS[classes[i] % 10]
        rows_html += (
            f'<div class="bar-row"><div class="d" style="color:{c} !important">{classes[i]}</div>'
            f'<div class="track"><div class="fill" style="width:{p * 100:.1f}%; background:{c}"></div></div>'
            f'<div class="p">{p:.0%}</div></div>'
        )
    st.markdown(f'<div class="panel"><h4>Top 3 guesses</h4>{rows_html}</div>', unsafe_allow_html=True)

    with st.expander("See what the model sees"):
        c1, c2 = st.columns([1, 2])
        with c1:
            st.image(x.reshape(28, 28), caption="28×28 input", width=140, clamp=True)
        with c2:
            st.bar_chart({"probability": {str(c): float(p) for c, p in zip(classes, probs)}})

st.markdown('<div class="foot">Built by Muntazir Jumani with Python, scikit-learn and Streamlit</div>', unsafe_allow_html=True)