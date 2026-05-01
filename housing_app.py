import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from PIL import Image
import warnings
import time
import os
import pickle
import json

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

 
st.set_page_config(
    page_title="AI House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    .custom-card {
        background: rgba(255,255,255,0.96);
        border-radius: 20px;
        padding: 1.8rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.12);
        margin-bottom: 1.2rem;
    }
    .animated-header {
        text-align: center;
        font-size: 2.8rem;
        font-weight: 800;
        color: white;
        margin-bottom: 0.4rem;
        letter-spacing: -0.5px;
    }
    .price-display {
        text-align: center;
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0.8rem 0;
    }
    .stat-card {
        background: rgba(255,255,255,0.92);
        border-radius: 16px;
        padding: 1rem 0.5rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        transition: transform 0.25s ease;
    }
    .stat-card:hover { transform: translateY(-4px); }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        border-radius: 12px !important;
        width: 100% !important;
        padding: 0.7rem 1.5rem !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(0,0,0,0.2) !important;
    }
    .result-box {
        background: linear-gradient(135deg,#667eea15,#764ba215);
        border: 1.5px solid #667eea40;
        border-radius: 16px;
        padding: 2.5rem 1.5rem;
        text-align: center;
    }
    .footer {
        text-align: center;
        padding: 1.5rem;
        background: rgba(0,0,0,0.08);
        border-radius: 12px;
        margin-top: 2rem;
        color: rgba(255,255,255,0.85);
    }
    @media (max-width: 768px) {
        .animated-header { font-size: 1.8rem; }
        .price-display   { font-size: 2.2rem; }
    }
</style>
""", unsafe_allow_html=True)
 


IMG_SIZE      = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])
 
# REQUIRED FILES CHECK — fail fast with clear message
 
REQUIRED_FILES = {
    'model_fusion_final.keras':  'Fusion model (from Google Drive)',
    'model_tabular_final.keras': 'Tabular model (from Google Drive)',
    'housing_clean.csv':         'Clean dataset (from local preprocessing)',
    'scaler.pkl':            'Fitted scaler (from Colab save step)',
    'zipcode_map.json':          'Zipcode map (from Colab save step)',
}

missing = [f"{v} → {k}" for k, v in REQUIRED_FILES.items() if not os.path.exists(k)]
if missing:
    st.error("❌ Missing required files:")
    for m in missing:
        st.markdown(f"- `{m}`")
    st.info("Make sure all files are in the same folder as housing_app.py")
    st.stop()
 
# LOAD MODELS — @cache_resource = loads only once per session
 
@st.cache_resource(show_spinner=False)
def load_everything():
    # Load Keras models (compile=False — faster load, we only predict)
    fusion_model  = tf.keras.models.load_model(
        'model_fusion_final.keras',  compile=False
    )
    tabular_model = tf.keras.models.load_model(
        'model_tabular_final.keras', compile=False
    )

    # ResNet50 — frozen, only for feature extraction at inference time
    resnet = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling='avg'           # Output: (batch, 2048)
    )
    resnet.trainable = False


    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # Load zipcode → mean price map
    with open('zipcode_map.json', 'r') as f:
        raw = json.load(f)
    zipcode_map = {int(k): float(v) for k, v in raw.items()}

    # Load dataset (for market position context only)
    df = pd.read_csv('housing_clean.csv')

    return fusion_model, tabular_model, resnet, scaler, zipcode_map, df

 
# IMAGE HELPERS
 
def safe_open_image(uploaded_file):
    """Open uploaded file as PIL Image. Returns None on failure."""
    try:
        return Image.open(uploaded_file).convert('RGB')
    except Exception:
        return None

def preprocess_image(pil_img):
    img = pil_img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr

def extract_features(resnet, pil_images):
    
    frames = []
    for img in pil_images:
        if img is not None:
            frames.append(preprocess_image(img))
        else:
            frames.append(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32))

    batch = np.stack(frames, axis=0)          # (4, 224, 224, 3)
    feats = resnet.predict(batch, verbose=0)  # (4, 2048)
    return feats.mean(axis=0)                  # (2048,)

 
def make_prediction(fusion_m, tabular_m, resnet,
                    scaler, zipcode_map, df,
                    bedrooms, bathrooms, area, zipcode,
                    uploads):
    """
    Run full prediction pipeline.
    Returns (predicted_price, model_label, error_message).
    predicted_price is None on failure.
    """
    has_images = any(u is not None for u in uploads)

    #   Build tabular feature vector  
    zipcode_enc = zipcode_map.get(
        int(zipcode),
        float(np.mean(list(zipcode_map.values())))
    )
    raw = np.array([[bedrooms, bathrooms, float(area), zipcode_enc]],
                   dtype=np.float32)

    X_tab = scaler.transform(raw)           # (1, 4)

    # Safety: check for NaN / Inf after scaling
    if not np.isfinite(X_tab).all():
        return None, None, "Tabular features contain Inf/NaN after scaling."

    #   Predict  
    if has_images:
        pil_imgs = [safe_open_image(u) for u in uploads]
        img_feat = extract_features(resnet, pil_imgs)

        if not np.isfinite(img_feat).all():
            return None, None, "Image features are invalid — try different images."

        X_img    = img_feat.reshape(1, -1)   # (1, 2048)
        log_pred = fusion_m.predict([X_tab, X_img], verbose=0)[0][0]
        label    = "Fusion (Images + Tabular)"
    else:
        log_pred = tabular_m.predict(X_tab, verbose=0)[0][0]
        label    = "Tabular Only"

    # Safety: check log prediction
    if not np.isfinite(log_pred):
        return None, None, "Model returned Inf/NaN. Check inputs."

    price = float(np.exp(log_pred))

    # Sanity range: $10k – $50M
    if price < 10_000 or price > 50_000_000:
        return None, None, (
            f"Prediction ${price:,.0f} is outside plausible range. "
            "Check area, zipcode and images."
        )

    return price, label, None

 

# Header
st.markdown("""
<div class="animated-header">🏠 AI House Price Predictor</div>
<div style="text-align:center; margin-bottom:2rem;">
    <span style="background:rgba(255,255,255,0.18); padding:0.45rem 1.1rem;
                 border-radius:20px; font-size:0.88rem; color:white;">
        🤖 Multimodal AI &nbsp;|&nbsp; ResNet50 + Deep Learning
        &nbsp;|&nbsp; Real-time Predictions
    </span>
</div>
""", unsafe_allow_html=True)

# Load models — spinner shown outside cache function
with st.spinner("🔄 Loading neural networks..."):
    (fusion_m, tabular_m, resnet,
     scaler, zipcode_map, df) = load_everything()

success_box = st.empty()
success_box.success("✅ Models loaded — ready to predict!")
time.sleep(0.8)
success_box.empty()

 
# STAT CARDS 
 
s1, s2, s3, s4 = st.columns(4)
for col, icon, val, label in zip(
    [s1, s2, s3, s4],
    ["🏘️", "📸",   "💰",      "⚡"],
    ["535","2,140","$172k",   "~1s"],
    ["Properties","Images","Best MAE","Inference"]
):
    with col:
        st.markdown(f"""
        <div class="stat-card">
            <div style="font-size:2rem;">{icon}</div>
            <div style="font-size:1.25rem;font-weight:700;">{val}</div>
            <div style="color:gray;font-size:0.82rem;">{label}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("---")

# TWO-COLUMN LAYOUT
left_col, right_col = st.columns([1, 1], gap="large")

 
# LEFT — INPUTS
 
with left_col:

    # Property details
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.subheader("📊 Property Details")

    t_input, t_tips = st.tabs(["📝 Input Fields", "💡 Tips"])

    with t_input:
        bedrooms  = st.slider("🛏️ Bedrooms",  1,   10,  3)
        bathrooms = st.slider("🚿 Bathrooms", 1.0, 8.0, 2.0, step=0.5)
        area      = st.number_input(
            "📐 Area (sqft)",
            min_value=300, max_value=15000,
            value=1800, step=50
        )
        zipcode = st.selectbox(
            "📍 Zipcode",
            sorted(zipcode_map.keys()),
            help="Location is the strongest price driver"
        )

        # Area avg price for selected zipcode (real data)
        area_avg = df[df['zipcode'] == zipcode]['price'].mean()
        if pd.notna(area_avg):
            st.caption(
                f"📍 Avg price in {zipcode}: **${area_avg:,.0f}**"
            )

    with t_tips:
        st.markdown("""
        <div style="background:#f8f9ff;padding:1rem;border-radius:12px;">
        <b>🎯 For best predictions:</b><br><br>
        📸 Upload all 4 images → activates Fusion model<br>
        🏡 Use well-lit, clear photos<br>
        📐 Double-check area in sqft<br>
        📍 Verify zipcode is correct<br>
        🔄 Missing images are handled automatically
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Image uploads
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.subheader("📸 Property Images")
    st.caption("Optional — upload to activate Fusion AI model")

    ca, cb = st.columns(2)
    with ca:
        up_bedroom = st.file_uploader(
            "🛏️ Bedroom",  type=['jpg','jpeg','png'], key='bed'
        )
        up_kitchen = st.file_uploader(
            "🍳 Kitchen",  type=['jpg','jpeg','png'], key='kit'
        )
    with cb:
        up_bathroom = st.file_uploader(
            "🚿 Bathroom", type=['jpg','jpeg','png'], key='bath'
        )
        up_frontal = st.file_uploader(
            "🏡 Frontal",  type=['jpg','jpeg','png'], key='front'
        )

    uploads      = [up_bedroom, up_bathroom, up_kitchen, up_frontal]
    upload_count = sum(1 for u in uploads if u is not None)
    has_images   = upload_count > 0

    if upload_count == 4:
        st.success("📸 4/4 images uploaded — Fusion AI activated!")
    elif upload_count > 0:
        st.info(f"📸 {upload_count}/4 images — Fusion AI partially active")
    else:
        st.info("💡 No images — Tabular model will be used")

    st.markdown('</div>', unsafe_allow_html=True)

 
# RIGHT — RESULTS
 
with right_col:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.subheader("🔮 Prediction Result")

    # Image preview strip
    if has_images:
        st.markdown("**🖼️ Image Preview**")
        pcols = st.columns(4)
        for i, (up, lbl) in enumerate(
            zip(uploads, ['Bedroom','Bathroom','Kitchen','Frontal'])
        ):
            with pcols[i]:
                if up:
                    img = safe_open_image(up)
                    if img:
                        st.image(img, caption=lbl, use_container_width=True)
                    else:
                        st.warning(f"❌ {lbl}")
                else:
                    st.markdown(f"""
                    <div style="background:#f0f2f6;padding:1.2rem 0.2rem;
                                text-align:center;border-radius:10px;
                                font-size:0.8rem;color:gray;">
                        📷<br>{lbl}<br>Missing
                    </div>""", unsafe_allow_html=True)
        st.markdown("---")

    #  PREDICT BUTTON  
    if st.button("🚀 Generate Price Prediction",
                 type="primary", use_container_width=True):

        # UX progress animation
        bar    = st.progress(0)
        status = st.empty()
        msgs   = ["🔍 Analyzing features...",
                  "🧠 Running neural network...",
                  "🎯 Computing market value...",
                  "✨ Finalizing prediction..."]
        for i in range(100):
            bar.progress(i + 1)
            status.text(msgs[min(i // 25, 3)])
            time.sleep(0.006)
        bar.empty()
        status.empty()

        # Run prediction
        price, model_label, error = make_prediction(
            fusion_m, tabular_m, resnet,
            scaler, zipcode_map, df,
            bedrooms, bathrooms, area, zipcode,
            uploads
        )

        if error:
            st.error(f"❌ Prediction Error: {error}")
            st.info("Check your inputs and try again.")

        else:
            #  PRICE  
            st.markdown(
                f'<div class="price-display">${price:,.0f}</div>',
                unsafe_allow_html=True
            )

            margin = 0.15 if has_images else 0.22
            low, high = price * (1 - margin), price * (1 + margin)
            st.markdown(f"""
            <p style="text-align:center;color:#555;font-size:0.95rem;">
                Estimated range: <b>${low:,.0f}</b> — <b>${high:,.0f}</b>
            </p>""", unsafe_allow_html=True)

            st.markdown("---")

            #  METRICS  
            m1, m2, m3 = st.columns(3)
            with m1:
                icon = "🔀" if has_images else "📊"
                st.metric("Model", f"{icon} {'Fusion' if has_images else 'Tabular'}")
            with m2:
                st.metric("Price / sqft", f"${price/area:,.0f}")
            with m3:
                st.metric("Images Used", f"{upload_count}/4")

            #   DETAILED REPORT  
            with st.expander("📋 Full Analysis Report", expanded=False):
                r1, r2 = st.columns(2)
                with r1:
                    st.markdown("**🏠 Property Specs**")
                    st.write(f"Bedrooms:  {bedrooms}")
                    st.write(f"Bathrooms: {bathrooms}")
                    st.write(f"Area:      {area:,} sqft")
                    st.write(f"Zipcode:   {zipcode}")
                with r2:
                    st.markdown("**📊 Analysis**")
                    st.write(f"Price/sqft:  ${price/area:,.0f}")
                    st.write(f"Model used:  {model_label}")
                    st.write(f"Images used: {upload_count}/4")
                    area_avg = df[df['zipcode'] == zipcode]['price'].mean()
                    if pd.notna(area_avg) and area_avg > 0:
                        diff = ((price - area_avg) / area_avg) * 100
                        st.write(f"vs area avg: {diff:+.1f}%")

            # ---- MARKET POSITION ----
            st.markdown("**📈 Market Position**")
            min_p = df['price'].min()
            max_p = df['price'].max()
            pct   = (price - min_p) / (max_p - min_p)
            pct   = float(np.clip(pct, 0.0, 1.0))
            st.progress(pct)
            st.caption(
                f"${min_p/1000:.0f}k (lowest) ←————→ ${max_p/1000:.0f}k (highest)"
            )

            if pct > 0.75:
                st.success("💎 Premium property — top 25% of market")
            elif pct > 0.50:
                st.info("📈 Mid-range property — solid investment")
            elif pct > 0.25:
                st.warning("💰 Value property — good entry point")
            else:
                st.error("🏚️ Below average — potential renovation opportunity")

    else:
        # Placeholder 
        st.markdown("""
        <div class="result-box">
            <div style="font-size:3.5rem;margin-bottom:0.8rem;">🏠</div>
            <div style="font-size:1.25rem;font-weight:700;color:#444;">
                Ready for Prediction
            </div>
            <div style="color:#888;margin-top:0.5rem;font-size:0.9rem;">
                Fill in property details on the left<br>
                and click the button above
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


st.markdown("""
<div class="footer">
    <div style="font-size:0.9rem;">
        🚀 Powered by ResNet50 + Deep Neural Networks
        &nbsp;|&nbsp; 535 Properties &nbsp;|&nbsp; 2,140 Images
    </div>
    <div style="font-size:0.78rem;margin-top:0.4rem;
                color:rgba(255,255,255,0.65);">
        Multimodal ML Project — Images + Tabular Data Fusion
    </div>
</div>
""", unsafe_allow_html=True)

 
# SIDEBAR — real metrics only

with st.sidebar:
    st.markdown("## 🧠 Model Info")
    st.markdown("---")

    st.markdown("### 🤖 Architecture")
    st.markdown("""
    - **Image:** ResNet50 (ImageNet weights)
    - **Tabular:** 4-layer Dense Network
    - **Fusion:** Concatenate + Dense + Dropout
    - **Output:** Log-price regression
    """)

    st.markdown("---")
    st.markdown("### 📈 Actual Test Performance")
    st.caption("Results on held-out test set (535 houses)")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Tabular MAE",  "$172k")
        st.metric("Tabular MAPE", "33.4%")
    with c2:
        st.metric("Fusion MAE",   "$202k")
        st.metric("Fusion MAPE",  "39.9%")

    st.info(
        "Tabular model performs better — "
        "expected for a 535-sample dataset. "
        "Fusion demonstrates multimodal ML pipeline."
    )

    st.markdown("---")
    st.markdown("### 📌 Dataset")
    st.markdown("""
    - 535 houses (Arizona + California)
    - 2,140 images (4 per house)
    - Features: beds, baths, area, zipcode
    - Price: $22k — $5.8M
    """)

    st.markdown("---")
    st.markdown("### 💡 Usage Tips")
    st.markdown("""
    1. Upload all 4 room photos
    2. Use well-lit, clear images
    3. Verify area in sqft
    4. Select the correct zipcode
    5. Missing images auto-handled
    """)