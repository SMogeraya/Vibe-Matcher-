import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ======================================
# 1️⃣ DATASET
# ======================================
@st.cache_data
def load_data():
    products = [
        {"id": 1, "name": "Boho Maxi Dress", "desc": "Flowy, floral maxi dress in earthy tones for a carefree bohemian look.", "tags": ["boho", "festival", "flowy"]},
        {"id": 2, "name": "Urban Denim Jacket", "desc": "Classic blue denim jacket with a modern cropped fit and street vibe.", "tags": ["urban", "street", "casual"]},
        {"id": 3, "name": "Cozy Cable Knit Sweater", "desc": "Thick cable knit, perfect for cozy evenings or winter outings.", "tags": ["cozy", "warm", "casual"]},
        {"id": 4, "name": "Sporty Running Shoes", "desc": "Breathable, lightweight shoes ideal for daily runs or active days.", "tags": ["sporty", "active", "energetic"]},
        {"id": 5, "name": "Elegant Silk Blouse", "desc": "Soft silk blouse with subtle shimmer for an elegant office or evening look.", "tags": ["elegant", "office", "chic"]},
        {"id": 6, "name": "Minimalist Leather Tote", "desc": "Structured leather tote, sleek and versatile for work or casual outings.", "tags": ["minimal", "office", "urban"]},
        {"id": 7, "name": "Vintage Floral Skirt", "desc": "Retro-inspired midi skirt with floral prints for a timeless charm.", "tags": ["vintage", "romantic", "boho"]},
        {"id": 8, "name": "Edgy Black Biker Jacket", "desc": "Bold leather biker jacket that adds a rebellious edge to any outfit.", "tags": ["edgy", "urban", "bold"]},
        {"id": 9, "name": "Relaxed Linen Pants", "desc": "Lightweight linen trousers with a breezy summer feel and neutral palette.", "tags": ["casual", "relaxed", "summer"]},
        {"id": 10, "name": "Polished Blazer", "desc": "Tailored blazer with clean lines, ideal for business casual looks.", "tags": ["office", "minimal", "formal"]},
        {"id": 11, "name": "Athletic Joggers", "desc": "Comfortable joggers with modern design, perfect for travel or lounging.", "tags": ["sporty", "comfy", "casual"]},
        {"id": 12, "name": "Romantic Lace Dress", "desc": "Delicate lace dress with soft pastels for dreamy date nights.", "tags": ["romantic", "feminine", "elegant"]},
        {"id": 13, "name": "Trendy Crop Top", "desc": "Chic cotton crop top for youthful, energetic street looks.", "tags": ["urban", "trendy", "youthful"]},
        {"id": 14, "name": "Classic Trench Coat", "desc": "Timeless beige trench with belt detailing for a sophisticated vibe.", "tags": ["classic", "formal", "elegant"]},
        {"id": 15, "name": "Comfy Hoodie", "desc": "Soft fleece hoodie that blends comfort with modern street style.", "tags": ["cozy", "urban", "street"]}
    ]
    df = pd.DataFrame(products)
    df["desc_aug"] = df["desc"] + " " + df["tags"].apply(lambda x: " ".join(x))
    return df

df = load_data()

# ======================================
# 2️⃣ MODEL LOADING
# ======================================
@st.cache_resource
def load_model():
    model = SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
    embeddings = model.encode(df["desc_aug"].tolist(), convert_to_numpy=True, show_progress_bar=False)
    return model, embeddings

model, emb_matrix = load_model()

# ======================================
# 3️⃣ UI
# ======================================
st.set_page_config(page_title="Vibe Matcher", page_icon="👗", layout="centered")

st.title("👗 Vibe Matcher — Smart Fashion Recommender")
st.markdown(
    """
    Enter a **vibe** (like *"energetic urban chic"* or *"cozy boho weekend"*)  
    to discover which outfits match that feeling best using semantic embeddings ✨
    """
)

threshold = st.slider("🎯 Similarity Threshold", 0.1, 0.9, 0.3, 0.05)
query = st.text_input("Enter a vibe query:", placeholder="e.g., energetic urban chic")

# ======================================
# 4️⃣ MATCHING FUNCTION
# ======================================
def get_top_matches(query, k=3):
    query_aug = query + " fashion style outfit look"
    query_emb = model.encode([query_aug], convert_to_numpy=True)
    sims = cosine_similarity(query_emb, emb_matrix).flatten()

    # Normalize similarities (0–1 scale)
    norm_sims = (sims - sims.min()) / (sims.max() - sims.min() + 1e-8)

    idx = np.argsort(-norm_sims)[:k]
    top_matches = df.iloc[idx].assign(score=norm_sims[idx])

    # Evaluation metrics
    good = np.sum(norm_sims[idx] > threshold)
    accuracy = round((good / k) * 100, 2)
    mean_sim = round(np.mean(norm_sims[idx]), 3)
    return top_matches, accuracy, mean_sim


if st.button("✨ Match My Vibe") or query:
    if query.strip() == "":
        st.warning("Please enter a vibe description.")
    else:
        with st.spinner("Finding your fashion vibes..."):
            results, accuracy, mean_sim = get_top_matches(query)
            st.subheader(f" Top Matches for: *{query}*")
            st.caption(f"Evaluation Score: **{accuracy}%**  |  Avg Similarity: **{mean_sim}**")

            for _, row in results.iterrows():
                st.markdown(
                    f"""
                    <div style='background-color:#f9f9f9;padding:15px;border-radius:10px;margin-bottom:10px;'>
                        <b>{row['name']}</b><br>
                        <span style='color:#555;font-size:14px;'>{row['desc']}</span><br>
                        <span style='color:#777;font-size:13px;'>Tags: {', '.join(row['tags'])}</span><br>
                        <span style='color:#999;font-size:13px;'>Similarity Score: {row['score']:.3f}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
