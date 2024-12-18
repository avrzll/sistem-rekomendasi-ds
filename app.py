import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer

#<======================================== PAGE CONFIG ========================================>
st.set_page_config(
    page_title="Sistem Rekomendasi Wisata Lumajang",
    page_icon="🏖️",
    layout="wide"
)

st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        padding: 10px 20px;
        border-radius: 4px 4px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .fixed-footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f0f2f6;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 -2px 5px rgba(0,0,0,0.1);
        z-index: 1000;
    }
    .fixed-footer p {
        margin: 0;
    }
    </style>
    """, unsafe_allow_html=True)

#<======================================== HEADER =======================================>
st.title("🏖️ Sistem Rekomendasi Wisata Lumajang")
st.markdown("""
    <div style='
            background-color: #4CAF50;
            padding: 2rem; 
            border-radius: 10px; 
            color: white; 
            margin-bottom: 
            2rem;'>
            <h3 style='margin: 0;'>Temukan Destinasi Wisata Terbaik di Lumajang</h3>
            <p style='margin-top: 1rem;'>Jelajahi berbagai tempat wisata menarik dan dapatkan rekomendasi sesuai preferensi Anda</p>
    </div>
""", unsafe_allow_html=True)

#<======================================== LOAD DATASETS ========================================>
@st.cache_data
def load_data(file_name):
    return pd.read_csv(file_name, skip_blank_lines=True)
places, ratings = load_data(
    "datasets/destination_place.csv").dropna(how='all'), load_data("./datasets/destination_rate_by_user.csv").dropna(how='all')

#<======================================= PREPROCESSING ======================================>
places['category'] = places['category'].fillna("tidak tersedia")
places['destination_name'] = places['destination_name'].fillna("")
ratings["rate_destination"] = ratings["rate_destination"].astype(float)

#<================================== FUNCTION CONTENT BASED ==================================>
# Content-Based Filtering (cosine_similarity)
@st.cache_data
def prepare_content_based():
    tf = TfidfVectorizer()
    tfidf_matrix = tf.fit_transform(places['category'])
    cosine_sim = cosine_similarity(tfidf_matrix)
    cosine_sim_df = pd.DataFrame(
        cosine_sim, index=places['destination_name'], columns=places['destination_name']
    )
    return cosine_sim_df

# Content-Based Filtering (Jaccard Similarity)
@st.cache_data
def calculate_jaccard_similarity(row, selected_keywords):
    keywords = set(row.split())
    selected_set = set(selected_keywords.split())
    intersection = len(keywords.intersection(selected_set))
    union = len(keywords.union(selected_set))
    return intersection / union if union > 0 else 0

#<===================================== SUMMARY DATASETS =====================================>
st.markdown("<div style='margin-bottom: 2rem;'>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
        <div class='metric-card'>
            <h3 style='margin: 0;'>📍 Total Destinasi</h3>
            <h2 style='margin: 10px 0; color: #4CAF50;'>{}</h2>
        </div>
    """.format(len(places)), unsafe_allow_html=True)
with col2:
    st.markdown("""
        <div class='metric-card'>
            <h3 style='margin: 0;'>🏷️ Kategori</h3>
            <h2 style='margin: 10px 0; color: #4CAF50;'>{}</h2>
        </div>
    """.format(len(places['category'].unique())), unsafe_allow_html=True)
with col3:
    st.markdown("""
        <div class='metric-card'>
            <h3 style='margin: 0;'>⭐ Rating Rata-rata</h3>
            <h2 style='margin: 10px 0; color: #4CAF50;'>{:.2f}</h2>
        </div>
    """.format(places['rate'].mean()), unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
    <style>
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Pencarian Wisata (Cosine Similarity)",
    "🗺️ Pencarian Wisata (Jaccard Similarity)",
    "👥 Rekomendasi Personal",
    "📋 Daftar Wisata"
])

#<================================= CBF - COSINE SIMILARITY =================================>
with tab1:
    st.header("🔍 Pencarian dan Rekomendasi Wisata Berdasarkan Cossine Similarity")
    st.write("Metode cosine similarity merupakan metode untuk menghitung kesamaan antara dua buah objek yang dinyatakan dalam dua buah vector dengan menggunakan keywords (kata kunci) dari sebuah dokumen sebagai ukuran")
    search_query = st.text_input(
        "Masukkan nama wisata atau kata kunci", placeholder="Contoh: Pantai, Gunung, dll.")

    num_recommendations = st.slider(
        "Jumlah rekomendasi:", min_value=1, max_value=10, value=5)

    def search_and_recommend(query, places, similarity_data, num_recommendations=num_recommendations):
        matched_places = places[places['destination_name'].str.contains(
            query, case=False, na=False)]

        if matched_places.empty:
            return "Maaf tidak ditemukan wisata yang relevan.", pd.DataFrame()

        recommended_places = (
            similarity_data[matched_places['destination_name'].iloc[0]]
            .sort_values(ascending=False)
            .iloc[1:num_recommendations + 1]
            .index
        )
        return "", places[places['destination_name'].isin(recommended_places)]

    if search_query:
        cosine_sim_df = prepare_content_based()
        message, recommendations = search_and_recommend(
            search_query, places, cosine_sim_df)
        if message:
            st.warning(message)
        else:
            st.subheader("✨ Rekomendasi Tempat Wisata")
            for _, row in recommendations.iterrows():
                with st.container():
                    st.markdown(f"""
                        <div style='
                                background-color: white;
                                padding: 1rem;
                                border-radius: 10px;
                                margin-bottom: 1rem;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                                <h4 style='margin: 0; color: #4CAF50;'>{row['destination_name']}</h4>
                                <p style='margin: 0.5rem 0;'>⭐ {row['rate']:.2f} | 🏷️ {row['category']} | 💰 Rp {row['price']:,.0f}</p>
                        </div>
                    """, unsafe_allow_html=True)

#<================================= CBF - JACCARD SIMILARITY ================================>
with tab2:
    st.header("🗺️ Rekomendasi Tempat Wisata Berdasarkan Jaccard Similarity")
    st.write("Jaccard Similarity merupakan algoritma yang berfugsi untuk membandingkan antar dua dokumen dengan menghitung kemeripan atau perbedaan dari beberapa objek.")
    selected_place = st.selectbox("Pilih Tempat Wisata untuk Melihat Rekomendasi", places['destination_name'].unique())

    num_recommendations = st.slider("Jumlah Rekomendasi:", min_value=1, max_value=10, value=5)

    if selected_place:
        selected_keywords = places.loc[places['destination_name'] == selected_place, 'category'].values[0]

        places['Jaccard_Score'] = places['category'].apply(lambda x: calculate_jaccard_similarity(x, selected_keywords))

        recommended_places = places.sort_values(by='Jaccard_Score', ascending=False).head(num_recommendations)

        st.subheader("✨ Rekomendasi Berdasarkan Jaccard Similarity")
        for _, row in recommended_places.iterrows():
            with st.container():
                st.markdown(f"""
                    <div style='
                            background-color: white; 
                            padding: 1rem; 
                            border-radius: 10px; 
                            margin-bottom: 1rem; 
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                            <h4 style='margin: 0; color: #4CAF50;'>{row['destination_name']}</h4>
                            <p style='margin: 0.5rem 0;'>⭐ {row['rate']:.2f} | 🏷️ {row['category']} | 💰 Rp {row['price']:,.0f}</p>
                            <p>Jaccard Similarity: {row['Jaccard_Score']:.2f}</p>
                    </div>
                """, unsafe_allow_html=True)

#<============================== COLLABORATIVE FILTERING (SVD) ==============================>
with tab3:
    # Collaborative Filtering
    if {"id_user", "id_destination", "rate_destination"}.issubset(ratings.columns):
        pivot_tb = ratings.pivot_table(index="id_user", columns="id_destination", values="rate_destination")

        imputer = SimpleImputer(strategy="constant", fill_value=0)
        pivot_tb_filled = imputer.fit_transform(pivot_tb)

        svd = TruncatedSVD(n_components=20)
        latent_matrix = svd.fit_transform(pivot_tb_filled)
        
        item_similarity = cosine_similarity(svd.components_.T)
        
        dest_to_index = {dest_id: idx for idx, dest_id in enumerate(pivot_tb.columns)}
        index_to_dest = {idx: dest_id for dest_id, idx in dest_to_index.items()}

        id_usr = int(st.number_input("Masukkan ID User: ", min_value=0, step=1))

        if id_usr in pivot_tb.index:
            user_ratings = pivot_tb.loc[id_usr]
            rated_items = user_ratings[user_ratings > 0].index.to_list()
            
            rated_indices = [dest_to_index[dest] for dest in rated_items if dest in dest_to_index]
            
            recommendations = {}
            for id_dest in pivot_tb.columns:
                if id_dest not in rated_items:
                    if id_dest in dest_to_index:
                        dest_index = dest_to_index[id_dest]
                        sim_scores = item_similarity[dest_index, :]
                        sim_ratings = [sim_scores[idx] for idx in rated_indices]
                        weighted_ratings = np.dot(sim_ratings, user_ratings[rated_items])
                        sim_sum = np.sum(sim_ratings)
                        predicted_rating = weighted_ratings / sim_sum if sim_sum > 0 else 0
                        recommendations[id_dest] = predicted_rating
                    
            sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
            
            recommendations_df = pd.DataFrame(sorted_recommendations, columns=["id_destination", "predicted_rating"])
            
            recommendations_with_details = pd.merge(
                    recommendations_df,
                    places,
                    on="id_destination",
                    how="inner"
                )
            
            st.subheader("Rekomendasi Destinasi Wisata")
            if not recommendations_with_details.empty:
                for _, row in recommendations_with_details.iterrows():
                    with st.container():
                        st.markdown(f"""
                            <div style='
                                    background-color: white;
                                    padding: 1rem;
                                    border-radius: 10px;
                                    margin-bottom: 1rem;
                                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                                    <h4 style='margin: 0; color: #4CAF50;'>{row['destination_name']}</h4>
                                    <small style='margin: 0; color: #4CAF50;'>{row['city']}</small>
                                    <p style='margin: 0.5rem 0;'>⭐ {row['rate']:.2f} | 🏷️ {row['category']} | 💰 Rp {row['price']:,.0f}</p>
                            </div>
                        """, unsafe_allow_html=True)
            else:
                st.write("Tidak ada rekomendasi yang tersedia.")
        else:
            st.error(f"User ID {id_usr} tidak ditemukan dalam dataset.")
    else:
        st.error("Dataset harus memiliki kolom: id_user, id_destination, rate_destination")

#<===================================== FILTER KATEGORI =====================================>
with tab4:
    st.header("📋 Daftar Wisata")
    col1, col2 = st.columns([1, 2])
    with col1:
        kategori = st.selectbox(
            "Pilih Kategori", ["Semua"] + sorted(places['category'].unique().tolist()))

    if kategori == "Semua":
        filtered_places = places
    else:
        filtered_places = places[places['category'] == kategori]

    st.subheader(f"🗺️ Tempat Wisata - {kategori}")
    for _, row in filtered_places.iterrows():
        with st.container():
            st.markdown(f"""
                <div style='
                        background-color: white; 
                        padding: 1rem; 
                        border-radius: 10px; 
                        margin-bottom: 1rem; 
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                        <h4 style='margin: 0; color: #4CAF50;'>{row['destination_name']}</h4>
                        <p style='margin: 0.5rem 0;'>⭐ {row['rate']:.2f} | 🏷️ {row['category']} | 💰 Rp {row['price']:,.0f}</p>
                </div>
            """, unsafe_allow_html=True)

#<========================================= FOOTER ========================================>
st.markdown("""
<div class='fixed-footer'>
    <p>Copyright © 2024 | Sistem Rekomendasi Wisata Lumajang</p>
</div>
""", unsafe_allow_html=True)
