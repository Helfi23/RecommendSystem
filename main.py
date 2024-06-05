import streamlit as st
import mysql.connector
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pyngrok import ngrok

# Fungsi untuk memuat data Collaborative Filtering (CF)
def load_data_CF():
    try:
        conn = mysql.connector.connect(user='root', host='localhost', database='db_course')
        cursor = conn.cursor()
        cursor.execute("""
        SELECT DISTINCT r.user_id AS user_id, r.course_id AS course_id, r.rating AS rating
                       FROM histories h 
                       JOIN reviews r ON h.course_id = r.course_id JOIN materials m ON m.id = h.material_id  JOIN chapters cp ON m.chapter_id = cp.id JOIN courses c ON r.course_id = c.id JOIN categoris ct ON c.categori_id = ct.id;
        """)

        records = cursor.fetchall()
        columns = ["user_id", "course_id", "rating"]
        cf = pd.DataFrame(records, columns=columns)
        
        # Pastikan kolom rating adalah numerik
        cf['rating'] = pd.to_numeric(cf['rating'], errors='coerce')
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()

    return cf

# Fungsi untuk memuat data Content-Based Filtering (CBF)
def load_data_CBF():
    try:
        conn = mysql.connector.connect(user='root', host='localhost', database='db_course')
        cursor = conn.cursor()
        cursor.execute("""
        SELECT DISTINCT  r.user_id AS user_id,  r.course_id AS course_id,  c.course_title AS course_title,  c.about AS description,  r.rating AS rating,  ct.type AS category_type
                       FROM histories h JOIN reviews r ON h.course_id = r.course_id JOIN materials m ON h.material_id = m.id JOIN chapters cp ON m.chapter_id = cp.id JOIN courses c ON r.course_id = c.id JOIN categoris ct ON c.categori_id = ct.id;
        """)

        records = cursor.fetchall()
        columns = [
            "user_id", "course_id", "course_title", "description", "rating",  "category_type"
        ]
        cbf = pd.DataFrame(records, columns=columns)
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()

    return cbf

# Muat data CF dan CBF
cf = load_data_CF()
cbf = load_data_CBF()

# Gabungkan nilai dari semua kolom teks menjadi satu string
def combine_text(row):
    return ' '.join(row.values.astype(str))

# Buat kolom baru 'all_text' yang berisi gabungan nilai dari semua kolom teks
cbf['all_text'] = cbf.apply(combine_text, axis=1)

# Proses pembuatan model TF-IDF untuk CBF
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix_cbf = tfidf_vectorizer.fit_transform(cbf['all_text'])

# Hitung kemiripan kosinus antar dokumen untuk CBF
cosine_sim_cbf = cosine_similarity(tfidf_matrix_cbf, tfidf_matrix_cbf)

# Fungsi untuk merekomendasikan kursus
def recommend(user_id, cosine_sim_cbf=cosine_sim_cbf, top_n=5):
    # Filter kursus yang belum dilihat oleh user
    user_courses = cf[cf['user_id'] == user_id]['course_id']
    unseen_courses = cbf[~cbf['course_id'].isin(user_courses)]
    
    # Jika tidak ada kursus yang belum dilihat oleh user, kembalikan dataframe kosong
    if unseen_courses.empty:
        return pd.DataFrame(columns=['course_title', 'About_Course', 'category_type'])
    
    # Prediksi rating untuk kursus yang belum dilihat oleh user (CF)
    user_ratings = cf[cf['user_id'] == user_id].groupby('course_id')['rating'].mean()
    
    # Jika tidak ada rating yang tersedia, set semua peringkat menjadi 0
    if user_ratings.empty:
        user_ratings = pd.Series(0, index=pd.RangeIndex(1, len(cosine_sim_cbf) + 1))
    else:
        user_ratings = user_ratings.reindex(pd.RangeIndex(1, len(cosine_sim_cbf) + 1), fill_value=0)
    
    # Gabungkan hasil CF dan CBF
    course_scores_cf = cosine_sim_cbf.dot(user_ratings)
    
    # Ambil N kursus teratas yang belum dilihat oleh user
    recommended_courses_indices = np.argsort(course_scores_cf)[::-1][:min(top_n, len(course_scores_cf))]
    recommended_courses = cbf.loc[recommended_courses_indices, ['course_id','course_title','category_type']]
    
    # Hapus duplikasi berdasarkan course_id
    recommended_courses = recommended_courses.drop_duplicates(subset=['course_id'])
    return recommended_courses.to_dict(orient='records')

# Aplikasi Streamlit
st.title('Course Recommendation System')

user_id = st.text_input('Enter User ID:')
if user_id:
    recommendations = recommend(int(user_id))
    if recommendations:
        st.write('Recommended Courses:')
        for course in recommendations:
            st.write(f"Course Title: {course['course_title']}, Category: {course['category_type']}")
    else:
        st.write('No recommendations available.')

# Memulai ngrok
public_url = ngrok.connect(8501)
st.write(f'Public URL: {public_url}')

# Menjalankan aplikasi Streamlit
if __name__ == '__main__':
    import os
    os.system('streamlit run main.py')
