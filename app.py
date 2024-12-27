import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay


# Custom CSS untuk desain 
st.markdown("""
     <style>
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #2D55B4; /* Darker background for better contrast */
            color: white;
        }
        [data-testid="stSidebar"] .css-1v3fvcr {
            font-size: 18px;
            font-weight: bold;
            color: white;
        }
        [data-testid="stSidebar"] a {
            text-decoration: none;
            color: #63B3ED; /* Light blue links */
        }
        [data-testid="stSidebar"] a:hover {
            color: #3182CE; /* Darker blue on hover */
        }

        /* Main page styling */
        .block-container {
            padding: 2rem;
            background-color: #f2f4f7; /* Light gray background */
        }

        /* Headings and titles */
        h1, h2, h3 {
            color: #808080; /* Gray color for headings */
        }

        /* Text */
        .stMarkdown p, .stWrite {
            color: #808080; /* Gray color for paragraph text */
        }

        /* Labels and widget text */
        .stMultiSelect label, .stSlider label, .stSelectbox label, .stTextInput label, .stFileUploader label {
            color: #808080; /* Gray color for widget labels */
        }

        /* Dataframe styling */
        .css-1v2lvtn {
            font-size: 16px;
            color: #808080; /* Gray color for dataframe text */
        }

        /* Warning messages styling */
        .stWarning {
            color: #808080; /* Gray color for warning messages */
        }

        /* Button styling */
        button[data-baseweb="button"] {
            background-color: #3182CE; /* Blue buttons */
            color: white;
            font-weight: bold;
        }
        button[data-baseweb="button"]:hover {
            background-color: #2B6CB0; /* Darker blue on hover */
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar Menu Navigation
menu = st.sidebar.radio("Pilih Halaman", ["Home", "K-Means", "Naive Bayes"])

# Halaman Home
if menu == "Home":
    st.title("Selamat Datang di Dashboard Analisis")
    st.subheader("Pilih salah satu algoritma di sidebar untuk memulai.")
    st.write("""  
    Dashboard ini menyediakan analisis clustering menggunakan **K-Means** dan klasifikasi menggunakan **Naive Bayes**. 
    Anda dapat mengunggah dataset untuk memulai analisis.
    """)

# Halaman K-Means
elif menu == "K-Means":
    st.title("K-Means Clustering")
    st.markdown('<p class="stMarkdown">Halaman ini digunakan untuk analisis clustering dengan algoritma K-Means.</p>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload dataset (CSV format)", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("Dataset yang diunggah:")
        st.dataframe(data)

        features = st.multiselect("Pilih fitur untuk clustering:", data.columns)
        if features:
            X = data[features]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
       
            wcss = []
            for i in range(1, 11):
                kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
                kmeans.fit(X_scaled)
                wcss.append(kmeans.inertia_)

            plt.figure(figsize=(8, 5))
            plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
            plt.title('Elbow Method for Optimal Clusters', fontsize=14)
            plt.xlabel('Number of Clusters')
            plt.ylabel('WCSS')
            st.pyplot(plt)

            optimal_clusters = st.slider("Pilih jumlah cluster:", 2, 10, 3)
            kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42)
            clusters = kmeans.fit_predict(X_scaled)
            data['Cluster'] = clusters

            st.write("Data dengan Cluster:")
            st.dataframe(data)

      
            sil_score = silhouette_score(X_scaled, clusters)
            st.write(f"**Silhouette Score**: {sil_score:.2f}")

        
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            plt.figure(figsize=(8, 6))
            plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.7)
            plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=200, marker='X', label='Centroids')
            plt.title('Clustering with K-Means (PCA Visualization)', fontsize=14)
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.legend()
            st.pyplot(plt)
        else:
            st.warning("Pilih fitur untuk memulai analisis clustering.")  
    else:
        st.write("Silakan unggah dataset untuk memulai analisis K-Means.")

# Halaman Naive Bayes
elif menu == "Naive Bayes":
    st.title("Naive Bayes Classification")
    st.markdown('<p class="stMarkdown">Halaman ini digunakan untuk analisis klasifikasi dengan algoritma Naive Bayes.</p>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload dataset (CSV format)", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("Dataset yang diunggah:")
        st.dataframe(data)

        
        if 'Hasil NS1' in data.columns:
            data['Uji NS1'] = data['Uji NS1'].map({"Positif": 1, "Negatif": 0})
            data['Hasil NS1'] = data['Hasil NS1'].map({"Positif": 1, "Negatif": 0})
            data['Gender_Laki-laki'] = data['Gender_Laki-laki'].astype(int)
            data['Gender_Perempuan'] = data['Gender_Perempuan'].astype(int)

            features = ['Usia', 'Durasi Gejalah', 'Trombosit', 'Hematokrit', 
                        'Trombosit Normalized', 'Hematokrit Normalized', 'Demam', 
                        'Nyeri Sendi', 'Mual', 'Ruam', 'Sakit Kepala', 'Nyeri Otot', 
                        'Muntah', 'Perdarahan', 'Trombosit_Normalized', 'Hematokrit_Normalized']
            X = data[features]
            y = data['Hasil NS1']  

            
            imputer = SimpleImputer(strategy="mean")
            X_imputed = imputer.fit_transform(X)
            y = y.fillna(y.mode()[0]) 

            
            X_imputed_df = pd.DataFrame(X_imputed, columns=features)

            st.write(f"Missing values in X: {X_imputed_df.isnull().sum().sum()}")
            st.write(f"Missing values in y: {y.isnull().sum()}")

           
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_imputed_df)

          
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

         
            nb_model = GaussianNB()
            nb_model.fit(X_train, y_train)
            y_pred = nb_model.predict(X_test)

        
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)

        
            st.write(f'**Accuracy**: {accuracy:.4f}')
            st.write(f'**Precision**: {precision:.4f}')
            st.write(f'**Recall**: {recall:.4f}')
            st.write(f'**F1-Score**: {f1:.4f}')

          
            disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Negatif', 'Positif'])
            disp.plot(cmap='Blues')
            st.pyplot(plt)
        else:
            st.warning("Dataset harus memiliki kolom `Hasil NS1` sebagai target klasifikasi!")
    else:
        st.write("Silakan unggah dataset untuk memulai analisis Naive Bayes.")