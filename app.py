import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from proses import upload_dataset, preprocess, tfidf, load_model, borderline_smote
from klasifikasi import klasifikasi_svm, metrik_klasifikasi, prediksi_svm

st.write("""
# KLASIFIKASI IKLAN LOWONGAN KERJA
""")

# Tombol untuk mengunggah dataset di sidebar
def browsefiles_clicked():
    file = st.sidebar.file_uploader("Unggah Dataset (CSV)", type=["csv"])
    return upload_dataset(file)

df, file_name = browsefiles_clicked()

# Streamlit Sidebar
st.sidebar.header("Pengaturan")
c = st.sidebar.selectbox("Nilai C", [0.1, 1.0])
algoritma = st.sidebar.selectbox("Pilih Algoritma", ["SVM", "SVM + Borderline-SMOTE"])

if(df is not None):
    if file_name is not None and "test" in file_name:
        df.drop(['salary_range','job_id'], axis = 1, inplace = True)
        df.fillna(" ",inplace = True)
        df['text'] = df['title'] +  ' ' + df['department'] + ' ' + df['company_profile'] + ' ' + df['description'] + ' ' + df['requirements'] + ' ' + df['benefits'] + ' ' + df['employment_type'] + ' ' + df['required_education'] + ' ' + df['industry'] + ' ' + df['function'] 
        df.drop(['title','department','company_profile','description','requirements','benefits','employment_type','required_experience','required_education','industry','function'], axis = 1, inplace = True)

        loaded_model = load_model(algoritma, c)

        df['text'] = df['text'].apply(preprocess)

        df_tfidf_test = tfidf(df, joblib.load('model/vocabulary.joblib'))

        y = df_tfidf_test['frauds']
        X = df_tfidf_test.drop(['frauds'], axis=1)

        predicted = prediksi_svm(loaded_model, X)
        cm, precision, recall, f1 = metrik_klasifikasi(y, predicted)

        st.text("Confusion Matrix:")
        st.write(cm)
            
        st.text("SVM Metrics:")
        st.text(
            f"SVM Precision: {precision}"
        )
        st.text(
            f"SVM Recall: {recall}"
        )
        st.text(
            f"SVM F1 Score: {f1}"
        )

        result_df = pd.DataFrame({
            'telecommuting' : df['telecommuting'],
            'has_company_logo' : df['has_company_logo'],
            'has_questions' : df['has_questions'],
            'text' : df['text'],
            'Label Asli': df['frauds'],
            'Prediksi Model': predicted 
        })
        result_df['Label Asli'] = result_df['Label Asli'].map({0: 'asli', 1: 'palsu'})
        result_df['Prediksi Model'] = result_df['Prediksi Model'].map({0: 'asli', 1: 'palsu'})
     
        result_df['Prediksi Benar'] = (result_df['Label Asli'] == result_df['Prediksi Model'])
        st.write(result_df)

    else:
        st.write(df.head())
        df.drop(['salary_range','job_id'], axis = 1, inplace = True)
        df.fillna(" ",inplace = True)
        df['text'] = df['title'] +  ' ' + df['department'] + ' ' + df['company_profile'] + ' ' + df['description'] + ' ' + df['requirements'] + ' ' + df['benefits'] + ' ' + df['employment_type'] + ' ' + df['required_education'] + ' ' + df['industry'] + ' ' + df['function'] 
        df.drop(['title','department','company_profile','description','requirements','benefits','employment_type','required_experience','required_education','industry','function'], axis = 1, inplace = True)

        df['text'] = df['text'].apply(preprocess)

        df_tfidf = tfidf(df)

        target = df_tfidf['frauds']
        features = df_tfidf.drop(['frauds'], axis=1)

        X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.3, random_state=42)
        
        df_val = pd.read_csv('data/val.csv')

        if algoritma == "SVM":
            st.subheader("SVM")
            
            model = klasifikasi_svm(c, X_train, y_train)
            predicted = prediksi_svm(model, X_val)
            cm, precision, recall, f1 = metrik_klasifikasi(y_val, predicted)

            st.text("Confusion Matrix:")
            st.write(cm)

            st.text("SVM Metrics:")
            st.text(
                f"SVM Precision: {precision}"
            )
            st.text(
                f"SVM Recall: {recall}"
            )
            st.text(
                f"SVM F1 Score: {f1}"
            )

            result_df = pd.DataFrame({
                'telecommuting' : df_val['telecommuting'],
                'has_company_logo' : df_val['has_company_logo'],
                'has_questions' : df_val['has_questions'],
                'text' : df_val['text'],
                'Label Asli': df_val['frauds'],
                'Prediksi Model': predicted 
            })
            result_df['Label Asli'] = result_df['Label Asli'].map({0: 'asli', 1: 'palsu'})
            result_df['Prediksi Model'] = result_df['Prediksi Model'].map({0: 'asli', 1: 'palsu'})
        
            result_df['Prediksi Benar'] = (result_df['Label Asli'] == result_df['Prediksi Model'])
            st.write(result_df)

        # SVM Dengan Borderline-SMOTE
        elif algoritma == "SVM + Borderline-SMOTE":
            st.subheader("SVM + Borderline-SMOTE")
            
            X_train_res, y_train_res = borderline_smote(X_train, y_train)
            model = klasifikasi_svm(c, X_train_res, y_train_res)
            predicted = prediksi_svm(model, X_val)
            cm, precision, recall, f1 = metrik_klasifikasi(y_val, predicted)

            st.text("Confusion Matrix:")
            st.write(cm)

            st.text("SVM Metrics:")
            st.text(
                f"SVM Precision: {precision}"
            )
            st.text(
                f"SVM Recall: {recall}"
            )
            st.text(
                f"SVM F1 Score: {f1}"
            )

            result_df = pd.DataFrame({
                'telecommuting' : df_val['telecommuting'],
                'has_company_logo' : df_val['has_company_logo'],
                'has_questions' : df_val['has_questions'],
                'text' : df_val['text'],
                'Label Asli': df_val['frauds'],
                'Prediksi Model': predicted 
            })
            result_df['Label Asli'] = result_df['Label Asli'].map({0: 'asli', 1: 'palsu'})
            result_df['Prediksi Model'] = result_df['Prediksi Model'].map({0: 'asli', 1: 'palsu'})
        
            result_df['Prediksi Benar'] = (result_df['Label Asli'] == result_df['Prediksi Model'])
            st.write(result_df)
