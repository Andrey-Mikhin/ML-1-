import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Car Price Forecast",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded")

st.title("Car Price Forecast")

@st.cache_resource
def load_all_models():
    with open('best_ridge_model.pickle', 'rb') as f:
        model_data = pickle.load(f)
        model = model_data['model']

    with open('model_metrics.pickle', 'rb') as f:
        metrics = pickle.load(f)

    with open('data_info.pickle', 'rb') as f:
        data_info = pickle.load(f)

    return model, metrics, data_info, model_data

model, metrics, data_info, model_data = load_all_models()

if 'train_df' not in st.session_state:
    st.session_state.train_df = None
if 'test_df' not in st.session_state:
    st.session_state.test_df = None

tab1, tab2, tab3 = st.tabs(["Data", "Forecast", "Metrics"])

with tab1:
    train_file = st.file_uploader("train csv", type=['csv'])
    if train_file:
        st.session_state.train_df = pd.read_csv(train_file)
        st.write(f"Train: {st.session_state.train_df.shape}")
        st.dataframe(st.session_state.train_df.head())

    test_file = st.file_uploader("test csv", type=['csv'])
    if test_file:
        st.session_state.test_df = pd.read_csv(test_file)
        st.write(f"Test: {st.session_state.test_df.shape}")
        st.dataframe(st.session_state.test_df.head())

with tab2:
    if st.session_state.test_df is not None:
        predictions = model.predict(st.session_state.test_df)

        results = pd.DataFrame({'Predicted Price': predictions})
        st.dataframe(results)

        fig, ax = plt.subplots()
        ax.hist(predictions, bins=20)
        st.pyplot(fig)
    else:
        st.write("Upload test data")


with tab3:
    st.write("**Pipeline:** StandardScaler + Ridge")
    
    if hasattr(model, 'named_steps') and 'ridge' in model.named_steps:
        ridge_model = model.named_steps['ridge']
        if hasattr(ridge_model, 'coef_'):
            st.write(f"**Coefficients:** {len(ridge_model.coef_)}")
            st.write(f"**Mean coefficient:** {ridge_model.coef_.mean():.4f}")

    st.write("**Metrics:**")
    st.write(f"R² train: {metrics.get('r2_train', 0):.4f}")
    st.write(f"R² test: {metrics.get('r2_test', 0):.4f}")
