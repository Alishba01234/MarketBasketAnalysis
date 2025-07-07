import streamlit as st
import pandas as pd
from surprise import Dataset, Reader, SVD
import pickle

# --- Load Data ---
df = pd.read_csv("electronics_dataset1.csv")

# --- Prepare Training Data ---
reader = Reader(rating_scale=(1, 5))
train_data = df[df['split'] == 0][['user_id', 'item_id', 'rating']]
data = Dataset.load_from_df(train_data, reader)
trainset = data.build_full_trainset()

# --- Train SVD Model (or load if saved) ---
@st.cache_resource
def train_model():
    model = SVD()
    model.fit(trainset)
    return model

model = train_model()

# --- Streamlit UI ---
st.title("📊 Recommender System (Collaborative Filtering)")

# Select user from test split
test_users = df[df['split'] == 2]['user_id'].unique()
selected_user = st.selectbox("Select a User from Test Set", test_users)

# Generate Recommendations
def get_top_n_recommendations(model, user_id, df_all, trainset, n=5):
    all_items = df_all['item_id'].unique()
    rated_items = df_all[(df_all['user_id'] == user_id) & (df_all['split'] == 0)]['item_id'].values

    predictions = []
    for item in all_items:
        if item not in rated_items:
            try:
                pred = model.predict(user_id, item)
                predictions.append((item, pred.est))
            except:
                continue

    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:n]

if st.button("🎯 Recommend Top 5 Items"):
    top_5 = get_top_n_recommendations(model, selected_user, df, trainset, n=5)
    st.subheader(f"Top 5 Recommendations for User {selected_user}")
    rec_df = pd.DataFrame(top_5, columns=["Item ID", "Predicted Rating"])
    st.table(rec_df)