import streamlit as st
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load your dataset (replace with your actual data loading)
@st.cache_data
def load_data():
    # This should be replaced with your actual data loading code
    df = pd.read_csv('electronics_dataset1.csv')  # Update with your file path
    return df

df = load_data()

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Market Basket Analysis", "Recommendation System"])

# Home Page
if page == "Home":
    st.title("E-Commerce Analytics Dashboard")
    st.write("""
    Welcome to the E-Commerce Analytics Dashboard! This application provides:
    - **Market Basket Analysis**: Discover frequently co-purchased items
    - **Personalized Recommendations**: Get item suggestions for users
    """)
    
    st.subheader("Dataset Preview")
    st.write(df.head(8))
    # Basic stats
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Users", df['user_id'].nunique())
    col2.metric("Total Products", df['item_id'].nunique())
    col3.metric("Average Rating", f"{df['rating'].mean():.2f} ⭐")

# Market Basket Analysis page
elif page == "Market Basket Analysis":
    st.title("🛍️ Market Basket Analysis")
st.write("Discover products that are frequently purchased together")

with st.expander("⚙️ Analysis Settings"):
    col1, col2 = st.columns(2)
    min_support = col1.slider("Minimum Support", 0.001, 0.2, 0.03, 0.001, 
                            help="Minimum frequency of itemset occurrence")
    min_lift = col2.slider("Minimum Lift", 0.5, 5.0, 1.2, 0.1,
                         help="Strength of association between items")

if st.button("🔍 Run Analysis", type="primary"):
    with st.spinner("Analyzing purchase patterns..."):
        try:
            # Filter training data
            train_data = df[df['split'] == 0]
            
            # Filter to users with at least 2 purchases
            user_counts = train_data['user_id'].value_counts()
            valid_users = user_counts[user_counts >= 2].index
            filtered_data = train_data[train_data['user_id'].isin(valid_users)]
            
            if len(filtered_data) == 0:
                st.error("❌ Insufficient transaction data. Try lowering the minimum support.")
                st.stop()
            
            # Prepare transactions
            user_transactions = filtered_data.groupby('user_id')['item_id'].apply(list).tolist()
            
            # One-hot encoding
            te = TransactionEncoder()
            te_data = te.fit(user_transactions).transform(user_transactions)
            df_te = pd.DataFrame(te_data, columns=te.columns_)
            
            # Apriori algorithm (now looking for triplets with max_len=3)
            frequent_itemsets = apriori(df_te, min_support=min_support, use_colnames=True, max_len=3)
            
            if len(frequent_itemsets) == 0:
                st.warning(f"⚠️ No itemsets found with support ≥ {min_support}. Try lowering the threshold.")
                st.stop()
            
            # Generate rules
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_lift)
            
            if len(rules) == 0:
                st.warning(f"⚠️ No rules found with lift ≥ {min_lift}. Try reducing the lift threshold.")
                st.stop()
            
            rules_sorted = rules.sort_values(by="lift", ascending=False)
            
            # Display results
            st.success(f"✅ Found {len(rules)} association rules")
            
            # Helper function to get product details
            def get_product_info(item_set):
                items = []
                for item_id in item_set:
                    item = df[df['item_id'] == item_id].iloc[0]
                    items.append(f"{item['brand']} {item['category']} (ID: {item_id})")
                return " + ".join(items)
            
            # TABBED DISPLAY FOR PAIRS AND TRIPLETS
            tab1, tab2 = st.tabs(["Top Product Pairs", "Top Product Triplets"])
            
            with tab1:
                st.subheader("Top Product Pairs")
                pairs = rules_sorted[
                    (rules_sorted['antecedents'].apply(len) == 1) & 
                    (rules_sorted['consequents'].apply(len) == 1)
                ]
                
                if len(pairs) > 0:
                    # Create enriched display
                    pairs_display = pairs[['antecedents', 'consequents', 'support', 'confidence', 'lift']].copy()
                    pairs_display['Rule'] = pairs_display.apply(
                        lambda x: f"{get_product_info(x['antecedents'])} → {get_product_info(x['consequents'])}", 
                        axis=1)
                    
                    # Show top 10 pairs
                    st.dataframe(
                        pairs_display[['Rule', 'support', 'confidence', 'lift']]
                        .rename(columns={
                            'support': 'Support', 
                            'confidence': 'Confidence',
                            'lift': 'Lift'
                        })
                        .head(10),
                        column_config={
                            "Rule": "Product Association",
                            "Support": st.column_config.NumberColumn(format="%.3f"),
                            "Confidence": st.column_config.NumberColumn(format="%.3f"),
                            "Lift": st.column_config.NumberColumn(format="%.2f")
                        }
                    )
                    
                    # Visualization for pairs
                    fig1, ax1 = plt.subplots(figsize=(10, 6))
                    top_pairs = pairs.head(5)
                    ax1.barh(
                        [f"{get_product_info(x['antecedents'])} → {get_product_info(x['consequents'])}" 
                         for _, x in top_pairs.iterrows()],
                        top_pairs['lift'],
                        color='skyblue'
                    )
                    ax1.set_xlabel('Lift Score')
                    ax1.set_title('Top Product Pairs by Lift')
                    st.pyplot(fig1)
                else:
                    st.info("ℹ️ No significant product pairs found")
            
            with tab2:
                st.subheader("Top Product Triplets")
                triplets = rules_sorted[
                    ((rules_sorted['antecedents'].apply(len) == 2) & 
                     (rules_sorted['consequents'].apply(len) == 1)) |
                    ((rules_sorted['antecedents'].apply(len) == 1) & 
                     (rules_sorted['consequents'].apply(len) == 2))
                ]
                
                if len(triplets) > 0:
                    # Create enriched display
                    triplets_display = triplets[['antecedents', 'consequents', 'support', 'confidence', 'lift']].copy()
                    triplets_display['Rule'] = triplets_display.apply(
                        lambda x: f"{get_product_info(x['antecedents'])} → {get_product_info(x['consequents'])}", 
                        axis=1)
                    
                    # Show top 10 triplets
                    st.dataframe(
                        triplets_display[['Rule', 'support', 'confidence', 'lift']]
                        .rename(columns={
                            'support': 'Support', 
                            'confidence': 'Confidence',
                            'lift': 'Lift'
                        })
                        .head(10),
                        column_config={
                            "Rule": "Product Association",
                            "Support": st.column_config.NumberColumn(format="%.3f"),
                            "Confidence": st.column_config.NumberColumn(format="%.3f"),
                            "Lift": st.column_config.NumberColumn(format="%.2f")
                        }
                    )
                    
                    # Visualization for triplets
                    fig2, ax2 = plt.subplots(figsize=(10, 6))
                    top_triplets = triplets.head(5)
                    ax2.barh(
                        [f"{get_product_info(x['antecedents'])} → {get_product_info(x['consequents'])}" 
                         for _, x in top_triplets.iterrows()],
                        top_triplets['lift'],
                        color='lightgreen'
                    )
                    ax2.set_xlabel('Lift Score')
                    ax2.set_title('Top Product Triplets by Lift')
                    st.pyplot(fig2)
                else:
                    st.info("ℹ️ No significant product triplets found")
            
            # Combined metrics visualization
            st.subheader("Association Rule Quality Metrics")
            fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Support vs Confidence
            ax3.scatter(rules['support'], rules['confidence'], alpha=0.5, color='blue')
            ax3.set_xlabel('Support')
            ax3.set_ylabel('Confidence')
            ax3.set_title('Support vs Confidence')
            
            # Support vs Lift
            ax4.scatter(rules['support'], rules['lift'], alpha=0.5, color='orange')
            ax4.set_xlabel('Support')
            ax4.set_ylabel('Lift')
            ax4.set_title('Support vs Lift')
            
            st.pyplot(fig3)
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Recommendation System Page
elif page == "Recommendation System":
    st.title("Personalized Recommendation System")
    
    # Train the model (cached for performance)
    @st.cache_resource
    def train_model():
        train_data = df[df['split'] == 0][['user_id', 'item_id', 'rating']]
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(train_data, reader)
        trainset = data.build_full_trainset()
        model = SVD()
        model.fit(trainset)
        return model, trainset
    
    model, trainset = train_model()
    
    # Get user input
    st.subheader("Get Recommendations")
    
    # Option 1: Select from test users
    test_users = df[df['split'] == 2]['user_id'].unique()
    selected_user = st.selectbox("Select a user ID", test_users)
    
    # Option 2: Manual user input
    manual_user = st.number_input("Or enter a user ID manually", 
                                min_value=min(df['user_id']), 
                                max_value=max(df['user_id']))
    
    # Use manual input if provided, otherwise use selected user
    user_id = manual_user if manual_user else selected_user
    
    n_recommendations = st.slider("Number of recommendations", 1, 10, 5)
    
    if st.button("Generate Recommendations"):
        # Get recommendations
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
        
        recommendations = get_top_n_recommendations(model, user_id, df, trainset, n_recommendations)
        
        # Display recommendations
        st.subheader(f"Top {n_recommendations} Recommendations for User {user_id}")
        
        for i, (item, score) in enumerate(recommendations, 1):
            item_data = df[df['item_id'] == item].iloc[0]
            st.write(f"""
            **#{i}**  
            **Item ID**: {item}  
            **Category**: {item_data['category']}  
            **Brand**: {item_data['brand']}  
            **Predicted Rating**: {score:.2f}  
            """)
            st.write("---")
