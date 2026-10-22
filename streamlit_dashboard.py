import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Nintendo Video Game Decision Analytics Dashboard",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-high {
        background-color: #d4edda;
        border-color: #28a745;
    }
    .prediction-medium {
        background-color: #fff3cd;
        border-color: #ffc107;
    }
    .prediction-low {
        background-color: #f8d7da;
        border-color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the data"""
    try:
        # Load the data
        df = pd.read_csv("deku_cleaned_final.csv")
        
        # Basic preprocessing
        df = df.dropna(subset=['msrp', 'release_year', 'is_multiplayer'])
        
        # Create derived features
        df['game_age'] = 2024 - df['release_year']
        df['price_category'] = pd.cut(df['msrp'], 
                                    bins=[0, 10, 25, 50, 100, float('inf')], 
                                    labels=['Budget', 'Low', 'Mid', 'High', 'Premium'])
        
        return df
    except FileNotFoundError:
        st.error("Data file 'deku_cleaned_final.csv' not found. Please ensure the file is in the same directory.")
        return None

@st.cache_resource
def train_model(df):
    """Train the Random Forest model for sequel prediction"""
    # Feature selection
    feature_cols = [
        'msrp', 'platform_count', 'min_players', 'max_players', 'is_multiplayer',
        'release_year', 'game_age', 'genre_count', 'mode_handheld', 'mode_docked', 'mode_tabletop'
    ]
    
    # Add genre columns if they exist
    genre_cols = [col for col in df.columns if col.startswith('genre_') and df[col].dtype != 'object']
    feature_cols.extend(genre_cols)
    
    # Prepare features and target
    X = df[feature_cols].fillna(0)
    y = df['is_sequel']
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X, y)
    
    return model, X.columns.tolist()

def create_filters(df):
    """Create sidebar filters"""
    st.sidebar.header("📊 Filters")
    
    # Genre filter
    if 'genre_primary_grouped' in df.columns:
        genres = ['All'] + sorted(df['genre_primary_grouped'].dropna().unique().tolist())
        selected_genre = st.sidebar.selectbox("Genre", genres)
    else:
        selected_genre = 'All'
    
    # Platform filter
    if 'platform_cleaned' in df.columns:
        platforms = ['All'] + sorted(df['platform_cleaned'].dropna().unique().tolist())
        selected_platform = st.sidebar.selectbox("Platform", platforms)
    else:
        selected_platform = 'All'
    
    # Release year range
    year_range = st.sidebar.slider(
        "Release Year Range",
        min_value=int(df['release_year'].min()),
        max_value=int(df['release_year'].max()),
        value=(int(df['release_year'].min()), int(df['release_year'].max()))
    )
    
    # Price range
    price_range = st.sidebar.slider(
        "Price Range ($)",
        min_value=float(df['msrp'].min()),
        max_value=float(df['msrp'].max()),
        value=(float(df['msrp'].min()), float(df['msrp'].max()))
    )
    
    # Multiplayer filter
    multiplayer_filter = st.sidebar.selectbox("Multiplayer", ['All', 'Yes', 'No'])
    
    return selected_genre, selected_platform, year_range, price_range, multiplayer_filter

def filter_data(df, genre, platform, year_range, price_range, multiplayer):
    """Apply filters to the dataset"""
    filtered_df = df.copy()
    
    if genre != 'All':
        filtered_df = filtered_df[filtered_df['genre_primary_grouped'] == genre]
    
    if platform != 'All':
        filtered_df = filtered_df[filtered_df['platform_cleaned'] == platform]
    
    filtered_df = filtered_df[
        (filtered_df['release_year'] >= year_range[0]) &
        (filtered_df['release_year'] <= year_range[1]) &
        (filtered_df['msrp'] >= price_range[0]) &
        (filtered_df['msrp'] <= price_range[1])
    ]
    
    if multiplayer != 'All':
        multiplayer_bool = 1 if multiplayer == 'Yes' else 0
        filtered_df = filtered_df[filtered_df['is_multiplayer'] == multiplayer_bool]
    
    return filtered_df

def sequel_prediction_section(df, model, feature_cols):
    """Sequel likelihood prediction section"""
    st.header("🎯 Sequel Likelihood Predictions")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Predict Sequel Likelihood for New Games")
        
        # Input form for new game prediction
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                msrp = st.number_input("MSRP ($)", min_value=0.0, max_value=200.0, value=29.99)
                platform_count = st.number_input("Platform Count", min_value=1, max_value=10, value=1)
                min_players = st.number_input("Min Players", min_value=1, max_value=100, value=1)
                max_players = st.number_input("Max Players", min_value=1, max_value=100, value=4)
                is_multiplayer = st.selectbox("Multiplayer", [0, 1])
                release_year = st.number_input("Release Year", min_value=1990, max_value=2024, value=2023)
            
            with col2:
                game_age = 2024 - release_year
                genre_count = st.number_input("Genre Count", min_value=1, max_value=10, value=1)
                mode_handheld = st.selectbox("Handheld Mode", [0, 1])
                mode_docked = st.selectbox("Docked Mode", [0, 1])
                mode_tabletop = st.selectbox("Tabletop Mode", [0, 1])
                
                # Add genre features (simplified)
                genre_action = st.selectbox("Action Genre", [0, 1])
                genre_adventure = st.selectbox("Adventure Genre", [0, 1])
                genre_rpg = st.selectbox("RPG Genre", [0, 1])
            
            submitted = st.form_submit_button("Predict Sequel Likelihood")
            
            if submitted:
                # Create feature vector
                features = [msrp, platform_count, min_players, max_players, is_multiplayer,
                          release_year, game_age, genre_count, mode_handheld, mode_docked, mode_tabletop]
                
                # Add genre features (pad with zeros if needed)
                genre_features = [genre_action, genre_adventure, genre_rpg]
                features.extend(genre_features)
                
                # Ensure we have enough features
                while len(features) < len(feature_cols):
                    features.append(0)
                
                features = features[:len(feature_cols)]
                
                # Make prediction
                prediction_proba = model.predict_proba([features])[0]
                sequel_probability = prediction_proba[1]
                
                with col2:
                    st.subheader("Prediction Result")
                    
                    # Determine prediction category
                    if sequel_probability >= 0.7:
                        prediction_class = "prediction-high"
                        prediction_text = "High Likelihood"
                        emoji = "🟢"
                    elif sequel_probability >= 0.4:
                        prediction_class = "prediction-medium"
                        prediction_text = "Medium Likelihood"
                        emoji = "🟡"
                    else:
                        prediction_class = "prediction-low"
                        prediction_text = "Low Likelihood"
                        emoji = "🔴"
                    
                    st.markdown(f"""
                    <div class="metric-card {prediction_class}">
                        <h3>{emoji} {prediction_text}</h3>
                        <h2>{sequel_probability:.1%}</h2>
                        <p>Probability of Sequel</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence explanation
                    if sequel_probability >= 0.7:
                        st.info("🎯 High confidence - Strong indicators suggest sequel potential")
                    elif sequel_probability >= 0.4:
                        st.warning("⚠️ Medium confidence - Mixed indicators, consider market research")
                    else:
                        st.error("❌ Low confidence - Limited sequel potential based on current features")

def pricing_optimization_section(df):
    """Pricing optimization suggestions section"""
    st.header("💰 Pricing Optimization Suggestions")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Price Analysis by Category")
        
        # Price distribution by genre
        if 'genre_primary_grouped' in df.columns:
            fig = px.box(df, x='genre_primary_grouped', y='msrp', 
                        title="Price Distribution by Genre",
                        labels={'genre_primary_grouped': 'Genre', 'msrp': 'MSRP ($)'})
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Price vs Game Age
        fig = px.scatter(df, x='game_age', y='msrp', color='is_multiplayer',
                        title="Price vs Game Age",
                        labels={'game_age': 'Game Age (Years)', 'msrp': 'MSRP ($)', 'is_multiplayer': 'Multiplayer'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Pricing Recommendations")
        
        # Calculate pricing insights
        avg_price = df['msrp'].mean()
        median_price = df['msrp'].median()
        price_std = df['msrp'].std()
        
        # Price optimization suggestions
        st.metric("Average Price", f"${avg_price:.2f}")
        st.metric("Median Price", f"${median_price:.2f}")
        st.metric("Price Standard Deviation", f"${price_std:.2f}")
        
        # Pricing recommendations based on data
        st.subheader("📊 Pricing Insights")
        
        # Genre-based pricing
        if 'genre_primary_grouped' in df.columns:
            genre_prices = df.groupby('genre_primary_grouped')['msrp'].agg(['mean', 'count']).sort_values('mean', ascending=False)
            
            st.write("**Top 5 Highest-Priced Genres:**")
            for genre, row in genre_prices.head().iterrows():
                st.write(f"• {genre}: ${row['mean']:.2f} (n={row['count']})")
        
        # Multiplayer pricing premium
        multiplayer_premium = df[df['is_multiplayer'] == 1]['msrp'].mean() - df[df['is_multiplayer'] == 0]['msrp'].mean()
        st.write(f"**Multiplayer Premium:** ${multiplayer_premium:.2f}")
        
        # Age-based pricing
        recent_games = df[df['game_age'] <= 2]['msrp'].mean()
        older_games = df[df['game_age'] > 5]['msrp'].mean()
        age_premium = recent_games - older_games
        st.write(f"**Recent vs Older Games Premium:** ${age_premium:.2f}")

def analytics_dashboard(df):
    """Main analytics dashboard"""
    st.header("📈 Analytics Dashboard")
    
    # Debug info
    if len(df) == 0:
        st.error("⚠️ No data available with current filters. Please adjust your filter settings.")
        st.write("**Debug Info:**")
        st.write(f"- DataFrame shape: {df.shape}")
        st.write(f"- DataFrame columns: {list(df.columns)}")
        if len(df) > 0:
            st.write(f"- MSRP range: {df['msrp'].min()} to {df['msrp'].max()}")
            st.write(f"- Release year range: {df['release_year'].min()} to {df['release_year'].max()}")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Games", f"{len(df):,}")
    
    with col2:
        sequel_count = df['is_sequel'].sum()
        sequel_pct = (sequel_count / len(df)) * 100
        st.metric("Sequels", f"{sequel_count:,} ({sequel_pct:.1f}%)")
    
    with col3:
        avg_price = df['msrp'].mean()
        st.metric("Average Price", f"${avg_price:.2f}")
    
    with col4:
        multiplayer_pct = (df['is_multiplayer'].sum() / len(df)) * 100
        st.metric("Multiplayer Games", f"{multiplayer_pct:.1f}%")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Price distribution
        if len(df) > 0 and df['msrp'].notna().sum() > 0:
            fig = px.histogram(df, x='msrp', nbins=30, title="Price Distribution",
                              labels={'msrp': 'MSRP ($)', 'count': 'Number of Games'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No price data available for visualization.")
        
        # Release year distribution
        if len(df) > 0 and df['release_year'].notna().sum() > 0:
            fig = px.histogram(df, x='release_year', nbins=20, title="Release Year Distribution",
                              labels={'release_year': 'Release Year', 'count': 'Number of Games'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No release year data available for visualization.")
    
    with col2:
        # Sequel vs Non-sequel price comparison
        if len(df) > 0 and df['msrp'].notna().sum() > 0 and df['is_sequel'].notna().sum() > 0:
            fig = px.box(df, x='is_sequel', y='msrp', title="Price Comparison: Sequels vs Non-Sequels",
                        labels={'is_sequel': 'Is Sequel', 'msrp': 'MSRP ($)'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for sequel price comparison.")
        
        # Multiplayer distribution
        multiplayer_counts = df['is_multiplayer'].value_counts()
        if len(multiplayer_counts) > 0:
            # Create dynamic labels based on actual data
            labels = ['Multiplayer' if idx == 1 else 'Single Player' for idx in multiplayer_counts.index]
            fig = px.pie(values=multiplayer_counts.values, names=labels,
                        title="Multiplayer vs Single Player Distribution")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for multiplayer distribution with current filters.")

def main():
    """Main application"""
    # Header
    st.markdown('<h1 class="main-header">🎮 Nintendo Video Game Decision Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Train model
    model, feature_cols = train_model(df)
    
    # Create filters
    selected_genre, selected_platform, year_range, price_range, multiplayer_filter = create_filters(df)
    
    # Apply filters
    filtered_df = filter_data(df, selected_genre, selected_platform, year_range, price_range, multiplayer_filter)
    
    # Show filtered data info
    st.sidebar.success(f"Showing {len(filtered_df):,} games out of {len(df):,} total")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📊 Analytics Dashboard", "🎯 Sequel Predictions", "💰 Pricing Optimization"])
    
    with tab1:
        analytics_dashboard(filtered_df)
    
    with tab2:
        sequel_prediction_section(filtered_df, model, feature_cols)
    
    with tab3:
        pricing_optimization_section(filtered_df)

if __name__ == "__main__":
    main() 