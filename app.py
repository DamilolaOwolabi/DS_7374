import streamlit as st
import numpy as np
import csv
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Nintendo Video Game Analytics Dashboard",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS and JavaScript for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #e60012;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #e60012;
        margin: 0.5rem 0;
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
    /* Make text more visible */
    .stMarkdown, .stText, .stMetric, .stDataFrame {
        color: #000000 !important;
    }
    /* Style links */
    a {
        color: #e60012 !important;
        text-decoration: underline;
        font-weight: bold;
    }
    a:hover {
        color: #cc0010 !important;
        text-decoration: underline;
        background-color: #fff3cd;
        padding: 2px 4px;
        border-radius: 3px;
    }
    /* Make sidebar text visible */
    .css-1d391kg {
        color: #000000 !important;
    }
    /* Make plot titles and labels visible */
    .js-plotly-plot .plotly .main-svg {
        color: #000000 !important;
    }
</style>

<script>
// Function to style tab elements
function styleTabs() {
    // Wait for tabs to be rendered
    setTimeout(() => {
        // Find all tab elements
        const tabElements = document.querySelectorAll('[data-baseweb="tab"]');
        
        tabElements.forEach(tab => {
            // Style the tab container
            tab.style.backgroundColor = '#f0f2f6';
            tab.style.borderRadius = '4px 4px 0px 0px';
            tab.style.padding = '10px 16px';
            tab.style.color = '#000000';
            tab.style.fontWeight = 'bold';
            
            // Find text elements within the tab
            const textElements = tab.querySelectorAll('text, tspan, span, div');
            textElements.forEach(text => {
                text.style.fill = '#000000';
                text.style.color = '#000000';
                text.style.fontWeight = 'bold';
            });
        });
        
        // Style selected tab
        const selectedTab = document.querySelector('[aria-selected="true"]');
        if (selectedTab) {
            selectedTab.style.backgroundColor = '#e60012';
            selectedTab.style.color = 'white';
            selectedTab.style.fontWeight = 'bold';
            
            const selectedTextElements = selectedTab.querySelectorAll('text, tspan, span, div');
            selectedTextElements.forEach(text => {
                text.style.fill = 'white';
                text.style.color = 'white';
                text.style.fontWeight = 'bold';
            });
        }
    }, 1000);
}

// Run styling when page loads
document.addEventListener('DOMContentLoaded', styleTabs);

// Run styling periodically to catch dynamically loaded content
setInterval(styleTabs, 2000);
</script>
""", unsafe_allow_html=True)

class DataFrame:
    """Simple DataFrame-like class to replace pandas"""
    def __init__(self, data):
        self.data = data
        self.columns = list(data.keys()) if data else []
        self._length = len(next(iter(data.values()))) if data else 0
    
    def __len__(self):
        return self._length
    
    def __getitem__(self, key):
        if isinstance(key, str):
            return self.data[key]
        elif isinstance(key, list):
            return DataFrame({col: self.data[col] for col in key})
        else:
            # Handle boolean indexing
            mask = key
            filtered_data = {}
            for col in self.columns:
                filtered_data[col] = [self.data[col][i] for i in range(len(self)) if mask[i]]
            return DataFrame(filtered_data)
    
    def copy(self):
        return DataFrame({col: self.data[col].copy() for col in self.columns})
    
    def groupby(self, column):
        groups = {}
        for i in range(len(self)):
            key = self.data[column][i]
            if key not in groups:
                groups[key] = []
            groups[key].append(i)
        return groups
    
    def loc(self, mask):
        return self[mask]
    
    def fillna(self, value):
        new_data = {}
        for col in self.columns:
            new_data[col] = [value if x is None else x for x in self.data[col]]
        return DataFrame(new_data)
    
    def sum(self):
        return sum(self.data[self.columns[0]])
    
    def mean(self):
        return sum(self.data[self.columns[0]]) / len(self)
    
    def std(self):
        values = self.data[self.columns[0]]
        mean_val = sum(values) / len(values)
        variance = sum((x - mean_val) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def min(self):
        return min(self.data[self.columns[0]])
    
    def max(self):
        return max(self.data[self.columns[0]])
    
    def value_counts(self):
        counts = {}
        for value in self.data[self.columns[0]]:
            counts[value] = counts.get(value, 0) + 1
        return counts
    
    def unique(self):
        return list(set(self.data[self.columns[0]]))
    
    def tolist(self):
        return self.data[self.columns[0]]

def cut(series, bins, labels):
    """Simple implementation of pandas cut function"""
    result = []
    for value in series:
        for i, (bin_min, bin_max) in enumerate(zip(bins[:-1], bins[1:])):
            if bin_min <= value <= bin_max:
                result.append(labels[i])
                break
        else:
            result.append(labels[-1])
    return result

DATA_FILE = "deku_cleaned_final.csv"

# Columns the dashboard and model use, with how to read each one from the CSV
NUMERIC_COLS = [
    'msrp', 'release_year', 'game_age', 'platform_count', 'min_players', 'max_players',
    'genre_count', 'is_multiplayer', 'is_sequel', 'mode_handheld', 'mode_docked', 'mode_tabletop'
]
TEXT_COLS = ['title', 'genre_primary_grouped', 'platform_cleaned', 'publisher_grouped', 'languages']
INT_COLS = {'release_year', 'platform_count', 'min_players', 'max_players', 'genre_count',
            'is_multiplayer', 'is_sequel', 'mode_handheld', 'mode_docked', 'mode_tabletop'}


def _to_number(value, as_int):
    """Convert a CSV cell ('2022.0', 'True', '') to a number."""
    value = (value or '').strip()
    if value in ('True', 'true'):
        return 1
    if value in ('False', 'false', ''):
        return 0
    number = float(value)
    return int(number) if as_int else number


@st.cache_data
def load_data():
    """Load the real, cleaned DekuDeals dataset (22,123 Nintendo eShop games)."""
    data = {col: [] for col in NUMERIC_COLS + TEXT_COLS}
    with open(DATA_FILE, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        # The file repeats a few column names; use the first occurrence of each
        index = {}
        for i, name in enumerate(header):
            index.setdefault(name, i)
        for row in reader:
            for col in NUMERIC_COLS:
                data[col].append(_to_number(row[index[col]], col in INT_COLS))
            for col in TEXT_COLS:
                data[col].append(row[index[col]].strip())

    data['game_name'] = data['title']
    # Release dates can be a few days ahead of the scrape date, so never show a negative age
    data['game_age'] = [max(0.0, age) for age in data['game_age']]
    data['price_category'] = cut(
        data['msrp'], bins=[0, 20, 35, 50, 1000], labels=['Budget', 'Standard', 'Premium', 'Deluxe']
    )
    return DataFrame(data)


@st.cache_resource
def train_model(_df):
    """Train the Random Forest model for sequel prediction"""
    # Feature selection
    feature_cols = [
        'msrp', 'platform_count', 'min_players', 'max_players', 'is_multiplayer',
        'release_year', 'game_age', 'genre_count', 'mode_handheld', 'mode_docked', 'mode_tabletop'
    ]
    
    # Prepare features and target
    X = []
    for i in range(len(_df)):
        row = []
        for col in feature_cols:
            value = _df.data[col][i]
            row.append(0 if value is None else value)
        X.append(row)
    
    y = _df.data['is_sequel']
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X, y)
    
    return model, feature_cols

def create_filters(df):
    """Create sidebar filters"""
    st.sidebar.header("📊 Filters")
    
    # Genre filter
    genres = ['All'] + sorted(list(set(df.data['genre_primary_grouped'])))
    selected_genre = st.sidebar.selectbox("Genre", genres)
    
    # Platform filter
    platforms = ['All'] + sorted(list(set(df.data['platform_cleaned'])))
    selected_platform = st.sidebar.selectbox("Platform", platforms)
    
    # Release year range
    year_range = st.sidebar.slider(
        "Release Year Range",
        min_value=int(min(df.data['release_year'])),
        max_value=int(max(df.data['release_year'])),
        value=(int(min(df.data['release_year'])), int(max(df.data['release_year'])))
    )
    
    # Price range
    price_range = st.sidebar.slider(
        "Price Range ($)",
        min_value=float(min(df.data['msrp'])),
        max_value=float(max(df.data['msrp'])),
        value=(float(min(df.data['msrp'])), float(max(df.data['msrp'])))
    )
    
    # Multiplayer filter
    multiplayer_filter = st.sidebar.selectbox("Multiplayer", ['All', 'Yes', 'No'])
    
    return selected_genre, selected_platform, year_range, price_range, multiplayer_filter

def filter_data(df, genre, platform, year_range, price_range, multiplayer):
    """Apply filters to the dataset"""
    filtered_indices = []
    
    for i in range(len(df)):
        include = True
        
        if genre != 'All' and df.data['genre_primary_grouped'][i] != genre:
            include = False
        
        if platform != 'All' and df.data['platform_cleaned'][i] != platform:
            include = False
        
        if not (year_range[0] <= df.data['release_year'][i] <= year_range[1]):
            include = False
        
        if not (price_range[0] <= df.data['msrp'][i] <= price_range[1]):
            include = False
        
        if multiplayer != 'All':
            multiplayer_bool = 1 if multiplayer == 'Yes' else 0
            if df.data['is_multiplayer'][i] != multiplayer_bool:
                include = False
        
        if include:
            filtered_indices.append(i)
    
    filtered_data = {}
    for col in df.columns:
        filtered_data[col] = [df.data[col][i] for i in filtered_indices]
    
    # Ensure derived columns are also included
    if 'game_age' in df.data:
        filtered_data['game_age'] = [df.data['game_age'][i] for i in filtered_indices]
    if 'price_category' in df.data:
        filtered_data['price_category'] = [df.data['price_category'][i] for i in filtered_indices]
    
    return DataFrame(filtered_data)

def safe_plotly_chart(fig, use_container_width=True):
    """Safely render plotly chart with error handling"""
    try:
        # Ensure the figure has proper configuration
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black'),
            title_font=dict(color='black', size=16),
            xaxis=dict(
                title_font=dict(color='black'),
                tickfont=dict(color='black'),
                gridcolor='lightgray'
            ),
            yaxis=dict(
                title_font=dict(color='black'),
                tickfont=dict(color='black'),
                gridcolor='lightgray'
            )
        )
        
        # Add explicit configuration for better rendering
        fig.update_layout(
            showlegend=True,
            legend=dict(
                font=dict(color='black', size=12),
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='black',
                borderwidth=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=use_container_width, config={
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d']
        })
    except Exception as e:
        st.error(f"Error rendering chart: {str(e)}")
        st.write("Chart data:", fig.data)

def analytics_dashboard(df):
    """Main analytics dashboard"""
    st.header("📈 Analytics Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Games", f"{len(df):,}")
    
    with col2:
        sequel_count = sum(df.data['is_sequel'])
        sequel_pct = (sequel_count / len(df)) * 100
        st.metric("Sequels", f"{sequel_count:,} ({sequel_pct:.1f}%)")
    
    with col3:
        avg_price = sum(df.data['msrp']) / len(df)
        st.metric("Average Price", f"${avg_price:.2f}")
    
    with col4:
        multiplayer_pct = (sum(df.data['is_multiplayer']) / len(df)) * 100
        st.metric("Multiplayer Games", f"{multiplayer_pct:.1f}%")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Price distribution
        try:
            # Create histogram with explicit data
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df.data['msrp'],
                nbinsx=30,
                marker_color='#e60012',
                marker_line_color='white',
                marker_line_width=1.5
            ))
            fig.update_layout(
                title="Price Distribution",
                xaxis_title="MSRP ($)",
                yaxis_title="Number of Games",
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating price distribution chart: {e}")
        
        # Release year distribution
        try:
            # Create histogram with explicit data
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df.data['release_year'],
                nbinsx=10,
                marker_color='#e60012',
                marker_line_color='white',
                marker_line_width=1.5
            ))
            fig.update_layout(
                title="Release Year Distribution",
                xaxis_title="Release Year",
                yaxis_title="Number of Games",
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating release year chart: {e}")
    
    with col2:
        # Sequel vs Non-sequel price comparison
        try:
            # Prepare data for box plot
            sequel_prices = [df.data['msrp'][i] for i in range(len(df)) if df.data['is_sequel'][i] == 1]
            non_sequel_prices = [df.data['msrp'][i] for i in range(len(df)) if df.data['is_sequel'][i] == 0]
            
            fig = go.Figure()
            fig.add_trace(go.Box(y=non_sequel_prices, name='Non-Sequels', marker_color='#ff6b6b'))
            fig.add_trace(go.Box(y=sequel_prices, name='Sequels', marker_color='#4ecdc4'))
            
            fig.update_layout(
                title="Price Comparison: Sequels vs Non-Sequels",
                yaxis_title="MSRP ($)",
                xaxis_title="Is Sequel"
            )
            safe_plotly_chart(fig)
        except Exception as e:
            st.error(f"Error creating sequel comparison chart: {e}")
        
        # Multiplayer distribution
        try:
            multiplayer_counts = {}
            for is_multi in df.data['is_multiplayer']:
                multiplayer_counts[is_multi] = multiplayer_counts.get(is_multi, 0) + 1
            
            labels = ['Single Player' if k == 0 else 'Multiplayer' for k in multiplayer_counts.keys()]
            values = list(multiplayer_counts.values())
            
            fig = px.pie(
                values=values, 
                names=labels,
                title="Multiplayer vs Single Player Distribution",
                color_discrete_sequence=['#ff6b6b', '#4ecdc4']
            )
            safe_plotly_chart(fig)
        except Exception as e:
            st.error(f"Error creating multiplayer distribution chart: {e}")

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
                msrp = st.number_input("MSRP ($)", min_value=0.0, max_value=200.0, value=19.99)
                platform_count = st.number_input("Platform Count", min_value=1, max_value=7, value=2)
                min_players = st.number_input("Min Players", min_value=1, max_value=100, value=1)
                max_players = st.number_input("Max Players", min_value=1, max_value=100, value=4)
                is_multiplayer = st.selectbox("Multiplayer", [0, 1])
                release_year = st.number_input("Release Year", min_value=2000, max_value=2026, value=2025)
            
            with col2:
                game_age = max(0, 2025 - release_year)
                genre_count = st.number_input("Genre Count", min_value=1, max_value=10, value=1)
                mode_handheld = st.selectbox("Handheld Mode", [0, 1])
                mode_docked = st.selectbox("Docked Mode", [0, 1])
                mode_tabletop = st.selectbox("Tabletop Mode", [0, 1])
            
            submitted = st.form_submit_button("Predict Sequel Likelihood")
            
            if submitted:
                # Create feature vector
                features = [msrp, platform_count, min_players, max_players, is_multiplayer,
                          release_year, game_age, genre_count, mode_handheld, mode_docked, mode_tabletop]
                
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
        try:
            # Create box plot with explicit data
            fig = go.Figure()
            genres = list(set(df.data['genre_primary_grouped']))
            for genre in genres:
                genre_prices = [df.data['msrp'][i] for i in range(len(df)) if df.data['genre_primary_grouped'][i] == genre]
                fig.add_trace(go.Box(
                    y=genre_prices,
                    name=genre,
                    marker_color='#e60012'
                ))
            fig.update_layout(
                title="Price Distribution by Genre",
                xaxis_title="Genre",
                yaxis_title="MSRP ($)",
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating price distribution by genre chart: {e}")
        
        # Price vs Game Age
        try:
            # Create scatter plot with explicit data
            fig = go.Figure()
            
            # Separate data by multiplayer status
            multiplayer_indices = [i for i in range(len(df)) if df.data['is_multiplayer'][i] == 1]
            single_player_indices = [i for i in range(len(df)) if df.data['is_multiplayer'][i] == 0]
            
            # Add multiplayer games
            if multiplayer_indices:
                fig.add_trace(go.Scatter(
                    x=[df.data['game_age'][i] for i in multiplayer_indices],
                    y=[df.data['msrp'][i] for i in multiplayer_indices],
                    mode='markers',
                    name='Multiplayer',
                    marker=dict(color='#4ecdc4', size=8)
                ))
            
            # Add single player games
            if single_player_indices:
                fig.add_trace(go.Scatter(
                    x=[df.data['game_age'][i] for i in single_player_indices],
                    y=[df.data['msrp'][i] for i in single_player_indices],
                    mode='markers',
                    name='Single Player',
                    marker=dict(color='#ff6b6b', size=8)
                ))
            
            fig.update_layout(
                title="Price vs Game Age",
                xaxis_title="Game Age (Years)",
                yaxis_title="MSRP ($)",
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating price vs game age chart: {e}")
    
    with col2:
        st.subheader("Pricing Recommendations")
        
        # Calculate pricing insights
        avg_price = sum(df.data['msrp']) / len(df)
        median_price = np.median(df.data['msrp'])
        
        # Calculate standard deviation manually
        mean_price = sum(df.data['msrp']) / len(df)
        variance = sum((price - mean_price) ** 2 for price in df.data['msrp']) / len(df)
        price_std = variance ** 0.5
        
        # Price optimization suggestions
        st.metric("Average Price", f"${avg_price:.2f}")
        st.metric("Median Price", f"${median_price:.2f}")
        st.metric("Price Standard Deviation", f"${price_std:.2f}")
        
        # Pricing recommendations based on data
        st.subheader("📊 Pricing Insights")
        
        # Genre-based pricing
        genre_avg_prices = {}
        for i, genre in enumerate(df.data['genre_primary_grouped']):
            if genre not in genre_avg_prices:
                genre_avg_prices[genre] = {'total': 0, 'count': 0}
            genre_avg_prices[genre]['total'] += df.data['msrp'][i]
            genre_avg_prices[genre]['count'] += 1
        
        # Sort by average price
        sorted_genres = sorted(genre_avg_prices.items(), key=lambda x: x[1]['total']/x[1]['count'], reverse=True)
        
        st.write("**Top 5 Highest-Priced Genres:**")
        for genre, stats in sorted_genres[:5]:
            avg_price = stats['total'] / stats['count']
            st.write(f"• {genre}: ${avg_price:.2f} (n={stats['count']})")
        
        # Multiplayer pricing premium
        multiplayer_prices = [df.data['msrp'][i] for i in range(len(df)) if df.data['is_multiplayer'][i] == 1]
        single_player_prices = [df.data['msrp'][i] for i in range(len(df)) if df.data['is_multiplayer'][i] == 0]
        multiplayer_premium = (sum(multiplayer_prices) / len(multiplayer_prices)) - (sum(single_player_prices) / len(single_player_prices))
        st.write(f"**Multiplayer Premium:** ${multiplayer_premium:.2f}")
        
        # Age-based pricing
        recent_prices = [df.data['msrp'][i] for i in range(len(df)) if df.data['game_age'][i] <= 2]
        older_prices = [df.data['msrp'][i] for i in range(len(df)) if df.data['game_age'][i] > 5]
        age_premium = (sum(recent_prices) / len(recent_prices)) - (sum(older_prices) / len(older_prices))
        st.write(f"**Recent vs Older Games Premium:** ${age_premium:.2f}")

def main():
    """Main application"""
    # Header
    st.markdown('<h1 class="main-header">🎮 Nintendo Video Game Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Load the real dataset
    df = load_data()
    
    # Train model
    model, feature_cols = train_model(df)
    
    # Create filters
    selected_genre, selected_platform, year_range, price_range, multiplayer_filter = create_filters(df)
    
    # Apply filters
    filtered_df = filter_data(df, selected_genre, selected_platform, year_range, price_range, multiplayer_filter)
    
    # Show filtered data info
    st.sidebar.success(f"Showing {len(filtered_df):,} games out of {len(df):,} total")
    
    # Data info in sidebar
    st.sidebar.write("**Data Info:**")
    st.sidebar.write(f"- Total games: {len(df):,}")
    st.sidebar.write(f"- MSRP range: ${min(df.data['msrp']):.2f} - ${max(df.data['msrp']):.2f}")
    st.sidebar.write(f"- Year range: {min(df.data['release_year'])} - {max(df.data['release_year'])}")
    
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