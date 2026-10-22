# 🎮 Nintendo Video Game Analytics Dashboard - Local Version

A simplified Streamlit dashboard for Nintendo video game analytics that runs locally with sample data.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_local.txt
```

### 2. Run the Dashboard
```bash
streamlit run local_dashboard.py
```

### 3. Open Your Browser
The dashboard will automatically open at `http://localhost:8501`

## 📊 Features

### Analytics Dashboard
- **Interactive Filters**: Filter by genre, platform, release year, price range, and multiplayer status
- **Key Metrics**: Total games, sequel count, average price, multiplayer percentage
- **Visualizations**: 
  - Price distribution histogram
  - Release year distribution
  - Sequel vs non-sequel price comparison
  - Multiplayer distribution pie chart

### Sequel Predictions
- **Machine Learning Model**: Random Forest classifier trained on sample data
- **Interactive Prediction Form**: Input game features to predict sequel probability
- **Confidence Levels**: High/Medium/Low likelihood indicators

### Pricing Optimization
- **Genre-based Pricing Analysis**: Price distributions across different genres
- **Multiplayer Premium Analysis**: Price differences between single and multiplayer games
- **Age-based Pricing**: How game age affects pricing
- **Pricing Recommendations**: Data-driven suggestions

## 🎯 Sample Data

The dashboard uses generated sample data that includes:
- 1,000 Nintendo video games
- Realistic pricing patterns (RPG games cost more, Sports games cost less)
- Sequel pricing premiums
- Various genres, platforms, and features

## 🔧 Customization

To use your own data:
1. Replace the `generate_sample_data()` function in `local_dashboard.py`
2. Ensure your data has the required columns:
   - `msrp`, `release_year`, `is_multiplayer`, `is_sequel`
   - `genre_primary_grouped`, `platform_cleaned`
   - Additional features for the ML model

## 🎨 Styling

The dashboard features:
- Nintendo red color scheme (#e60012)
- Clean, modern interface
- Responsive design
- Interactive charts with Plotly

## 📁 Files

- `local_dashboard.py` - Main dashboard application
- `requirements_local.txt` - Python dependencies
- `README_LOCAL.md` - This file

## 🚨 Troubleshooting

If you encounter issues:
1. Make sure all dependencies are installed
2. Check Python version (3.8+ recommended)
3. Ensure you're in the correct directory when running the command

## 🎮 Happy Analyzing!

The dashboard will show all charts working properly with the sample data, including:
- ✅ Price distribution
- ✅ Release year distribution  
- ✅ Sequel price comparison
- ✅ Multiplayer distribution 