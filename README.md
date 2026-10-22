---
title: Nintendo Video Game Analytics Dashboard
emoji: 🎮
colorFrom: red
colorTo: pink
sdk: streamlit
sdk_version: 1.50.0
app_file: app.py
pinned: false
---

# 🎮 Nintendo Video Game Decision Analytics Dashboard

A comprehensive Streamlit dashboard for Nintendo video game sequel likelihood predictions and pricing optimization, designed to be deployed on Hugging Face Spaces.

## 🚀 Features

### 📊 Analytics Dashboard
- **Interactive Filters**: Filter by genre, platform, release year, price range, and multiplayer status
- **Key Metrics**: Total games, sequel count, average price, multiplayer percentage
- **Visualizations**: Price distributions, release year trends, sequel vs non-sequel comparisons
- **Real-time Updates**: All charts update based on selected filters

### 🎯 Sequel Likelihood Predictions
- **Machine Learning Model**: Random Forest classifier trained on historical data
- **Interactive Prediction Form**: Input game features to predict sequel probability
- **Confidence Levels**: High/Medium/Low likelihood indicators with explanations
- **Feature Importance**: Model insights on what drives sequel success

### 💰 Pricing Optimization
- **Genre-based Pricing Analysis**: Price distributions across different genres
- **Multiplayer Premium Analysis**: Price differences between single and multiplayer games
- **Age-based Pricing**: How game age affects pricing
- **Pricing Recommendations**: Data-driven suggestions for optimal pricing

## 🛠️ Installation & Setup

### Local Development

1. **Clone the repository**:
```bash
git clone <your-repo-url>
cd nintendo-video-game-analytics-dashboard
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Prepare your data**:
   - Ensure your `deku_cleaned_final.csv` file is in the same directory
   - The file should contain columns like: `msrp`, `release_year`, `is_multiplayer`, `is_sequel`, `genre_primary_grouped`, `platform_cleaned`, etc.

4. **Run the dashboard**:
```bash
streamlit run streamlit_dashboard.py
```

### 🚀 Deployment on Hugging Face Spaces

1. **Create a new Space**:
   - Go to [Hugging Face Spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Choose "Streamlit" as the SDK
   - Name your space (e.g., "nintendo-video-game-analytics-dashboard")

2. **Upload your files**:
   - Upload `app.py` (main entry point)
   - Upload `requirements.txt`
   - Upload your data file `deku_cleaned_final.csv`
   - Upload this `README.md`

3. **Automatic Deployment**:
   - Hugging Face will automatically detect Streamlit and install dependencies
   - Your dashboard will be available at: `https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME`

## 📁 File Structure

```
nintendo-video-game-analytics-dashboard/
├── app.py                    # Main entry point for Hugging Face Spaces
├── streamlit_dashboard.py    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                # This file with Spaces configuration
├── deku_cleaned_final.csv   # Your dataset (not included in repo)
└── FinalEDA_2_0.py         # Original EDA notebook (for reference)
```

## 🔧 Configuration

### Data Requirements

Your CSV file should contain the following columns:
- `msrp`: Manufacturer's suggested retail price
- `release_year`: Year the game was released
- `is_multiplayer`: Binary indicator (0/1) for multiplayer games
- `is_sequel`: Binary indicator (0/1) for sequel games
- `genre_primary_grouped`: Primary genre category
- `platform_cleaned`: Platform name
- `platform_count`: Number of platforms
- `min_players`: Minimum number of players
- `max_players`: Maximum number of players
- `genre_count`: Number of genres
- `mode_handheld`: Binary indicator for handheld mode
- `mode_docked`: Binary indicator for docked mode
- `mode_tabletop`: Binary indicator for tabletop mode

### Model Configuration

The dashboard uses a Random Forest classifier with the following settings:
- **n_estimators**: 100
- **random_state**: 42
- **class_weight**: 'balanced' (to handle imbalanced sequel data)

## 🎨 Customization

### Styling
The dashboard uses custom CSS for styling. You can modify the styles in the `st.markdown()` section at the top of the script.

### Features
To add new features:
1. **New Filters**: Add to the `create_filters()` function
2. **New Visualizations**: Add to the `analytics_dashboard()` function
3. **New Predictions**: Modify the `sequel_prediction_section()` function
4. **New Pricing Analysis**: Extend the `pricing_optimization_section()` function

### Model Improvements
To improve the prediction model:
1. **Feature Engineering**: Add new features in the `train_model()` function
2. **Hyperparameter Tuning**: Modify the RandomForestClassifier parameters
3. **Model Selection**: Try different algorithms (XGBoost, LightGBM, etc.)

## 📊 Dashboard Sections

### 1. Analytics Dashboard
- **Overview Metrics**: Key statistics about the dataset
- **Price Analysis**: Distribution and trends in game pricing
- **Release Trends**: Temporal analysis of game releases
- **Genre Analysis**: Distribution across different genres
- **Platform Analysis**: Platform-specific insights

### 2. Sequel Predictions
- **Prediction Form**: Interactive form for new game predictions
- **Probability Display**: Visual probability indicators
- **Confidence Levels**: High/Medium/Low confidence explanations
- **Feature Importance**: Model interpretability insights

### 3. Pricing Optimization
- **Genre-based Pricing**: Price analysis by genre
- **Multiplayer Premium**: Price differences for multiplayer games
- **Age-based Pricing**: How game age affects pricing
- **Recommendations**: Data-driven pricing suggestions

## 🔍 Usage Examples

### Filtering Data
1. Use the sidebar filters to narrow down your analysis
2. Select specific genres, platforms, or year ranges
3. Adjust price ranges to focus on specific market segments
4. Filter by multiplayer status for targeted analysis

### Making Predictions
1. Navigate to the "Sequel Predictions" tab
2. Fill out the prediction form with game features
3. Click "Predict Sequel Likelihood"
4. Review the probability and confidence level
5. Use the insights to inform business decisions

### Pricing Analysis
1. Go to the "Pricing Optimization" tab
2. Review genre-based pricing trends
3. Analyze multiplayer pricing premiums
4. Consider age-based pricing strategies
5. Apply insights to your pricing strategy

## 🚨 Troubleshooting

### Common Issues

1. **Data File Not Found**:
   - Ensure `deku_cleaned_final.csv` is in the same directory
   - Check file permissions and path

2. **Missing Dependencies**:
   - Run `pip install -r requirements.txt`
   - Check Python version compatibility

3. **Model Training Errors**:
   - Verify all required columns are present in your data
   - Check for missing values in key columns

4. **Hugging Face Deployment Issues**:
   - Ensure all files are uploaded to the Space
   - Check the Space logs for error messages
   - Verify the requirements.txt format

### Performance Optimization

1. **Large Datasets**:
   - Use `@st.cache_data` for data loading
   - Implement data sampling for large datasets
   - Consider using database connections for very large datasets

2. **Model Performance**:
   - Cache the trained model using `@st.cache_resource`
   - Consider model serialization for faster loading
   - Implement model versioning for updates

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with Streamlit for the web interface
- Powered by scikit-learn for machine learning
- Visualizations created with Plotly and Matplotlib
- Deployed on Hugging Face Spaces for easy sharing

## 📞 Support

For questions or issues:
1. Check the troubleshooting section above
2. Review the Hugging Face Spaces documentation
3. Open an issue in the repository
4. Contact the development team

---

**Happy Analyzing! 🎮📊** 