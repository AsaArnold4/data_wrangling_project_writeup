# NFL Timeout Decision Analyzer - Streamlit App

A data-driven web application for analyzing NFL timeout decisions using machine learning. This app helps coaches and analysts evaluate whether to call a timeout in any game situation based on predicted win probability impacts.

## Features

### 📊 Three Analysis Modes

1. **Standard Timeout Decision**
   - Should you call a timeout in this situation?
   - Compares WP with timeout vs without timeout
   - Provides clear recommendations with probability impact

2. **Delay of Game vs Timeout**
   - Should you accept a 5-yard penalty or use a timeout?
   - Evaluates tradeoff between field position and timeout preservation
   - Shows detailed scenario comparison

3. **Batch Analysis**
   - Upload CSV with multiple situations
   - Analyze dozens or hundreds of scenarios at once
   - Export results for further analysis

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or download the files:**
   ```bash
   # You should have these files:
   # - timeout_analyzer_app.py
   # - prepare_model.py
   # - nfl_pbp_full.csv
   ```

2. **Install required packages:**
   ```bash
   pip install streamlit pandas numpy xgboost scikit-learn
   ```

3. **Prepare the model:**
   ```bash
   python prepare_model.py
   ```
   
   This will:
   - Train the XGBoost model on your NFL data
   - Save the model and feature columns
   - Create example situations for testing
   - Takes 5-10 minutes depending on your data size

4. **Launch the app:**
   ```bash
   streamlit run timeout_analyzer_app.py
   ```

5. **Open your browser:**
   - Streamlit will automatically open your default browser
   - Or navigate to: http://localhost:8501

## Usage Guide

### Standard Timeout Analysis

1. Navigate to the "⏱️ Standard Timeout Decision" tab
2. Input the game situation:
   - **Game State**: Quarter, time remaining, score differential
   - **Field Position**: Down, distance, yard line
   - **Team Info**: Team abbreviations, timeouts, play type
3. Click "Analyze Timeout Decision"
4. Review the results:
   - Win probability with timeout
   - Win probability without timeout
   - Timeout boost (difference)
   - Clear recommendation

### Delay of Game Analysis

1. Navigate to the "⚠️ Delay of Game vs Timeout" tab
2. Input the pre-snap situation (when facing delay of game)
3. Click "Analyze Penalty vs Timeout"
4. Review the comparison:
   - WP if using timeout (stay at current line)
   - WP if accepting penalty (lose 5 yards)
   - Advantage calculation
   - Detailed scenario breakdown

### Batch Analysis

1. Navigate to the "📊 Batch Analysis" tab
2. Prepare a CSV file with required columns (see below)
3. Upload the CSV
4. Select analysis type
5. Click "Analyze All Situations"
6. Download results CSV

#### Required CSV Columns for Batch Analysis

```
qtr, game_seconds_remaining, half_seconds_remaining, score_differential,
down, ydstogo, yardline_100, posteam_timeouts_remaining, 
defteam_timeouts_remaining, posteam, defteam, play_type, clock_stop
```

**Example CSV row:**
```csv
qtr,game_seconds_remaining,half_seconds_remaining,score_differential,down,ydstogo,yardline_100,posteam_timeouts_remaining,defteam_timeouts_remaining,posteam,defteam,play_type,clock_stop
4,120,120,-3,3,5,35,2,3,KC,BAL,pass,0
```

## Understanding the Inputs

### Field Position
- **Yard Line**: Enter as "your territory" (e.g., "30" = your own 30)
- The app converts this to yardline_100 automatically

### Score Differential
- **Positive**: Your team is winning (e.g., +7 = winning by 7)
- **Negative**: Your team is losing (e.g., -3 = losing by 3)
- **Zero**: Tied game

### Clock Stop
- **Checked**: Clock is stopped (after incomplete pass, out of bounds, timeout, etc.)
- **Unchecked**: Clock is running

### Play Type
- **pass**: Expected passing play
- **run**: Expected running play
- **qb_scramble**: Quarterback scramble expected

## Model Information

### Training Data
- NFL play-by-play data from nflfastR
- Filtered to late-game situations:
  - Q2 with ≤2 minutes remaining
  - All of Q4
- Excludes garbage time (WP between 5% and 95%)

### Features Used
**Numeric Features:**
- Score differential
- Game/half seconds remaining
- Down, distance, field position
- Timeouts remaining (both teams)
- Clock state (running/stopped)
- Trailing indicator

**Categorical Features:**
- Quarter
- Play type

### Model Performance
The model uses XGBoost regression to predict win probability. After training, you should see metrics like:
- MAE: ~0.05-0.07 (mean absolute error in WP)
- RMSE: ~0.08-0.10 (root mean squared error)
- R²: ~0.85-0.90 (coefficient of determination)

## Interpretation Guide

### Timeout Boost
- **> +1%**: Strong recommendation to use timeout
- **0% to +1%**: Slight advantage to using timeout
- **-1% to 0%**: Slight advantage to not using timeout
- **< -1%**: Strong recommendation to not use timeout

### When Timeouts Help Most
Based on the model, timeouts typically provide the most value when:
1. Clock is running and you need to preserve time
2. You're trailing late in the game
3. You need to avoid a delay of game in critical situations
4. You want to ice the kicker (though model doesn't directly capture this)

### When Timeouts Hurt
Timeouts can be detrimental when:
1. You'll need them more urgently later
2. Accepting a delay of game penalty has minimal impact
3. The clock is already stopped
4. You're winning and want the clock to run

## Troubleshooting

### "Model files not found!"
- Make sure you ran `prepare_model.py` first
- Check that `xgb_timeout_model.pkl` and `feature_columns.pkl` exist

### "Error analyzing situation"
- Verify all input values are within valid ranges
- Check team abbreviations are 2-3 letters
- Ensure timeouts are 0-3

### Model training takes too long
- Reduce `n_estimators` in `prepare_model.py` (line ~80)
- Use a smaller sample of data for initial testing
- Consider using the pre-tuned hyperparameters

### Predictions seem off
- Model is trained on historical data; unusual situations may not be well-represented
- Extreme scores or field positions may have limited training examples
- Consider the model as one input to decision-making, not the only factor

## Customization

### Using Your Own Model
1. Train your model using `TimeoutModelV4.py` with custom hyperparameters
2. Save as `xgb_timeout_model.pkl`
3. Ensure feature columns match
4. Or use the sidebar upload in the app

### Adjusting Recommendation Thresholds
Edit `timeout_analyzer_app.py` to change when recommendations trigger:
```python
# Current thresholds (lines ~320, 340):
if boost > 0.01:  # Change 0.01 to your threshold
    st.success("Call a Timeout")
```

### Styling
Modify the CSS in the `st.markdown()` section (lines ~30-50) to change colors, fonts, etc.

## Files Included

- `timeout_analyzer_app.py` - Main Streamlit application
- `prepare_model.py` - Model training and preparation script
- `README.md` - This file
- `TimeoutModelV4.py` - Original model training script (reference)
- `nfl_pbp_full.csv` - NFL play-by-play data (not included in repo)

## Generated Files (after running prepare_model.py)

- `xgb_timeout_model.pkl` - Trained XGBoost model
- `feature_columns.pkl` - List of feature column names
- `feature_info.pkl` - Feature metadata
- `example_situations.csv` - Example test cases

## Advanced Usage

### Command Line Options
```bash
# Run on different port
streamlit run timeout_analyzer_app.py --server.port 8502

# Run with custom config
streamlit run timeout_analyzer_app.py --server.headless true
```

### API Integration
The app can be deployed to Streamlit Cloud, Heroku, or AWS for team-wide access.

## Credits

Based on the timeout decision model inspired by nfl4th and nflfastR methodologies.

## License

MIT License - Feel free to use and modify for your coaching/analysis needs.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the input requirements
3. Verify model training completed successfully
4. Check Streamlit logs for error messages

## Future Enhancements

Potential additions:
- Historical situation lookup
- Similar situations from past games
- Coach/team-specific timeout tendencies
- Real-time game integration
- Mobile-responsive design improvements
- Situational recommendations (2-minute drill, goal line, etc.)
