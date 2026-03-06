import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import holidays

model = joblib.load('rf_model.pkl')
features = joblib.load('features.pkl')
hourly_data = pd.read_csv('hourly_data.csv', index_col='Opened', parse_dates=True)

def generate_forecast(hourly_data):
    last_known = hourly_data.iloc[-168:].copy()
    last_timestamp = hourly_data.index[-1]
    future_index = pd.date_range(start=last_timestamp + pd.Timedelta(hours=1), periods=168, freq='h')
    future_df = pd.DataFrame(index=future_index)
    future_df['hour'] = future_df.index.hour
    future_df['day_of_week'] = future_df.index.dayofweek
    future_df['is_weekend'] = future_df['day_of_week'].isin([5, 6]).astype(int)
    years = future_df.index.year.unique()
    us_holidays = holidays.US(years=years)
    future_df['is_holiday'] = future_df.index.normalize().isin(us_holidays).astype(int)
    future_df['hour_sin'] = np.sin(2 * np.pi * future_df['hour'] / 24)
    future_df['hour_cos'] = np.cos(2 * np.pi * future_df['hour'] / 24)
    future_df['guests_lag168'] = last_known['# of Guests'].values
    future_df['guests_lag24'] = hourly_data['# of Guests'].iloc[-24:].tolist() * 7
    future_df['guests_lag1'] = hourly_data['# of Guests'].iloc[-168:].shift(1).fillna(0).tolist()
    future_df['guests_lag2'] = hourly_data['# of Guests'].iloc[-168:].shift(2).fillna(0).tolist()
    future_df['predicted_guests'] = model.predict(future_df[features])
    future_df['predicted_guests'] = future_df['predicted_guests'].clip(lower=0).round(2)
    return future_df

forecast_df = generate_forecast(hourly_data)

# App layout
st.title("🍜 Customer Traffic Dashboard")

# Date picker
all_dates = pd.concat([
    pd.Series(hourly_data.index.normalize().unique()),
    pd.Series(forecast_df.index.normalize().unique())
]).drop_duplicates().sort_values()

selected_date = st.date_input(
    "Select Date for Prediction:",
    value=forecast_df.index.normalize().unique()[0],
    min_value=all_dates.min(),
    max_value=all_dates.max()
)

selected_date = pd.Timestamp(selected_date)
is_forecast = selected_date in forecast_df.index.normalize()

if is_forecast:
    day_data = forecast_df[forecast_df.index.normalize() == selected_date]['predicted_guests']
    hist_data = None
else:
    day_data = hourly_data[hourly_data.index.normalize() == selected_date]['# of Guests']
    hist_data = None

# Calculate historical average for the same day of week
dow = selected_date.dayofweek
hist_avg = hourly_data[hourly_data.index.dayofweek == dow].groupby(hourly_data[hourly_data.index.dayofweek == dow].index.hour)['# of Guests'].mean()

# Metrics
total_guests = int(day_data.sum())
peak_hour_val = day_data.idxmax() if day_data.sum() > 0 else None
peak_hour_str = f"{peak_hour_val.strftime('%I %p')}" if peak_hour_val else "N/A"
staff_hours = round(total_guests * 0.75)

col1, col2, col3 = st.columns(3)
col1.metric("📅 Selected Date", selected_date.strftime('%m/%d/%Y'))
col2.metric("🔵 Predicted Peak Traffic", f"{total_guests} customers")
col3.metric("🟢 Suggested Labor (Hours)", f"{staff_hours} staff-hours")

# Chart
st.subheader(f"Hourly Traffic for {selected_date.strftime('%a %b %d %Y')}")

fig, ax = plt.subplots(figsize=(12, 5))

hours = range(24)
predicted_vals = [day_data[day_data.index.hour == h].values[0] if h in day_data.index.hour else 0 for h in hours]
hist_vals = [hist_avg[h] if h in hist_avg.index else 0 for h in hours]

x = np.arange(24)
width = 0.4

ax.bar(x - width/2, predicted_vals, width, label='Predicted Traffic', color='steelblue')
ax.bar(x + width/2, hist_vals, width, label='Historical Average', color='lightgray')

ax.set_xticks(x)
ax.set_xticklabels([f"{h}h" for h in hours], rotation=45)
ax.set_ylabel("Number of Customers")
ax.set_xlabel("Hour")
ax.legend()
ax.set_title(f"Hourly Traffic for {selected_date.strftime('%a %b %d %Y')}")
plt.tight_layout()
st.pyplot(fig)

