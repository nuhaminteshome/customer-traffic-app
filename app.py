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

# Business hours only (11am - 8pm)
BUSINESS_HOURS = list(range(11, 21))
hour_labels = ['11 AM', '12 PM', '1 PM', '2 PM', '3 PM', '4 PM', '5 PM', '6 PM', '7 PM', '8 PM']

st.title("🍜 Customer Traffic Dashboard")

# Top row: date picker | peak hour
col_left, col_right = st.columns([1, 1])

all_dates = pd.concat([
    pd.Series(hourly_data.index.normalize().unique()),
    pd.Series(forecast_df.index.normalize().unique())
]).drop_duplicates().sort_values()

with col_left:
    selected_date = st.date_input(
        "Select a date",
        value=forecast_df.index.normalize().unique()[0],
        min_value=all_dates.min(),
        max_value=all_dates.max()
    )

selected_date = pd.Timestamp(selected_date)
is_forecast = selected_date in forecast_df.index.normalize()

if is_forecast:
    day_data = forecast_df[forecast_df.index.normalize() == selected_date]['predicted_guests']
    label = "Predicted # of Guests"
    tag = "📅 Forecast"
else:
    day_data = hourly_data[hourly_data.index.normalize() == selected_date]['# of Guests']
    label = "Actual # of Guests"
    tag = "📋 Historical"

# Filter to business hours only
day_data_biz = day_data[day_data.index.hour.isin(BUSINESS_HOURS)]

# Peak hour
if not day_data_biz.empty and day_data_biz.sum() > 0:
    peak_hour = day_data_biz.idxmax()
    peak_hour_str = peak_hour.strftime('%I:%M %p').lstrip('0')
    peak_guests = round(day_data_biz.max(), 1)
else:
    peak_hour_str = 'N/A'
    peak_guests = 0

with col_right:
    st.metric("🔥 Predicted Peak Hour", peak_hour_str, f"{peak_guests} guests")

# Summary metrics
st.subheader(f"{tag} — {selected_date.strftime('%A, %B %d %Y')}")
col1, col2, col3 = st.columns(3)
col1.metric("Total Guests", int(day_data_biz.sum()))
col2.metric("Avg Guests/Hour", round(day_data_biz.mean(), 2))
col3.metric("Busiest Period", "Lunch" if day_data_biz.idxmax().hour < 15 else "Dinner" if not day_data_biz.empty and day_data_biz.sum() > 0 else "N/A")

# Bar chart - business hours only
st.subheader("Hourly Traffic (11 AM – 8 PM)")
fig, ax = plt.subplots(figsize=(12, 4))
values = [day_data_biz[day_data_biz.index.hour == h].values[0] if len(day_data_biz[day_data_biz.index.hour == h]) > 0 else 0 for h in BUSINESS_HOURS]
ax.bar(range(len(BUSINESS_HOURS)), values, color=[
    'green' if v < 2 else 'orange' if v < 4 else 'red' for v in values
])
ax.set_xticks(range(len(BUSINESS_HOURS)))
ax.set_xticklabels(hour_labels, rotation=30)
ax.set_ylabel(label)
ax.set_title(f"Hourly Traffic — {selected_date.strftime('%A, %B %d %Y')}")
st.pyplot(fig)

# Weekly forecast overview
st.subheader("📊 7-Day Forecast Overview")
weekly = forecast_df.groupby(forecast_df.index.normalize())['predicted_guests'].sum()
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.bar(range(len(weekly)), weekly.values, color='orange')
ax2.set_xticks(range(len(weekly)))
ax2.set_xticklabels([d.strftime('%a %b %d') for d in weekly.index], rotation=30)
ax2.set_ylabel("Total Predicted Guests")
ax2.set_title("Total Predicted Guests per Day (Next 7 Days)")
st.pyplot(fig2)
