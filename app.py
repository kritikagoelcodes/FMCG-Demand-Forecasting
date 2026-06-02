import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

st.set_page_config(
    page_title="FMCG Demand Forecasting Dashboard",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #f8fafc;
}

h1 {
    color: #1f2937;
    font-size: 42px;
}

h2, h3 {
    color: #111827;
}

[data-testid="stMetric"] {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
}

.block-container {
    padding-top: 2rem;
}

hr {
    margin-top: 25px;
    margin-bottom: 25px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    train = pd.read_csv("train.csv")
    stores = pd.read_csv("stores.csv")
    train["date"] = pd.to_datetime(train["date"])
    data = train.merge(stores, on="store_nbr", how="left")
    return train, data


train, data = load_data()

st.sidebar.title("Dashboard Navigation")
st.sidebar.info("""
FMCG Demand Forecasting & Inventory Analytics

Built using:
- Python
- Pandas
- Streamlit
- Matplotlib
- Meta Prophet
""")

st.title("📊 FMCG Demand Forecasting & Inventory Analytics")
st.write("End-to-end analytics and forecasting dashboard built using FMCG retail sales data.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Sales Records", f"{len(train):,}")
col2.metric("Stores", train["store_nbr"].nunique())
col3.metric("Product Families", train["family"].nunique())
col4.metric("Cities", data["city"].nunique())

st.divider()

st.header("📦 Top Product Families by Sales")
family_sales = train.groupby("family")["sales"].sum().sort_values(ascending=False).head(10)

fig, ax = plt.subplots(figsize=(14, 6))
family_sales.plot(kind="bar", ax=ax, color="#4F46E5")
ax.set_xlabel("Product Family")
ax.set_ylabel("Total Sales")
ax.set_title("Top 10 Product Families by Sales")
ax.tick_params(axis="x", rotation=45)
st.pyplot(fig)

st.header("🏙️ Top Cities by Revenue")
city_sales = data.groupby("city")["sales"].sum().sort_values(ascending=False).head(10)

fig, ax = plt.subplots(figsize=(14, 6))
city_sales.plot(kind="barh", ax=ax, color="#10B981")
ax.set_xlabel("Total Sales")
ax.set_ylabel("City")
ax.set_title("Top 10 Cities by Revenue")
st.pyplot(fig)

st.header("🏬 Store Performance & Stockout Risk")
store_sales = data.groupby("store_nbr")["sales"].sum().sort_values(ascending=False)
average_sales = store_sales.mean()

fig, ax = plt.subplots(figsize=(14, 6))
store_sales.plot(kind="bar", ax=ax, color="#F59E0B")
ax.axhline(average_sales, color="red", linestyle="--", label="Average Sales")
ax.set_xlabel("Store Number")
ax.set_ylabel("Total Sales")
ax.set_title("Sales by Store - Stockout Risk Analysis")
ax.legend()
st.pyplot(fig)

st.header("🎯 Promotion Impact Analysis")
promo_data = train.copy()
promo_data["Promo_Flag"] = promo_data["onpromotion"] > 0
promo_avg = promo_data.groupby("Promo_Flag")["sales"].mean()

col1, col2 = st.columns(2)
col1.metric("Avg Sales Without Promotion", round(promo_avg[False], 2))
col2.metric("Avg Sales With Promotion", round(promo_avg[True], 2))

st.write(
    "Products under promotion achieved significantly higher average sales compared to non-promotional products."
)

st.header("📈 Monthly Sales Trend")
monthly_sales = train.groupby(pd.Grouper(key="date", freq="M"))["sales"].sum().reset_index()

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(monthly_sales["date"], monthly_sales["sales"], color="#2563EB", linewidth=2)
ax.set_xlabel("Date")
ax.set_ylabel("Total Sales")
ax.set_title("Monthly Sales Trend")
ax.grid(True, alpha=0.3)
st.pyplot(fig)

st.header("🔮 90-Day Demand Forecast using Prophet")

daily_sales = train.groupby("date")["sales"].sum().reset_index()
prophet_data = daily_sales.rename(columns={"date": "ds", "sales": "y"})

with st.spinner("Training forecasting model..."):
    model = Prophet()
    model.fit(prophet_data)
    future = model.make_future_dataframe(periods=90)
    forecast = model.predict(future)

fig1 = model.plot(forecast)
plt.title("90-Day FMCG Sales Forecast")
st.pyplot(fig1)

st.subheader("Forecast Components")
fig2 = model.plot_components(forecast)
st.pyplot(fig2)

st.header("📌 Key Business Insights")
st.markdown("""
- Grocery I and Beverages are the highest-selling product families.
- Quito generated the highest city-level revenue.
- Store 44 was the highest-performing store.
- Products under promotion achieved around 7x higher average sales.
- Sales show a long-term upward trend from 2013 to 2017.
- December shows strong seasonal demand.
- Weekend sales patterns are stronger than mid-week demand.
""")