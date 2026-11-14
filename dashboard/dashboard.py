import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk
from scipy import stats



MIN_VALID_HOURS = 18
PM25_THRESHOLD = 75
MAX_SCATTER_POINTS = 20000

COLOR_MAX = "#FF0000"
COLOR_BASE = "#2E98BF"

SEASON_MAP = {
    12: "Musim Dingin", 1: "Musim Dingin", 2: "Musim Dingin",
    3: "Musim Semi", 4: "Musim Semi", 5: "Musim Semi",
    6: "Musim Panas", 7: "Musim Panas", 8: "Musim Panas",
    9: "Musim Gugur", 10: "Musim Gugur", 11: "Musim Gugur"
}

SEASON_ORDER = ["Musim Semi", "Musim Panas", "Musim Gugur", "Musim Dingin"]

MONTH_LABELS = {
    1: "Januari", 2: "Februari", 3: "Maret", 4: "April",
    5: "Mei", 6: "Juni", 7: "Juli", 8: "Agustus",
    9: "September", 10: "Oktober", 11: "November", 12: "Desember"
}

WEATHER_COLS = ["WSPM", "TEMP", "PRES", "DEWP", "RAIN"]

STATION_INFO = {
    "Aotizhongxin": {"latitude": 39.982, "longitude": 116.397, "station_type": "Main Urban"},
    "Changping": {"latitude": 40.217, "longitude": 116.230, "station_type": "Sub Urban"},
    "Dingling": {"latitude": 40.292, "longitude": 116.220, "station_type": "Sub Urban"},
    "Dongsi": {"latitude": 39.929, "longitude": 116.417, "station_type": "Main Urban"},
    "Guanyuan": {"latitude": 39.929, "longitude": 116.339, "station_type": "Main Urban"},
    "Gucheng": {"latitude": 39.914, "longitude": 116.184, "station_type": "Main Urban"},
    "Huairou": {"latitude": 40.328, "longitude": 116.628, "station_type": "Sub Urban"},
    "Nongzhanguan": {"latitude": 39.937, "longitude": 116.461, "station_type": "Main Urban"},
    "Shunyi": {"latitude": 40.127, "longitude": 116.655, "station_type": "Sub Urban"},
    "Tiantan": {"latitude": 39.886, "longitude": 116.407, "station_type": "Main Urban"},
    "Wanliu": {"latitude": 39.987, "longitude": 116.287, "station_type": "Main Urban"},
    "Wanshouxigong": {"latitude": 39.878, "longitude": 116.352, "station_type": "Main Urban"}
}

PM25_BINS = [0, 35, 75, 115, 150, 250, np.inf]
PM25_LABELS = [
    "Excellent", "Good", "Slight pollution",
    "Moderate pollution", "Heavy pollution", "Severe pollution"
]
PM25_COLORS = {
    "Excellent": "#2E8B57",
    "Good": "#A8D672",
    "Slight pollution": "#F4D03F",
    "Moderate pollution": "#F39C12",
    "Heavy pollution": "#E74C3C",
    "Severe pollution": "#8E44AD"
}


@st.cache_data
def load_data():
    df = pd.read_csv("main_data.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df

def create_data_pm_df(df):
    data_pm = df.dropna(subset=["PM2.5"]).copy()
    period_start_year = data_pm["year"] - data_pm["month"].lt(3).astype(int)

    data_pm["period"] = (
        period_start_year.astype(str) + "–" + (period_start_year + 1).astype(str)
    )
    data_pm["season"] = pd.Categorical(
        data_pm["month"].map(SEASON_MAP),
        categories=SEASON_ORDER,
        ordered=True
    )
    return data_pm

def create_group_stats_df(df, by, order=None):
    result = df.groupby(by, observed=True)["PM2.5"].mean().to_frame("mean")
    return result.reindex(order) if order is not None else result

def create_year_month_avg_df(df):
    return (
        df.pivot_table(index="year", columns="month", values="PM2.5", aggfunc="mean")
        .reindex(columns=range(1, 13))
        .round(2)
    )

def create_daily_stats_df(df):
    station = df[["datetime", "station", "PM2.5"]].copy()
    station["tanggal"] = station["datetime"].dt.normalize()

    daily_stats = (
        station.groupby(["station", "tanggal"], observed=True)["PM2.5"]
        .agg(daily_mean="mean", valid_hours="count")
        .reset_index()
    )
    daily_stats = daily_stats.loc[daily_stats["valid_hours"] >= MIN_VALID_HOURS].copy()
    daily_stats["high_day"] = daily_stats["daily_mean"] > PM25_THRESHOLD
    return daily_stats

def create_high_day_summary_df(daily_stats):
    result = daily_stats.groupby("station", observed=True).agg(
        hari_valid=("tanggal", "count"),
        hari_tinggi=("high_day", "sum")
    )
    result["pct_hari_tinggi"] = result["hari_tinggi"] / result["hari_valid"] * 100
    return result

def create_station_stats_df(data_pm, high_day_summary):
    result = (
        data_pm.groupby("station", observed=True)["PM2.5"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .join(high_day_summary, how="left")
    )

    overall_mean = data_pm["PM2.5"].mean()
    average_high_pct = result["pct_hari_tinggi"].mean()
    idx_mean = result["mean"] / overall_mean * 100

    if pd.notna(average_high_pct) and average_high_pct > 0:
        idx_high_pct = result["pct_hari_tinggi"] / average_high_pct * 100
    elif pd.notna(average_high_pct):
        idx_high_pct = result["pct_hari_tinggi"].where(
            result["pct_hari_tinggi"].isna(), 100
        )
    else:
        idx_high_pct = result["pct_hari_tinggi"]

    result["indeks_gabungan"] = (idx_mean + idx_high_pct) / 2
    result["rank_gabungan"] = result["indeks_gabungan"].rank(
        ascending=False, method="min"
    ).astype("Int64")
    return result

def create_weather_df(df):
    return df[["PM10"] + WEATHER_COLS].copy()

def create_regression_df(weather):
    rows = []

    for column in WEATHER_COLS:
        group = weather[[column, "PM10"]].dropna()
        if len(group) < 2 or group[column].nunique() < 2:
            continue

        fit = stats.linregress(group[column], group["PM10"])
        rows.append({
            "Variabel": column,
            "slope": fit.slope,
            "intercept": fit.intercept,
            "r2": fit.rvalue ** 2,
            "p_value": fit.pvalue,
            "n": len(group)
        })

    if rows:
        return pd.DataFrame(rows).set_index("Variabel")

    return pd.DataFrame(
        columns=["slope", "intercept", "r2", "p_value", "n"]
    ).rename_axis("Variabel")

def create_station_geo_df(daily_stats):
    station_geo = daily_stats.copy()

    for column in ["latitude", "longitude", "station_type"]:
        station_geo[column] = station_geo["station"].map(
            lambda value: STATION_INFO.get(value, {}).get(column)
        )

    station_geo = (
        station_geo.groupby(
            ["station", "station_type", "latitude", "longitude"],
            observed=True
        )
        .agg(mean_pm25=("daily_mean", "mean"))
        .reset_index()
        .sort_values("mean_pm25", ascending=False)
    )

    station_geo["kategori_pm25"] = pd.cut(
        station_geo["mean_pm25"],
        bins=PM25_BINS,
        labels=PM25_LABELS,
        include_lowest=True,
        right=True
    )
    station_geo["color"] = station_geo["kategori_pm25"].astype(str).map(PM25_COLORS)
    return station_geo.dropna(subset=["latitude", "longitude"])


def get_colors(values):
    values = np.asarray(values, dtype=float)
    return [
        COLOR_MAX if value == np.nanmax(values) else COLOR_BASE
        for value in values
    ]


def show_line_chart(series, xlabel, ylabel, x_labels=None):
    values = series.astype(float).to_numpy()
    labels = x_labels if x_labels is not None else series.index.astype(str)
    positions = np.arange(len(values))

    fig, ax = plt.subplots(figsize=(11, 4))

    ax.plot(positions, values, marker="o", color=COLOR_BASE)

    if np.isfinite(values).any():
        max_pos = int(np.nanargmax(values))
        ax.scatter(max_pos, values[max_pos], color=COLOR_MAX, s=100, zorder=5)

    for x, y in zip(positions, values):
        if np.isfinite(y):
            ax.annotate(f"{y:.1f}", (x, y), xytext=(0, 9),
                        textcoords="offset points", ha="center")

    ax.set(xticks=positions, xticklabels=labels, xlabel=xlabel, ylabel=ylabel)
    ax.set_ylim(bottom=0)
    ax.set_yticks(range(0, 131, 20))

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def show_bar_chart(series, xlabel, ylabel, sort=False, yticks=None):
    series = series.dropna()

    if sort:
        series = series.sort_values(ascending=False)

    if series.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 4))
    bars = ax.bar(series.index.astype(str), series.values,
                  color=get_colors(series.values))

    ax.set(xlabel=xlabel, ylabel=ylabel)

    if yticks is not None:
        ax.set_yticks(yticks)

    ax.tick_params(axis="x", rotation=45 if sort else 0)
    ax.bar_label(bars, fmt="%.2f", padding=3)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def show_correlation_heatmap(matrix):
    fig, ax = plt.subplots(figsize=(7, 6))

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        cbar=False,
        ax=ax
    )

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def show_correlation_comparison(correlation_result):
    plot_data = correlation_result[["Pearson", "Spearman"]].copy()

    plot_data = plot_data.reindex(WEATHER_COLS)
    plot_data = plot_data.dropna(how="all")

    if plot_data.empty:
        st.warning("Tidak ada data korelasi yang dapat ditampilkan.")
        return

    y = np.arange(len(plot_data))
    height = 0.35

    fig, ax = plt.subplots(figsize=(11, 5.5))

    bars_pearson = ax.barh(
        y + height / 2,
        plot_data["Pearson"],
        height,
        label="Pearson",
        color="#4C84B8"
    )

    bars_spearman = ax.barh(
        y - height / 2,
        plot_data["Spearman"],
        height,
        label="Spearman",
        color="#FFA500"
    )

    ax.axvline(0, color="black", linewidth=0.8)

    for bars in [bars_pearson, bars_spearman]:
        for bar in bars:
            value = bar.get_width()

            if pd.isna(value):
                continue

            offset = 0.015 if value >= 0 else -0.015

            ax.text(
                value + offset,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                va="center",
                ha="left" if value >= 0 else "right",
                fontsize=10
            )

    ax.set_yticks(y)
    ax.set_yticklabels(plot_data.index)

    ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel("Koefisien Korelasi")
    ax.set_ylabel("Variabel Meteorologis")

    ax.legend(title="Metode")
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def show_regression_chart(weather, regression_df, variable):
    if variable not in regression_df.index:
        st.warning(f"Data untuk {variable} tidak cukup untuk regresi.")
        return

    pair = weather[[variable, "PM10"]].dropna()

    if pair.empty:
        st.warning(f"Data untuk {variable} tidak cukup untuk regresi.")
        return

    plot_pair = (
        pair.sample(MAX_SCATTER_POINTS, random_state=42)
        if len(pair) > MAX_SCATTER_POINTS else pair
    )

    row = regression_df.loc[variable]
    x_line = np.array([pair[variable].min(), pair[variable].max()])
    y_line = row["slope"] * x_line + row["intercept"]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.scatter(plot_pair[variable], plot_pair["PM10"],
               alpha=0.2, s=12, color=COLOR_BASE)

    ax.plot(x_line, y_line, color="red",
            label=f"R² = {row['r2']:.3f}")

    ax.set(xlabel=variable, ylabel="PM10")
    ax.set_ylim(bottom=0)
    ax.legend()

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def hex_to_rgb(value):
    try:
        value = str(value).lstrip("#")
        return [int(value[i:i + 2], 16) for i in (0, 2, 4)]
    except (ValueError, IndexError):
        return [128, 128, 128]


try:
    df = load_data()
except FileNotFoundError:
    st.error("File main_data.csv harus berada satu folder dengan file dashboard.")
    st.stop()

REQUIRED_COLS = (
    ["datetime", "year", "month", "hour", "station", "PM2.5", "PM10"]
    + WEATHER_COLS
)
missing_cols = [column for column in REQUIRED_COLS if column not in df.columns]

if missing_cols:
    st.error("Kolom tidak ditemukan: " + ", ".join(missing_cols))
    st.stop()

min_date, max_date = df["datetime"].min().date(), df["datetime"].max().date()


sns.set_theme(style="dark")
st.set_page_config(page_title="Analisis Kualitas Udara Beijing", layout="wide")

with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; justify-content: center;">
            <img src="https://raw.githubusercontent.com/Puturahmaa7/air-quality-dataset-missing-value-analysis/547566e7f2da4c3c2698dfd3e691ffb98392f3e7/LogoAnalisisKualitasUdaraBeijing.png" width="120">
        </div>
        """,
        unsafe_allow_html=True
    )
    st.title("Filter Data")
    selected_station = st.selectbox(
        "Stasiun",
        ["Semua Stasiun"] + sorted(df["station"].dropna().unique().tolist())
    )
    st.divider()
    st.title("Filter Tanggal")
    date_range = st.date_input(
        "Rentang Waktu", (min_date, max_date), min_date, max_date,
        format="MM.DD.YYYY", key="date_filter"
    )


data = df.copy()

if len(date_range) == 2:
    data = data[data["datetime"].dt.date.between(date_range[0], date_range[1])]
elif len(date_range) == 1:
    data = data[data["datetime"].dt.date == date_range[0]]

if selected_station != "Semua Stasiun":
    data = data[data["station"] == selected_station]

if data.empty:
    st.warning("Tidak ada data pada filter yang dipilih.")
    st.stop()

data_pm = create_data_pm_df(data)

if data_pm.empty:
    st.warning("Tidak ada data PM2.5 pada filter yang dipilih.")
    st.stop()


year_stats = create_group_stats_df(data_pm, "year")
period_stats = create_group_stats_df(data_pm, "period")
month_stats = create_group_stats_df(data_pm, "month", range(1, 13))
season_stats = create_group_stats_df(data_pm, "season", SEASON_ORDER)
hour_stats = create_group_stats_df(data_pm, "hour", range(24))
year_month_avg = create_year_month_avg_df(data_pm)

daily_stats = create_daily_stats_df(data_pm)
high_day_summary = create_high_day_summary_df(daily_stats)
station_stats = create_station_stats_df(data_pm, high_day_summary)

weather = create_weather_df(data)
pearson_matrix = weather.corr(method="pearson")
spearman_matrix = weather.corr(method="spearman")

pearson_corr = pearson_matrix["PM10"].drop("PM10")
spearman_corr = spearman_matrix["PM10"].drop("PM10")
paired_valid_count = (
    weather.notna().astype(int).T.dot(weather.notna().astype(int))
    .loc[WEATHER_COLS, "PM10"]
)

correlation_result = pd.DataFrame({
    "Pearson": pearson_corr,
    "Spearman": spearman_corr,
    "Jumlah Pasangan Valid": paired_valid_count
}).sort_values("Pearson", key=lambda values: values.abs(), ascending=False)

regression_df = create_regression_df(weather)
station_geo = create_station_geo_df(daily_stats)

highest_year = year_stats["mean"].idxmax()
highest_period = period_stats["mean"].idxmax()
highest_month = month_stats["mean"].idxmax()
highest_season = season_stats["mean"].idxmax()
highest_hour = hour_stats["mean"].idxmax()
highest_mean_station = station_stats["mean"].idxmax()
highest_station = station_geo.loc[station_geo["mean_pm25"].idxmax()]


st.header("Dashboard Analisis Kualitas Udara Beijing")
st.caption(
    "Analisis pola PM2.5 berdasarkan waktu, perbandingan tingkat polusi "
    "antarstasiun, dan hubungan faktor meteorologi dengan PM10 di Beijing."
)

c1, c2, c3 = st.columns(3)
c1.metric("Rata-rata PM2.5", f"{data_pm['PM2.5'].mean():.1f} µg/m³")
c2.metric("Rata-rata PM10", f"{data['PM10'].mean():.1f} µg/m³")
c3.metric("Jumlah Stasiun", data["station"].nunique())

tab1, tab2, tab3, tab4 = st.tabs([
    "Analisis Temporal PM2.5",
    "Analisis PM2.5 Antarstasiun",
    "Analisis Hubungan Meteorologi dan PM10",
    "Analisis Geospasial"
])


with tab1:
    temporal_charts = [
        ("Rata-rata PM2.5 per Tahun", year_stats, "Tahun", None,
         highest_year, f"{highest_year}", "tahun"),
        ("Rata-rata PM2.5 per Periode (Maret–Februari)", period_stats,
         "Periode", None, highest_period, str(highest_period), "periode"),
        ("Rata-rata PM2.5 per Bulan", month_stats, "Bulan",
         [MONTH_LABELS[m][:3] for m in month_stats.index],
         highest_month, MONTH_LABELS[highest_month], "bulan"),
        ("Rata-rata PM2.5 per Jam", hour_stats, "Jam",
         [f"{h:02d}" for h in hour_stats.index],
         highest_hour, f"{highest_hour:02d}:00", "pukul")
    ]

    for heading, result, xlabel, labels, highest, label, unit in temporal_charts:
        st.subheader(heading)
        show_line_chart(result["mean"], xlabel, "Rata-rata PM2.5 (µg/m³)", labels)
        st.markdown(
            f"Rata-rata tertinggi terjadi pada {unit} **{label}** "
            f"sebesar **{result.loc[highest, 'mean']:.2f} µg/m³**."
        )

    st.subheader("Rata-rata PM2.5 per Musim")

    show_bar_chart(
    season_stats["mean"],
    "Musim",
    "Rata-rata PM2.5 (µg/m³)",
    yticks=range(0, 121, 20)
    )

    st.markdown(
        f"Rata-rata tertinggi terjadi pada **{highest_season}** "
        f"sebesar **{season_stats.loc[highest_season, 'mean']:.2f} µg/m³**."
    )

    st.subheader("Rata-rata PM2.5 per Tahun dan Bulan")
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(
        year_month_avg, annot=True, fmt=".1f", cmap="YlOrRd",
        linewidths=0.5, ax=ax,
        cbar_kws={"label": "Rata-rata PM2.5"}
    )
    ax.set(xlabel="Bulan", ylabel="Tahun")
    ax.set_xticklabels([MONTH_LABELS[m][:3] for m in year_month_avg.columns], rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


with tab2:
    st.subheader("Rata-rata PM2.5 Antarstasiun")
    show_bar_chart(
    station_stats["mean"],
    "Stasiun",
    "Rata-rata PM2.5 (µg/m³)",
    sort=True,
    yticks=range(0, 121, 20)
    )
    st.markdown(
        f"Stasiun **{highest_mean_station}** memiliki rata-rata tertinggi, "
        f"yaitu **{station_stats.loc[highest_mean_station, 'mean']:.2f} µg/m³**."
    )

    st.subheader(f"Persentase Hari dengan PM2.5 > {PM25_THRESHOLD}")

    if station_stats["pct_hari_tinggi"].notna().any():
        show_bar_chart(
            station_stats["pct_hari_tinggi"],
            "Stasiun",
            "Persentase Hari (%)",
            sort=True,
            yticks=range(0, 101, 20)
        )

        highest_high_day_station = station_stats["pct_hari_tinggi"].idxmax()
        st.markdown(
            f"Stasiun **{highest_high_day_station}** memiliki persentase tertinggi, "
            f"yaitu **{station_stats.loc[highest_high_day_station, 'pct_hari_tinggi']:.2f}%** "
            f"(hanya menghitung hari dengan minimal {MIN_VALID_HOURS} jam data valid)."
        )

        st.subheader("Indeks Gabungan")

        show_bar_chart(
            station_stats["indeks_gabungan"],
            "Stasiun",
            "Indeks Gabungan",
            sort=True,
            yticks=range(0, 141, 20)
        )

        highest_combined_station = station_stats["indeks_gabungan"].idxmax()
        st.markdown(
            f"Stasiun **{highest_combined_station}** memiliki indeks gabungan tertinggi, "
            f"yaitu **{station_stats.loc[highest_combined_station, 'indeks_gabungan']:.2f}**."
        )
    else:
        st.warning(
            f"Tidak ada hari dengan minimal {MIN_VALID_HOURS} jam data valid "
            "pada filter yang dipilih."
        )

    st.subheader("Ringkasan Statistik Antarstasiun")
    display_station_df = station_stats[[
        "count", "mean", "median", "std", "min", "max",
        "hari_valid", "hari_tinggi", "pct_hari_tinggi",
        "indeks_gabungan", "rank_gabungan"
    ]].copy()

    display_station_df.columns = [
        "Jumlah Data", "Mean", "Median", "Std", "Min", "Max",
        "Hari Valid", "Hari Tinggi", "Persentase Hari Tinggi (%)",
        "Indeks Gabungan", "Rank Gabungan"
    ]
    display_station_df.index.name = "Stasiun"
    st.dataframe(display_station_df.sort_values("Rank Gabungan").round(2))


with tab3:
    if correlation_result["Pearson"].isna().all():
        st.warning("Data valid tidak cukup untuk melakukan analisis korelasi.")
    else:
        st.subheader("Matriks Korelasi")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Pearson**")
            show_correlation_heatmap(pearson_matrix)

        with col2:
            st.markdown("**Spearman**")
            show_correlation_heatmap(spearman_matrix)

        st.subheader("Perbandingan Korelasi Pearson dan Spearman")
        show_correlation_comparison(correlation_result)

        strongest_var = correlation_result["Pearson"].abs().idxmax()
        strongest_value = correlation_result.loc[strongest_var, "Pearson"]
        st.markdown(
            f"Variabel dengan korelasi Pearson terkuat terhadap PM10 adalah "
            f"**{strongest_var}** (r = **{strongest_value:.3f}**). "
            "Korelasi tidak membuktikan hubungan sebab-akibat."
        )

        if not regression_df.empty:
            st.subheader("Regresi Linear Sederhana")

            st.subheader("Hubungan PM10 dengan Variabel Meteorologi")
            st.caption(
                f"Scatter plot memakai sampel maksimal {MAX_SCATTER_POINTS:,} titik; "
                "garis regresi dihitung dari seluruh data."
            )

            for i in range(0, len(WEATHER_COLS), 2):
                cols = st.columns(2)
                for col, variable in zip(cols, WEATHER_COLS[i:i + 2]):
                    with col:
                        show_regression_chart(weather, regression_df, variable)


with tab4:
    if station_geo.empty:
        st.warning(
            f"Tidak ada hari dengan minimal {MIN_VALID_HOURS} jam data valid "
            "pada filter yang dipilih."
        )
    else:
        st.subheader("Peta Stasiun Pemantauan")
        map_df = station_geo.copy()
        map_df["rgb"] = map_df["color"].map(hex_to_rgb)
        map_df["radius"] = 800 + map_df["mean_pm25"] * 15
        map_df["kategori"] = map_df["kategori_pm25"].astype(str)
        map_df["mean_label"] = map_df["mean_pm25"].round(2)

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df[[
                "station", "station_type", "latitude", "longitude",
                "rgb", "radius", "kategori", "mean_label"
            ]],
            get_position="[longitude, latitude]",
            get_fill_color="rgb",
            get_radius="radius",
            opacity=0.85,
            pickable=True
        )

        view_state = pdk.ViewState(
            latitude=float(map_df["latitude"].mean()),
            longitude=float(map_df["longitude"].mean()),
            zoom=8.5 if len(map_df) > 1 else 10
        )

        st.pydeck_chart(pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            map_style="light",
            tooltip={
                "html": (
                    "<b>{station}</b><br/>"
                    "Tipe: {station_type}<br/>"
                    "Rata-rata PM2.5: {mean_label} µg/m³<br/>"
                    "Kategori: {kategori}"
                )
            }
        ))

        st.markdown(
            " &nbsp; ".join(
                f'<span style="color:{color}; font-size:18px;">●</span> {label}'
                for label, color in PM25_COLORS.items()
            ),
            unsafe_allow_html=True
        )

        st.subheader("Rata-rata Harian PM2.5 Berdasarkan Stasiun")
        station_order = station_geo.sort_values("mean_pm25", ascending=False)["station"]

        fig, ax = plt.subplots(figsize=(13, 6))
        sns.barplot(
            data=station_geo,
            x="station",
            y="mean_pm25",
            hue="station_type",
            order=station_order,
            dodge=False,
            palette={
                "Main Urban": COLOR_MAX,
                "Sub Urban": COLOR_BASE
            },
            ax=ax
        )

        for bar in ax.patches:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, height,
                    f"{height:.2f}", ha="center", va="bottom", fontsize=10
                )

        ax.set(xlabel="Stasiun", ylabel="Rata-rata Harian PM2.5 (µg/m³)")
        ax.tick_params(axis="x", labelsize=11, rotation=45)
        ax.tick_params(axis="y", labelsize=11)
        ax.legend(title="Tipe Stasiun")
        ax.grid(axis="y", alpha=0.25)
        ax.set_yticks(range(0, 101, 20))

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown(
            f"Rata-rata harian PM2.5 tertinggi terdapat pada stasiun "
            f"**{highest_station['station']}** sebesar "
            f"**{highest_station['mean_pm25']:.2f} µg/m³**, "
            f"dengan kategori **{highest_station['kategori_pm25']}**."
        )