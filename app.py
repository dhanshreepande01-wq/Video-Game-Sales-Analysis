
import streamlit as st
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Video Game Sales Analysis",
    page_icon="🎮",
    layout="wide"
)

st.title("🎮 Video Game Sales Analysis")
st.write("Analysis of video game sales across platforms, genres, publishers and regions.")

# File Upload
uploaded_file = st.file_uploader("Upload vgsales.csv", type=["csv"])

if uploaded_file is None:
    st.info("Please upload your vgsales.csv file.")
    st.stop()

df = pd.read_csv(uploaded_file)

# Data Preparation
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df['Year'] = df['Year'].fillna(0)
df['Publisher'] = df['Publisher'].fillna('Unknown')

# Navigation
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Analysis",
    [
        "Data Preparation",
        "EDA",
        "Top Selling Games",
        "Platform Performance",
        "Genre Analysis",
        "Publisher Performance",
        "Regional Sales Analysis"
    ]
)

# Helper Function
def show_plot(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

def make_bar(series, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(10, 5))
    sb.barplot(x=series.index.astype(str), y=series.values, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    show_plot(fig)


# 1. DATA PREPARATION
if page == "Data Preparation":

    st.header("1. Data Preparation")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Rows", len(df))
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Duplicate Rows", df.duplicated().sum())

    st.subheader("Missing Values")
    st.dataframe(df.isnull().sum())

    st.subheader("Data Types")
    st.dataframe(df.dtypes.astype(str))

    st.subheader("Column Names")
    st.write(df.columns.tolist())


# 2. EXPLORATORY DATA ANALYSIS
elif page == "EDA":

    st.header("2. Exploratory Data Analysis")

    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe(), use_container_width=True)

    st.subheader("Sales by Genre and Region")

    genre_stats = df.groupby('Genre')[
        ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
    ].agg(['sum', 'mean'])

    st.dataframe(genre_stats, use_container_width=True)

    st.subheader("Platform Sales")

    platform_stats = df.groupby('Platform')['Global_Sales'].agg(
        ['sum', 'mean']
    )

    st.dataframe(platform_stats, use_container_width=True)

    st.subheader("Genre Global Sales")

    genre_global = df.groupby('Genre')['Global_Sales'].agg(
        ['sum', 'mean']
    )

    st.dataframe(genre_global, use_container_width=True)

    # Global Sales Histogram
    st.subheader("Global Sales Distribution")

    fig, ax = plt.subplots(figsize=(10, 5))
    sb.histplot(data=df, x='Global_Sales', bins=30, ax=ax)
    ax.set_title('Distribution of Global Sales')
    ax.set_xlabel('Global Sales (Millions)')
    ax.set_ylabel('Frequency')
    show_plot(fig)

    # Boxplot
    st.subheader("Global Sales Box Plot")

    fig, ax = plt.subplots(figsize=(10, 3))
    sb.boxplot(data=df, x='Global_Sales', ax=ax)
    ax.set_title('Global Sales Outliers')
    show_plot(fig)

    # Regional Histograms
    regions = [
        ('NA_Sales', 'North America Sales Distribution'),
        ('EU_Sales', 'Europe Sales Distribution'),
        ('JP_Sales', 'Japan Sales Distribution'),
        ('Other_Sales', 'Other Regions Sales Distribution')
    ]

    for col, title in regions:

        fig, ax = plt.subplots(figsize=(9, 4))
        sb.histplot(data=df, x=col, bins=30, ax=ax)
        ax.set_title(title)
        show_plot(fig)

    st.subheader("Games by Genre")

    make_bar(
        df['Genre'].value_counts(),
        'Games by Genre',
        'Genre',
        'Number of Games'
    )


# 3. TOP SELLING GAMES
elif page == "Top Selling Games":

    st.header("3. Top Selling Games")

    st.subheader("Top 10 Games by Global Sales")

    top_games = df.sort_values(
        by='Global_Sales',
        ascending=False
    )[['Name', 'Global_Sales']].head(10)

    st.dataframe(top_games, use_container_width=True)

    total_sales = df['Global_Sales'].sum()

    st.metric(
        "Total Global Sales",
        f"{total_sales:,.2f} Million Units"
    )


# 4. PLATFORM PERFORMANCE
elif page == "Platform Performance":

    st.header("4. Platform Performance")

    st.subheader("Top Platforms by Global Sales")

    p_sales = df.groupby('Platform')['Global_Sales'].sum().nlargest(10)

    st.dataframe(p_sales.rename("Total Global Sales"))

    st.subheader("Platforms with Most Top-Selling Games Over Time")

    # Exclude unknown years
    valid_years = df[df['Year'] != 0]

    top_games = valid_years.loc[
        valid_years.groupby('Year')['Global_Sales'].idxmax()
    ]

    p = top_games['Platform'].value_counts()

    make_bar(
        p,
        'Platforms with the Most Top-Selling Games Over Time',
        'Platform',
        'Number of Years'
    )

    st.subheader("Global Sales Distribution Across Platforms")

    make_bar(
        p_sales,
        'Top 10 Platforms by Global Sales',
        'Platform',
        'Total Global Sales (Millions)'
    )


# 5. GENRE ANALYSIS
elif page == "Genre Analysis":

    st.header("5. Genre Analysis")

    st.subheader("Global Sales by Genre")

    genre_sales = df.groupby('Genre')['Global_Sales'].sum()

    st.dataframe(
        genre_sales.sort_values(ascending=False)
    )

    # Genre vs Platform
    st.subheader("Genre Sales Across Platforms")

    g_platform = pd.pivot_table(
        data=df,
        index='Genre',
        columns='Platform',
        values='Global_Sales',
        aggfunc='sum'
    )

    st.dataframe(g_platform, use_container_width=True)

    fig, ax = plt.subplots(figsize=(14, 6))
    g_platform.plot(kind='bar', ax=ax)

    ax.set_title('Genre Sales Across Platforms')
    ax.set_xlabel('Genre')
    ax.set_ylabel('Global Sales (Millions)')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Platform', bbox_to_anchor=(1.02, 1))

    fig.tight_layout()
    show_plot(fig)

    # Genre vs Region
    st.subheader("Genre Sales Across Regions")

    g_region = df.groupby('Genre')[
        ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
    ].sum()

    st.dataframe(g_region, use_container_width=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    g_region.plot(kind='bar', ax=ax)

    ax.set_title('Genre Sales Across Regions')
    ax.set_xlabel('Genre')
    ax.set_ylabel('Total Sales (Millions)')
    ax.tick_params(axis='x', rotation=45)

    fig.tight_layout()
    show_plot(fig)

    # Genre vs Year
    st.subheader("Genre Sales Over the Years")

    g_year = (
        df[df['Year'] != 0]
        .groupby(['Year', 'Genre'])['Global_Sales']
        .sum()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    sb.lineplot(
        data=g_year,
        x='Year',
        y='Global_Sales',
        hue='Genre',
        ax=ax
    )

    ax.set_title('Global Sales by Genre Over the Years')
    ax.set_xlabel('Year')
    ax.set_ylabel('Global Sales (Millions)')

    fig.tight_layout()
    show_plot(fig)


# 6. PUBLISHER PERFORMANCE
elif page == "Publisher Performance":

    st.header("6. Publisher Performance")

    st.subheader("Top Publisher by Global Sales in Each Genre")

    p_genre = df.groupby(
        ['Genre', 'Publisher']
    )['Global_Sales'].sum().reset_index()

    # Highest publisher for each genre
    result = p_genre.loc[
        p_genre.groupby('Genre')['Global_Sales'].idxmax()
    ].sort_values('Global_Sales', ascending=False)

    st.dataframe(result, use_container_width=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    sb.barplot(
        data=result,
        x='Genre',
        y='Global_Sales',
        hue='Publisher',
        dodge=False,
        ax=ax
    )

    ax.set_title('Top Publisher by Global Sales in Each Genre')
    ax.set_xlabel('Genre')
    ax.set_ylabel('Global Sales (Millions)')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Publisher', bbox_to_anchor=(1.02, 1))

    fig.tight_layout()
    show_plot(fig)

    # Publisher vs Region
    st.subheader("Publisher Performance Across Regions")

    p_region = df.groupby('Publisher')[
        ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
    ].sum()

    top10 = p_region.sum(axis=1).nlargest(10).index

    p_region = p_region.loc[top10]

    st.dataframe(p_region, use_container_width=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    p_region.plot(kind='bar', ax=ax)

    ax.set_title('Publisher Performance Across Regions')
    ax.set_xlabel('Publisher')
    ax.set_ylabel('Total Sales (Millions)')
    ax.tick_params(axis='x', rotation=45)

    fig.tight_layout()
    show_plot(fig)


# 7. REGIONAL SALES ANALYSIS
elif page == "Regional Sales Analysis":

    st.header("7. Regional Sales Analysis")

    region = df[
        ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
    ].sum()

    st.subheader("Total Sales Across Regions")

    st.dataframe(
        region.rename("Total Sales (Millions)")
    )

    make_bar(
        region,
        'Total Sales Across Regions',
        'Region',
        'Total Sales (Millions)'
    )

    # Regional Sales by Genre
    st.subheader("Regional Sales by Genre")

    g = df.groupby('Genre')[
        ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
    ].sum()

    st.dataframe(g, use_container_width=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    g.plot(kind='bar', stacked=True, ax=ax)

    for container in ax.containers:
        ax.bar_label(
            container,
            fmt='%.0f',
            label_type='center',
            fontsize=8
        )

    ax.set_title('Regional Sales by Genre')
    ax.set_xlabel('Genre')
    ax.set_ylabel('Total Sales (Millions)')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Region')

    fig.tight_layout()
    show_plot(fig)


# Footer
st.divider()
st.caption(
    "Video Game Sales Analysis | Python, Pandas, "
    "NumPy, Seaborn, Matplotlib & Streamlit"
)
