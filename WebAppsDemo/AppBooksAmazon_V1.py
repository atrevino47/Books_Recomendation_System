# Import Libraries

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pydeck as pdk
import numpy as np
from streamlit_option_menu import option_menu
from PIL import Image
import plotly.express as px
import folium
from streamlit_folium import st_folium
from streamlit_folium import folium_static
from folium import plugins
import plotly.graph_objects as go
import streamlit.components.v1 as components


st.set_page_config(page_title="Book Recommendation System", page_icon="🐿")


@st.cache_data  # This decorator caches the result of this function, so it's not reloaded on each change.
def load_data(file):
    df = pd.read_parquet(file)
    df = df.fillna("None")
    return df


# // TODO: Turn code into object oriented programming


reviews_wBooks_data = load_data(
    "C:/Users/adria/Documents/Projects/Books/Resources/DataFrames/reviews_wBooks_data.parquet"
)

books_data = load_data(
    "C:/Users/adria/Documents/Data_bases/Books/Amazon_Books_Reviews/books_data/books_data_compresed.parquet"
)

amazon_logo = Image.open(
    r"C:/Users/adria/Documents/Projects/Books/Resources/Images/amazon_logo.jpg"
)

# // NOTE: DATA PREPROCESSING
# // TODO: Turn into functions


grouped_df = (
    reviews_wBooks_data.groupby("title")
    .agg({"title": "count", "review_score": "mean"})
    .rename(columns={"title": "title_Count", "review_score": "average_review_score"})
    .reset_index()
)

# Merge the aggregated data back to the original DataFrame
reviews_wBooks_data = pd.merge(reviews_wBooks_data, grouped_df, on="title")

# Top 10 Titles by Count and Average Score Data Frame
top_titles_10 = (
    reviews_wBooks_data.groupby("title")
    .agg({"review_score": ["count", "mean"]})
    .reset_index()
)
top_titles_10.columns = ["title", "Count", "Average_Score"]
top_titles_10 = top_titles_10.sort_values(by="Count", ascending=False).head(10)

# Prepare DataFrame for map
top_titles_12 = (
    reviews_wBooks_data.groupby("title")
    .agg({"review_score": ["count", "mean"]})
    .reset_index()
)
top_titles_12.columns = ["title", "Count", "Average_Score"]
top_titles_12 = top_titles_12.sort_values(by="Count", ascending=False).head(12)
top_titles_12.reset_index(inplace=True)

merged_df = pd.merge(
    top_titles_12,
    books_data[["title", "authors", "image", "publishedDate"]],
    on="title",
    how="inner",
)
merged_df = merged_df.dropna()
merged_df.reset_index(inplace=True)

# merged_df.drop('level_0', axis=1, inplace=True) # Possibly need to change level_0 to index

authors_dict = {
    "['J. R. R. Tolkien']": {
        "lanlon": "-29.082899, 26.159786",
        "place_of_birth": "Bloemfontein",
        "country": "South Africa",
    },
    "['Jane Austen']": {
        "lanlon": "51.229223, -1.220092",
        "place_of_birth": "Steventon, Basingstoke",
        "country": "UK",
    },
    "['Emily Bronte']": {
        "lanlon": "53.790707, -1.846649",
        "place_of_birth": "Thornton, Bradford",
        "country": "UK",
    },
    "['Lois Lowry']": {
        "lanlon": "21.314281, -157.851258",
        "place_of_birth": "Honolulu, Hawái",
        "country": "USA",
    },
    "['Charles Dickens']": {
        "lanlon": "50.813335, -1.086365",
        "place_of_birth": "Landport, Portsmouth",
        "country": "UK",
    },
    "['J. K. Rowling']": {
        "lanlon": "51.541983, -2.414024",
        "place_of_birth": "Yate, Bristol",
        "country": "UK",
    },
    "['Aldous Huxley']": {
        "lanlon": "51.193015, -0.612654",
        "place_of_birth": "Godalming",
        "country": "UK",
    },
    "['C. S. Lewis']": {
        "lanlon": "54.595004, -5.922083",
        "place_of_birth": "Belfast",
        "country": "UK",
    },
    "['Óscar Wilde']": {
        "lanlon": "53.342877, -6.249795",
        "place_of_birth": "Westland Row, Dublin",
        "country": "Irland",
    },
    "['Jane Austen']": {
        "lanlon": "51.229223, -1.220092",
        "place_of_birth": "Steventon, Basingstoke",
        "country": "UK",
    },
}

# Create locations for top 10 books

authors_df = pd.DataFrame.from_dict(authors_dict, orient="index")
authors_df.reset_index(inplace=True)
authors_df.rename(columns={"index": "authors"}, inplace=True)

# Merge with locations with books df
merged_df_withloc = pd.merge(
    merged_df,
    authors_df[["authors", "lanlon", "place_of_birth", "country"]],
    on="authors",
    how="inner",
)

# Separate latitude and longitude
merged_df_withloc[["latitude", "longitude"]] = (
    merged_df_withloc["lanlon"].str.split(", ", expand=True).astype(float)
)

# // TODO: Search for a better way to filter the data or outright remove it later
reviews_wBooks_data = reviews_wBooks_data[reviews_wBooks_data["authors"] != "NoAuthor"]
reviews_wBooks_data = reviews_wBooks_data[reviews_wBooks_data["genre"] != "NoGenre"]


# def make_clickable(val):
#     return '<a target="_blank" href="{}">Amazon</a>'.format(val, val)


# def show_image(val):
#     return '<a href="{}"><img src="{}" width=60></img></a>'.format(val, val)


st.title("Amazon Book Recommendation System")
st.image(amazon_logo, width=130)

# // NOTE: STREAMLIT

with st.expander("Objective"):
    st.markdown(
        """This website aims to explore and explain how this project was developed. It will provide an
        interactive dashboard to visualize Key Insights & KPI's. 
        Finally, create a Book Recommendation system with Machine Learning."""
    )

with st.sidebar:
    selected_menu = option_menu(
        "Menu",
        ["Dashboard", "KPI-1", "KPI-2", "KPI-3", "Map", "Book Recommendation System"],
        icons=[
            "speedometer2",
            "arrow-bar-right",
            "arrow-bar-right",
            "arrow-bar-right",
            "geo-alt-fill",
            "book",
        ],
        menu_icon="app-indicator",
        default_index=0,
        styles={
            "container": {"padding": "6!important", "background-color": "#fafafa"},
            "icon": {"color": "blue", "font-size": "25px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#eee",
            },
            "nav-link-selected": {"background-color": "#02ab21"},
        },
    )


def create_kpi_col():
    col1, col2, col3 = st.columns(3)
    col1.metric("Number of recorded squirrels", 3)  # , len(data), len(data) - 2373)
    col2.metric(
        "Squirrels per hectare",
        3
        # round(len(data) / 350, 2),
        # round((len(data) - 2373) / 350, 2),
    )
    col3.metric("Number of primary colors", 3)


create_kpi_col()

# // NOTE: Graphs


def displaymap():
    # Create a Folium Map
    my_map = folium.Map(
        location=[
            merged_df_withloc["latitude"].mean(),
            merged_df_withloc["longitude"].mean(),
        ],
        zoom_start=3,
    )

    # Add markers for each book in myMap
    for index, row in merged_df_withloc.iterrows():
        # Create a popup with book information
        popup_text = f"title: {row['title']}<br>Average Score: {row['Average_Score']}<br>Place of Birth: {row['place_of_birth']}"

        custom_icon = folium.CustomIcon(
            icon_image=row["image"],
            icon_size=(60, 60),
            # icon_anchor=(15, 15),
            popup_anchor=(0, -15),
        )

        # Add a marker to the map
        marker = folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=folium.Popup(popup_text, max_width=300),
            icon=custom_icon,
        ).add_to(my_map)

    plugins.MarkerCluster().add_to(my_map)

    authors_map = folium_static(my_map)
    return authors_map


def display_graph1():
    # Top 10 Titles by Count and Average Score
    sunburst_top10_titles_fig = px.sunburst(
        top_titles_10,
        path=["title"],
        values="Count",
        color="Average_Score",
        color_continuous_scale="Viridis_r",
        hover_data=["Average_Score"],
        title="Top 10 Titles by Count and Average Score",
    )
    sunburst_top10_titles_figdef = st.plotly_chart(
        sunburst_top10_titles_fig, use_container_width=True
    )
    return sunburst_top10_titles_figdef


def display_graph2():
    # SunBurst Chart of Author birth country
    labels = merged_df_withloc["country"].tolist()
    values = merged_df_withloc["Count"].tolist()

    colors = px.colors.sequential.Viridis[: len(labels)]

    # Change to sunburst
    top10_authors_countryPie_fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                marker=dict(colors=colors, line=dict(color="#000000", width=2)),
            )
        ]
    )  # Specify the color of the slice borders
    top10_authors_countryPie_fig.update_layout(
        title_text="Top 10 Authors Country of Birth Distribution"
    )
    top10_authors_countryPie_figdef = st.plotly_chart(
        top10_authors_countryPie_fig, use_container_width=True
    )
    return top10_authors_countryPie_figdef


def display_graph3():
    colors2 = px.colors.sequential.Viridis[:5]

    all_ratings_fig = px.histogram(
        x=reviews_wBooks_data["review_score"],
        color=reviews_wBooks_data["review_score"],
        color_discrete_sequence=colors2,
    )

    # Set the chart title
    all_ratings_fig.update_layout(title_text="Distribution of ratings")

    # Show the plot
    all_ratings_figdef = st.plotly_chart(all_ratings_fig, use_container_width=True)
    return all_ratings_figdef


def display_graph4():
    # Create a Sunburst chart

    top_genres = reviews_wBooks_data["genre"].value_counts().nlargest(2).index
    df_top_genres = reviews_wBooks_data[reviews_wBooks_data["genre"].isin(top_genres)]

    # Step 2: For each top genre, filter by top 10 authors
    top_authors_by_genre = (
        df_top_genres.groupby("genre")["authors"]
        .value_counts()
        .groupby("genre", group_keys=False)
        .nlargest(2)
        .index.get_level_values("authors")
    )

    df_final = df_top_genres[df_top_genres["authors"].isin(top_authors_by_genre)]
    df_final_grouped = (
        df_final.groupby(["genre", "authors", "title"])
        .agg(
            {
                "review_score": "mean",
                "title": "count",
            }
        )
        .rename(
            columns={"title": "title_Count", "review_score": "average_review_score"}
        )
        .reset_index()
    )

    fig = px.sunburst(
        df_final_grouped,
        path=["genre", "authors", "title"],
        values="title_Count",
        title="Sunburst Chart of Categories, Authors, and Titles",
        hover_data="average_review_score",
    )

    sunburst_genre = st.plotly_chart(fig, use_container_width=True)
    return sunburst_genre


# // NOTE: MENU

if selected_menu == "Dashboard":
    st.write("Dashboard")
    display_graph1()
    display_graph2()
    display_graph3()
    display_graph4()

elif selected_menu == "KPI-1":
    st.write("KPI-1")

elif selected_menu == "KPI-2":
    st.write("KPI-2")

elif selected_menu == "KPI-3":
    st.write("KPI-3")

elif selected_menu == "Map":
    st.title("Map")
    displaymap()

elif selected_menu == "Book Recommendation System":
    st.write("Book Recommendation System")

# ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)


# // NOTE: to run streamlit app use:
# // NOTE: streamlit run C:\Users\adria\Documents\Projects\Books\WebAppsDemo\AppBooksAmazon.py
