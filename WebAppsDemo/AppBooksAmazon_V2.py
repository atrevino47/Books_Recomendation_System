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
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from streamlit_elements import elements, mui, html
from streamlit_elements import dashboard


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

df_ratings_books_processed = load_data(
    "C:/Users/adria/Documents/Projects/Books/Resources/DataFrames/SentimentAnalysis/200plusRatingsPerUser_60plusRatingsPerBook/df_ratings_books_processed.parquet"
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

authors_df = pd.read_parquet(
    "C:/Users/adria/Documents/Projects/Books/Resources/DataFrames/GenericDF/authors_df.parquet"
)

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


# // NOTE: Begin STREAMLIT APP

# // NOTE: CSS

custom_palette = {
    "amazon_orange": "#e47911",
    "amazon_blue": "#232f3e",
    "amazon_lightblue1": "#48a3c6",
    "amazon_lightblue2": "#007eb9",
    "amazon_greyblue": "#37475a",
    "amazon_black": "#111111",
    "amazon_white": "#ffffff",
    "amazon_grey": "#cccccc",
    "amazon_mediumgrey": "#fafafa",
    "background": "#ecf0f1",  # Light Gray
    "text": "#2c3e50",  # Dark Gray
}


# // NOTE: Create color pallette
def create_divergent_color_scale(start_color, end_color, num_steps):
    start_rgb = mcolors.hex2color(start_color)
    end_rgb = mcolors.hex2color(end_color)

    r = np.linspace(start_rgb[0], end_rgb[0], num_steps)
    g = np.linspace(start_rgb[1], end_rgb[1], num_steps)
    b = np.linspace(start_rgb[2], end_rgb[2], num_steps)

    colors = np.column_stack((r, g, b))

    return [mcolors.rgb2hex(color) for color in colors]


num_steps = 10
divergent_colors = create_divergent_color_scale(
    custom_palette["amazon_orange"], custom_palette["amazon_blue"], num_steps
)
continuous_colors = [
    (0, custom_palette["amazon_orange"]),
    (1, custom_palette["amazon_blue"]),
]

# // NOTE: Apply the custom color palette using CSS
custom_css = f"""
    <style>
        body {{
            background-color: {custom_palette['amazon_grey']};
            color: {custom_palette['text']};
        }}
        .st-bw {{
            background-color: {custom_palette['amazon_blue']};
        }}
        .st-c3 {{
            color: {custom_palette['amazon_orange']};
        }}
    </style>
"""

# Render the custom CSS
st.markdown(custom_css, unsafe_allow_html=True)


col1, col2 = st.columns((0.8, 0.2))
with col1:
    st.title("Amazon Book Recommendation System")
with col2:
    st.image(amazon_logo, width=130)

# // NOTE: STREAMLIT

with st.expander("Objective"):
    st.markdown(
        """This website aims to explore and explain how this project was developed. It will provide an
        interactive dashboard to visualize Key Insights & KPI's. 
        Finally, create a Book Recommendation system with Machine Learning."""
    )

menu_container = st.container()

with menu_container:
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
        orientation="horizontal",
        styles={
            "container": {
                "padding": "6!important",
                "background-color": custom_palette["amazon_mediumgrey"],
            },
            "icon": {"color": custom_palette["amazon_lightblue1"], "font-size": "25px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": custom_palette["amazon_orange"],
            },
            "nav-link-selected": {"background-color": custom_palette["amazon_blue"]},
        },
    )


def create_kpi_col():
    col1, col2, col3 = st.columns(3)
    col1.metric("Number of recorded squirrels", 3)
    col2.metric("Squirrels per hectare", 3)
    col3.metric("Number of primary colors", 3)


create_kpi_col()

# // NOTE: Graphs


@st.cache_resource
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


@st.cache_resource
def display_graph1():
    # Top 10 Titles by Count and Average Score
    sunburst_top10_titles_fig = px.sunburst(
        top_titles_10,
        path=["title"],
        values="Count",
        color="Average_Score",
        color_continuous_scale=continuous_colors,
        hover_data=["Average_Score"],
        title="Top 10 Titles by Count and Average Score",
    )
    sunburst_top10_titles_figdef = st.plotly_chart(
        sunburst_top10_titles_fig, use_container_width=True
    )
    return sunburst_top10_titles_figdef


@st.cache_resource
def display_graph2():
    # SunBurst Chart of Author birth country
    labels = merged_df_withloc["country"].tolist()
    values = merged_df_withloc["Count"].tolist()

    colors = divergent_colors[: len(labels)]

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


@st.cache_resource
def display_graph3():
    colors2 = divergent_colors[:5]

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


@st.cache_resource
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

    midpoint = df_final_grouped["title_Count"].max() / 2

    fig = px.sunburst(
        df_final_grouped,
        path=["genre", "authors", "title"],
        values="title_Count",
        title="Sunburst Chart of Categories, Authors, and Titles",
        hover_data="average_review_score",
        # // TODO: Add hover template with html
        # hovertemplate='<b>%{label} </b> <br> Sales: %{value}<br> Success rate: %{color:.2f}',
        color="title_Count",
        color_continuous_scale=[
            (0, custom_palette["amazon_blue"]),
            (1, custom_palette["amazon_orange"]),
        ],
        color_continuous_midpoint=midpoint,
    )

    sunburst_genre = st.plotly_chart(fig, use_container_width=True)
    return sunburst_genre


# // NOTE: MENU

if selected_menu == "Dashboard":
    st.write("Dashboard")
    # display_graph1()
    # display_graph2()
    # display_graph3()
    # display_graph4()

    with elements("dashboard"):
        # You can create a draggable and resizable dashboard using
        # any element available in Streamlit Elements.

        # First, build a default layout for every element you want to include in your dashboard

        layout = [
            # Parameters: element_identifier, x_pos, y_pos, width, height, [item properties...]
            dashboard.Item("first_item", 0, 0, 2, 2),
            dashboard.Item("second_item", 2, 0, 2, 2),
            dashboard.Item("third_item", 0, 2, 1, 1, isResizable=False),
        ]

        # Next, create a dashboard layout using the 'with' syntax. It takes the layout
        # as first parameter, plus additional properties you can find in the GitHub links below.

        with dashboard.Grid(layout):
            with mui.Box(key="first_item"):
                with elements("displaygraph1"):
                    display_graph1()

            with mui.Box(key="second_item"):
                with elements("displaygraph2"):
                    display_graph2()

            with mui.Box(key="third_item"):
                with elements("displaygraph4"):
                    display_graph4()

    with elements("nivo_charts"):
        # Streamlit Elements includes 45 dataviz components powered by Nivo.

        from streamlit_elements import nivo

        DATA = [
            {"taste": "fruity", "chardonay": 93, "carmenere": 61, "syrah": 114},
            {"taste": "bitter", "chardonay": 91, "carmenere": 37, "syrah": 72},
            {"taste": "heavy", "chardonay": 56, "carmenere": 95, "syrah": 99},
            {"taste": "strong", "chardonay": 64, "carmenere": 90, "syrah": 30},
            {"taste": "sunny", "chardonay": 119, "carmenere": 94, "syrah": 103},
        ]

        with mui.Box(sx={"height": 500}):
            nivo.Radar(
                data=reviews_wBooks_data,
                keys=["title_count"],
                indexBy="genre",
                valueFormat=">-.2f",
                margin={"top": 70, "right": 80, "bottom": 40, "left": 80},
                borderColor={"from": "color"},
                gridLabelOffset=36,
                dotSize=10,
                dotColor={"theme": "background"},
                dotBorderWidth=2,
                motionConfig="wobbly",
                legends=[
                    {
                        "anchor": "top-left",
                        "direction": "column",
                        "translateX": -50,
                        "translateY": -40,
                        "itemWidth": 80,
                        "itemHeight": 20,
                        "itemTextColor": "#999",
                        "symbolSize": 12,
                        "symbolShape": "circle",
                        "effects": [
                            {"on": "hover", "style": {"itemTextColor": "#000"}}
                        ],
                    }
                ],
                theme={
                    "background": "#FFFFFF",
                    "textColor": "#31333F",
                    "tooltip": {
                        "container": {
                            "background": "#FFFFFF",
                            "color": "#31333F",
                        }
                    },
                },
            )

            # mui.Box(display_graph2(), key="second_item")
            # mui.Box(display_graph4(), key="third_item")

        # If you want to retrieve updated layout values as the user move or resize dashboard items,
        # you can pass a callback to the onLayoutChange event parameter.

        # def handle_layout_change(updated_layout):
        #     # You can save the layout in a file, or do anything you want with it.
        #     # You can pass it back to dashboard.Grid() if you want to restore a saved layout.
        #     print(updated_layout)

        # with dashboard.Grid(layout, onLayoutChange=handle_layout_change):
        #     mui.Paper("First item", key="first_item")
        #     mui.Paper("Second item (cannot drag)", key="second_item")
        #     mui.Paper("Third item (cannot resize)", key="third_item")


elif selected_menu == "KPI-1":
    st.write("KPI-1")
    with elements("trial_element"):
        mui.Typography("hello world haha")

    with elements("multiple_buttons"):
        mui.Button(mui.icon.EmojiPeople, mui.icon.DouleArrow, "Multiple buttons")

        with mui.Button:
            mui.icon.DouleArrow,
            mui.Typography("Multiple buttons children")

    with elements("nested_children"):
        with mui.Paper:
            with mui.Typography:
                html.p("Hello world")
                html.p("Goodbye world")

elif selected_menu == "KPI-2":
    st.write("KPI-2")
    with elements("dashboard"):
        # You can create a draggable and resizable dashboard using
        # any element available in Streamlit Elements.

        # First, build a default layout for every element you want to include in your dashboard

        layout = [
            # Parameters: element_identifier, x_pos, y_pos, width, height, [item properties...]
            dashboard.Item("first_item", 0, 0, 2, 2),
            dashboard.Item("second_item", 2, 0, 2, 2),
            dashboard.Item("third_item", 0, 2, 1, 1, isResizable=False),
        ]

        # Next, create a dashboard layout using the 'with' syntax. It takes the layout
        # as first parameter, plus additional properties you can find in the GitHub links below.

        with dashboard.Grid(layout):
            mui.Paper("First item", key="first_item")
            mui.Paper("Second item (cannot drag)", key="second_item")
            mui.Paper("Third item (cannot resize)", key="third_item")

        # If you want to retrieve updated layout values as the user move or resize dashboard items,
        # you can pass a callback to the onLayoutChange event parameter.

        # def handle_layout_change(updated_layout):
        #     # You can save the layout in a file, or do anything you want with it.
        #     # You can pass it back to dashboard.Grid() if you want to restore a saved layout.
        #     print(updated_layout)

        # with dashboard.Grid(layout, onLayoutChange=handle_layout_change):
        #     mui.Paper("First item", key="first_item")
        #     mui.Paper("Second item (cannot drag)", key="second_item")
        #     mui.Paper("Third item (cannot resize)", key="third_item")


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
# // NOTE: streamlit run C:\Users\adria\Documents\Projects\Books\WebAppsDemo\AppBooksAmazon_V2.py
