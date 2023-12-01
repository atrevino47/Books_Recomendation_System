# Import Libraries
# Hello this is me hahahaha

import streamlit as st
import pandas as pd

# import seaborn as sns
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
import pickle as pk
from surprise import Reader, Dataset
from surprise import SVD, model_selection, accuracy
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from matplotlib.colors import LinearSegmentedColormap


st.set_page_config(
    page_title="Book Recommendation System", page_icon="🐿", layout="wide"
)


@st.cache_data  # This decorator caches the result of this function, so it's not reloaded on each change.
def load_data(file):
    df = pd.read_parquet(file)
    df = df.fillna("None")
    return df


@st.cache_data()
def load_dataframe_1(_data):
    # Replace this with your code to generate or load the DataFrame
    df = _data
    return df


@st.cache_data()
def load_dataframe_2(_data):
    # Replace this with your code to generate or load the DataFrame
    df = _data
    return df


# // TODO: Turn code into object oriented programming

reviews_wBooks_data = load_data("Resources/DataFrames/reviews_wBooks_data.parquet")

books_data = load_data("Resources/DataFrames/books_data.parquet")

df_ratings_books_processed = load_data(
    "Resources/DataFrames/SentimentAnalysis/200plusRatingsPerUser_60plusRatingsPerBook/df_ratings_books_processed.parquet"
)

df_books_processed = load_data(
    "Resources/DataFrames/SentimentAnalysis/200plusRatingsPerUser_60plusRatingsPerBook/df_books_processed.parquet"
)

amazon_logo = Image.open(r"Resources/Images/amazon_logo.jpg")


# Load Book Recommender Model
@st.cache_data
def load_models(file):
    model = pk.load(open(file, "rb"))
    return model


model_SA = load_models("Resources/Models/BookRecommendation/model_svd_SA.pkl")


# // NOTE: DATA PREPROCESSING
# // TODO: Turn into functions

books_data.rename(columns={"Title": "title", "categories": "genre"}, inplace=True)

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

authors_df = pd.read_parquet("Resources/DataFrames/GenericDF/authors_df.parquet")

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

with open("Resources/DataFrames/Styles/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


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


# Header
col1, col2 = st.columns((0.8, 0.2))
with col1:
    # st.title("Amazon Book Recommendation System")
    st.markdown(
        '<p class="dashboard_title">Amazon Book Recommendation System</p>',
        unsafe_allow_html=True,
    )

with col2:
    st.image(amazon_logo, width=130)

# // NOTE: STREAMLIT


# // NOTE: Sidebar menu


# with st.sidebar:
#     st.title("Filters")
#     selected_genres = st.multiselect(
#         "Select Genres", df_ratings_books_processed["genre"].unique()
#     )
#     selected_authors = st.multiselect(
#         "Select Authors", df_ratings_books_processed["authors"].unique()
#     )

# # Filter the dataset based on selected genres and authors
# df_ratings_books_processed_filt = df_ratings_books_processed[
#     (df_ratings_books_processed["genre"].isin(selected_genres))
#     & (df_ratings_books_processed["authors"].isin(selected_authors))
# ]


with st.expander("Objective"):
    st.markdown(
        """This website aims to explore and explain how this project was developed. It will provide an
        interactive dashboard to visualize Key Insights & KPI's. 
        Finally, create a Book Recommendation system with Machine Learning."""
    )

# // NOTE: MENU Config

menu_container = st.container()

with menu_container:
    selected_menu = option_menu(
        "Menu",
        [
            "Dashboard",
            "Sentiment Analysis",
            "Book Comparison",
            "Book Recommendation System",
        ],
        icons=[
            "speedometer2",
            "graph-up-arrow",
            "book",
            "person-circle",
        ],
        menu_icon="app-indicator",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {
                "padding": "4!important",
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
    # folium.TileLayer("stamentoner").add_to(my_map)

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

    authors_map = folium_static(my_map, width=700, height=400)
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

    sunburst_top10_titles_fig.update_layout(coloraxis_colorbar=None)
    # sunburst_top10_titles_fig.update_traces(showscale=False)

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
    top10_authors_countryPie_fig.update_layout(coloraxis_colorbar=None)
    # top10_authors_countryPie_fig.update_traces(showscale=False)

    top10_authors_countryPie_figdef = st.plotly_chart(
        top10_authors_countryPie_fig, use_container_width=True
    )
    return top10_authors_countryPie_figdef


# def display_graph3():
#     colors2 = divergent_colors[:5]

#     all_ratings_fig = px.histogram(
#         x=reviews_wBooks_data["review_score"],
#         color=reviews_wBooks_data["review_score"],
#         color_discrete_sequence=colors2,
#     )

#     # Set the chart title
#     all_ratings_fig.update_layout(title_text="Distribution of ratings")

#     # Show the plot
#     all_ratings_figdef = st.plotly_chart(all_ratings_fig, use_container_width=True)
#     return all_ratings_figdef


@st.cache_resource
def display_graph4():
    # Create a Sunburst chart

    top_genres = reviews_wBooks_data["genre"].value_counts().nlargest(3).index
    df_top_genres = reviews_wBooks_data[reviews_wBooks_data["genre"].isin(top_genres)]

    # Step 2: For each top genre, filter by top 10 authors
    top_authors_by_genre = (
        df_top_genres.groupby("genre")["authors"]
        .value_counts()
        .groupby("genre", group_keys=False)
        .nlargest(3)
        .index.get_level_values("authors")
    )

    df_final = df_top_genres[df_top_genres["authors"].isin(top_authors_by_genre)]

    title_mapping = {
        "The Hobbit; Or, There and Back Again": "The Hobbit",
        "The Hobbitt, or there and back again; illustrated by the author.": "The Hobbit",
        "The Hobbit or There and Back Again": "The Hobbit",
        "The Hobbit There and Back Again": "The Hobbit",
    }
    df_final["grouped_title"] = (
        df_final["title"].map(title_mapping).fillna(df_final["title"])
    )

    # Perform groupby operation on "grouped_title" and calculate the mean of "review_score"
    df_final_grouped = (
        df_final.groupby(["genre", "authors", "grouped_title"])
        .agg({"review_score": "mean", "grouped_title": "count"})
        .rename(
            columns={
                "grouped_title": "title_Count",
                "review_score": "average_review_score",
            }
        )
        .reset_index()
    )

    midpoint = df_final_grouped["title_Count"].max() / 2

    fig = px.sunburst(
        df_final_grouped,
        path=["genre", "authors", "grouped_title"],
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
    fig.update_layout(coloraxis_colorbar=None)
    # fig.update_traces(showscale=False)

    sunburst_genre = st.plotly_chart(fig, use_container_width=True)
    return sunburst_genre


def display_graph5():
    average_scores = (
        df_ratings_books_processed.groupby("title")["roberta_compound"]
        .mean()
        .reset_index()
    )

    # Select the top 5 books
    title_mapping = {
        "The Hobbit; Or, There and Back Again": "The Hobbit",
        "The Hobbitt, or there and back again; illustrated by the author.": "The Hobbit",
        "The Hobbit or There and Back Again": "The Hobbit",
        "The Hobbit There and Back Again": "The Hobbit",
    }

    average_scores["title"] = (
        average_scores["title"].map(title_mapping).fillna(average_scores["title"])
    )
    average_scores = (
        average_scores.groupby("title")["roberta_compound"].mean().reset_index()
    )

    top_5_books = average_scores.nlargest(15, "roberta_compound")
    # print(top_5_books)
    # Plotly horizontal bar chart
    fig = px.bar(
        top_5_books,
        x="roberta_compound",
        y="title",
        orientation="h",
        labels={
            "roberta_compound": "Average Roberta Compound Score",
            "title": "",
        },
        color="roberta_compound",
        color_continuous_scale=continuous_colors,
        title="Top 5 Books by Average Roberta Compound Score",
    )

    fig.update_layout(coloraxis_colorbar=None)
    # fig.update_traces(showscale=False)
    fig.update_layout(xaxis=dict(range=[0.65, 0.95]))
    fig.update_layout(yaxis={"categoryorder": "total ascending"})

    # Display the chart in Streamlit
    best_books_by_roberta_compound = st.plotly_chart(fig, use_container_width=True)
    return best_books_by_roberta_compound


def display_graph6_SA_PIE():
    df_ratings_books_processed["Sentiment"] = df_ratings_books_processed[
        "roberta_compound"
    ].apply(
        lambda x: "positive" if x >= 0.05 else "negative" if x < -0.05 else "neutral"
    )
    sentiment_counts = df_ratings_books_processed["Sentiment"].value_counts()

    # Create a pie chart using Plotly Express
    fig = px.pie(
        values=sentiment_counts,
        names=sentiment_counts.index,
        labels=["positive", "negative", "neutral"],
        title="Sentiment Distribution",
        hole=0.3,  # Add a hole in the center for a donut chart effect
        color_discrete_sequence=divergent_colors,
    )

    # Display the pie chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)


def display_graph6_SA_BAR():
    df_ratings_books_processed["Sentiment"] = df_ratings_books_processed[
        "roberta_compound"
    ].apply(
        lambda x: "positive" if x >= 0.05 else "negative" if x < -0.05 else "neutral"
    )
    sentiment_counts = (
        df_ratings_books_processed["Sentiment"].value_counts().reset_index()
    )

    # Create a horizontal bar chart using Plotly Express
    fig = px.bar(
        sentiment_counts,
        y="index",  # Use the index as the y-axis
        x="Sentiment",
        labels={"Sentiment": "Count"},
        title="Sentiment Distribution (Horizontal Bar Chart)",
        orientation="h",  # Set the orientation to horizontal
        color_continuous_scale=continuous_colors,
    )

    # Display the horizontal bar chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)


def display_graph8_SA():
    label_encoder = LabelEncoder()
    df_ratings_books_processed["genre_encoded"] = label_encoder.fit_transform(
        df_ratings_books_processed["genre"]
    )
    df_ratings_books_processed["publisher_encoded"] = label_encoder.fit_transform(
        df_ratings_books_processed["publisher"]
    )
    cormat = df_ratings_books_processed[
        [
            "publication_year",
            "review_score",
            "roberta_compound",
            "genre_encoded",
            "publisher_encoded",
        ]
    ].corr()
    round(cormat, 2)
    plt.figure(figsize=(12, 8))
    colors = [(0, custom_palette["amazon_orange"]), (1, custom_palette["amazon_blue"])]
    cmap = LinearSegmentedColormap.from_list("custom", colors, N=256)

    sns.heatmap(cormat, cmap=cmap, annot=True, linewidths=2)
    st.pyplot(plt, use_container_width=True)


# // NOTE: user_id select & Title

unique_user_ids_1 = df_ratings_books_processed["user_id"].unique()
unique_user_ids_2 = df_ratings_books_processed["user_id"].unique()

reader_SA = Reader(rating_scale=(-1, 1))

# // NOTE: Book recommendation system functions


def make_clickable(val):
    return '<a target="_blank" href="{}">Google Books</a>'.format(val, val)


def show_image(val):
    return '<a href="{}"><img src="{}" width=60></img></a>'.format(val, val)


def recommendation_svd_1(model_select, reader_select, metric_score, user_id_select):
    user_id = user_id_select
    book_id = set(
        df_ratings_books_processed[df_ratings_books_processed["user_id"] == user_id][
            "book_id"
        ]
    )  # has book ids that have been reviewed

    # Group df_ratings_books_processed by the book id. of all the books that haven't been reviewd by the user
    # Get the mean of review_score and mean of roberta_compound
    # Then merge the df_books_processed to extract 'previewLink', 'image' by 'book_id'

    user_books = df_ratings_books_processed[
        ~df_ratings_books_processed["book_id"].isin(book_id)
    ]  # has all books that haven't been reviewd
    user_books["user_id"] = len(user_books) * [user_id]
    user_books.reset_index(drop=True, inplace=True)

    df_svd_predict = Dataset.load_from_df(
        user_books[["user_id", "book_id", metric_score]], reader_select
    )
    NA, test = model_selection.train_test_split(df_svd_predict, test_size=1.0)
    predictions = model_select.test(test)
    predictions = [prediction.est for prediction in predictions]
    user_books["rating"] = predictions

    user_books_grouped = (
        user_books.groupby("book_id")
        .agg({"review_score": "mean", "roberta_compound": "mean", "rating": "mean"})
        .reset_index()
    )

    user_books_merged = pd.merge(
        user_books_grouped,
        df_books_processed[
            [
                "book_id",
                "title",
                # "previewLink",
                "image",
                "genre",
                "publisher",
                "authors",
            ]
        ].drop_duplicates(),
        how="inner",
        on="book_id",
    )

    top_50_books_for_user_content = user_books_merged.sort_values(
        by=["rating"], ascending=False
    )[:50]
    top_50_books_for_user_content.to_csv(
        "top_50_books_for_user_content.csv", index=False
    )
    book_title_liked_by_user = set(
        df_ratings_books_processed[
            df_ratings_books_processed["user_id"] == user_id
        ].sort_values(by=metric_score, ascending=False)["title"]
    )
    # print("Books highly rated by given user: \n")

    most_liked = most_liked = (
        df_ratings_books_processed[df_ratings_books_processed["user_id"] == user_id]
        .sort_values(by=metric_score, ascending=False)["title"]
        .iloc[0]
    )
    most_recommended = top_50_books_for_user_content["title"].iloc[0]
    least_recommended = top_50_books_for_user_content["title"].iloc[-1]
    top_50_books_for_user_content = top_50_books_for_user_content[
        [
            "book_id",
            "title",
            "review_score",
            # "previewLink",
            "image",
            "rating",
            "roberta_compound",
        ]
    ].head(10)
    # .style.format({"previewLink": make_clickable, "image": show_image})

    # for count, books in tqdm(enumerate(list(book_title_liked_by_user)[:20])):
    #     print(count + 1, ".  ", books)
    return (
        top_50_books_for_user_content,
        most_liked,
        most_recommended,
        least_recommended,
    )

    # // TODO: Fix titles, like filtering The hobbit
    # // NOTE: AEGR7IBHUM3R7 User to use for book recommendations


def recommendation_svd_2(model_select, reader_select, metric_score, user_id_select):
    user_id = user_id_select
    book_id = set(
        df_ratings_books_processed[df_ratings_books_processed["user_id"] == user_id][
            "book_id"
        ]
    )  # has book ids that have been reviewed

    # Group df_ratings_books_processed by the book id. of all the books that haven't been reviewd by the user
    # Get the mean of review_score and mean of roberta_compound
    # Then merge the df_books_processed to extract 'previewLink', 'image' by 'book_id'

    user_books = df_ratings_books_processed[
        ~df_ratings_books_processed["book_id"].isin(book_id)
    ]  # has all books that haven't been reviewd
    user_books["user_id"] = len(user_books) * [user_id]
    user_books.reset_index(drop=True, inplace=True)

    df_svd_predict = Dataset.load_from_df(
        user_books[["user_id", "book_id", metric_score]], reader_select
    )
    NA, test = model_selection.train_test_split(df_svd_predict, test_size=1.0)
    predictions = model_select.test(test)
    predictions = [prediction.est for prediction in predictions]
    user_books["rating"] = predictions

    user_books_grouped = (
        user_books.groupby("book_id")
        .agg({"review_score": "mean", "roberta_compound": "mean", "rating": "mean"})
        .reset_index()
    )

    user_books_merged = pd.merge(
        user_books_grouped,
        df_books_processed[
            [
                "book_id",
                "title",
                "previewLink",
                "image",
                "genre",
                "publisher",
                "authors",
            ]
        ].drop_duplicates(),
        how="inner",
        on="book_id",
    )

    top_50_books_for_user_content = user_books_merged.sort_values(
        by=["rating"], ascending=False
    )[:50]
    top_50_books_for_user_content.to_csv(
        "top_50_books_for_user_content.csv", index=False
    )
    book_title_liked_by_user = set(
        df_ratings_books_processed[
            df_ratings_books_processed["user_id"] == user_id
        ].sort_values(by=metric_score, ascending=False)["title"]
    )
    # print("Books highly rated by given user: \n")

    most_liked = (
        df_ratings_books_processed[df_ratings_books_processed["user_id"] == user_id]
        .sort_values(by=metric_score, ascending=False)["title"]
        .iloc[0]
    )
    most_recommended = top_50_books_for_user_content["title"].iloc[0]
    least_recommended = top_50_books_for_user_content["title"].iloc[-1]
    top_50_books_for_user_content = top_50_books_for_user_content[
        [
            "book_id",
            "title",
            "review_score",
            # "previewLink",
            "image",
            "rating",
            "roberta_compound",
        ]
    ].head(10)
    # .style.format({"previewLink": make_clickable, "image": show_image})

    # for count, books in tqdm(enumerate(list(book_title_liked_by_user)[:20])):
    #     print(count + 1, ".  ", books)
    return (
        top_50_books_for_user_content,
        most_liked,
        most_recommended,
        least_recommended,
    )

    # // TODO: Fix titles, like filtering The hobbit
    # // NOTE: AEGR7IBHUM3R7 User to use for book recommendations


# // NOTE: MENU

if selected_menu == "Dashboard":
    st.write("Dashboard")

    average_rating = reviews_wBooks_data["review_score"].mean()
    average_roberta_compound = df_ratings_books_processed["roberta_compound"].mean()
    top_book = df_ratings_books_processed.loc[
        df_ratings_books_processed["review_score"].idxmax()
    ]["title"]

    # def create_kpi_col():
    #     col1, col2, col3 = st.columns(3)
    #     col1.metric("Average Rating Stars:", "{:.2f}".format(average_rating))
    #     col2.metric(
    #         "Average Sentiment Analysis Score:",
    #         "{:.2f}".format(average_roberta_compound),
    #     )
    #     col3.metric("Top Book:", top_book)

    # create_kpi_col()

    empcol1, col1, col2, col3, empcol2 = st.columns((2.5, 1.5, 1.5, 3.5, 1))
    with col1:
        with st.container():
            kpi1 = "{:.2f}".format(average_rating)
            st.markdown(
                f'<p class="kpi1_text">KPI1 / Review Score<br></p><p class="price_details">{kpi1}</p>',
                unsafe_allow_html=True,
            )
    with col2:
        with st.container():
            kpi2 = "{:.2f}".format(average_roberta_compound)
            st.markdown(
                f'<p class="kpi2_text">KPI2 / Sentiment Analysis<br></p><p class="price_details">{kpi2}</p>',
                unsafe_allow_html=True,
            )
    with col3:
        with st.container():
            kpi3 = top_book
            st.markdown(
                f'<p class="kpi3_text">KPI3 / Top Book<br></p><p class="price_details">{kpi3}</p>',
                unsafe_allow_html=True,
            )

    col1_graph, col2_graph = st.columns(2)
    with col1_graph:
        display_graph4()
    with col2_graph:
        tab1, tab2 = st.tabs(["MAP", "PIE"])

        with tab1:
            displaymap()

        with tab2:
            display_graph2()

    col3_graph, col4_graph = st.columns(2)
    with col3_graph:
        display_graph5()
    with col4_graph:
        display_graph1()


elif selected_menu == "Sentiment Analysis":
    st.write("Sentiment Analysis")
    average_rating = reviews_wBooks_data["review_score"].mean()
    average_roberta_compound = df_ratings_books_processed["roberta_compound"].mean()
    top_book = df_ratings_books_processed.loc[
        df_ratings_books_processed["review_score"].idxmax()
    ]["title"]

    empcol1, col1, col2, col3, empcol2 = st.columns((2.5, 1.5, 1.5, 3.5, 1))
    with col1:
        with st.container():
            kpi1 = "{:.2f}".format(average_rating)
            st.markdown(
                f'<p class="kpi1_text">KPI1 / Review Score<br></p><p class="price_details">{kpi1}</p>',
                unsafe_allow_html=True,
            )
    with col2:
        with st.container():
            kpi2 = "{:.2f}".format(average_roberta_compound)
            st.markdown(
                f'<p class="kpi2_text">KPI2 / Sentiment Analysis<br></p><p class="price_details">{kpi2}</p>',
                unsafe_allow_html=True,
            )
    with col3:
        with st.container():
            kpi3 = top_book
            st.markdown(
                f'<p class="kpi3_text">KPI3 / Top Book<br></p><p class="price_details">{kpi3}</p>',
                unsafe_allow_html=True,
            )

    # col5_graph, col6_graph = st.columns(2)
    # with col5_graph:
    display_graph5()
    # with col6_graph:
    # tab1, tab2 = st.tabs(["BAR", "PIE"])

    # with tab1:
    #     display_graph6_SA_BAR()
    # with tab2:
    #     display_graph6_SA_PIE()

    col7_graph, col8_graph = st.columns(2)
    with col7_graph:
        display_graph6_SA_PIE()
    with col8_graph:
        display_graph8_SA()

elif selected_menu == "Book Comparison":
    st.write("Book Comparison")

    # def create_kpi_col():
    #     col1, col2, col3 = st.columns(3)
    #     col1.metric(
    #         "Average RoBERTa Compound Score",
    #         df_rbp_filt_title_1["roberta_compound"].mean(),
    #     )
    #     col2.metric("Average Stars Rating", df_rbp_filt_title_1["review_score"].mean())
    #     col3.metric("another metric", 3)

    # create_kpi_col()

    # (
    #     empcol1,
    #     col1_book_comparison,
    #     col2_book_comparison,
    #     empcol2,
    #     col3_metrics,
    #     empcol3,
    (
        col1_metrics,
        col1_book_comparison,
        empcol1,
        col2_metrics,
        col2_book_comparison,
        empcol2,
        col3_benchmark,
    ) = st.columns((1, 1, 0.5, 1, 1, 0.5, 1))

    with col1_book_comparison:
        selected_title_1 = st.selectbox(
            "Select a Title to compare :smile:",
            df_ratings_books_processed["title"].unique(),
        )

        # Filter the dataset based on selected title
        df_rbp_filt_title_1 = df_ratings_books_processed[
            df_ratings_books_processed["title"].isin([selected_title_1])
        ]

        image_link_1 = df_books_processed.loc[
            df_books_processed["title"] == selected_title_1, "image"
        ].iloc[0]

        # Display the image in Streamlit
        st.image(
            image_link_1,
            caption=f"{selected_title_1} Book Cover",
            use_column_width=True,
        )

        # // FIXME: make a group by before displaying the df maybe?

        # st.dataframe(df_rbp_filt_title_1, use_container_width=True)

    with col2_book_comparison:
        selected_title_2 = st.selectbox(
            "Select a Title to benchmark :smile:",
            df_ratings_books_processed["title"].unique(),
        )

        df_rbp_filt_title_2 = df_ratings_books_processed[
            df_ratings_books_processed["title"].isin([selected_title_2])
        ]

        image_link2 = df_books_processed.loc[
            df_books_processed["title"] == selected_title_2, "image"
        ].iloc[0]

        # Display the image in Streamlit
        st.image(
            image_link2, caption=f"{selected_title_2} Book Cover", use_column_width=True
        )

        # st.dataframe(df_rbp_filt_title_2, use_container_width=True)

    with col1_metrics:
        # Book 1
        with st.container():
            kpi1 = selected_title_1
            st.markdown(
                f'<p class="kpi1_text">Book Title<br></p><p class="price_details">{kpi1}</p>',
                unsafe_allow_html=True,
            )
        with st.container():
            kpi2_b1 = "{:.2f}".format(df_rbp_filt_title_1["review_score"].mean())
            st.markdown(
                f'<p class="kpi2_text">KPI1 / Review Score<br></p><p class="price_details">{kpi2_b1}</p>',
                unsafe_allow_html=True,
            )
        with st.container():
            kpi3_b1 = "{:.2f}".format(df_rbp_filt_title_1["roberta_compound"].mean())
            st.markdown(
                f'<p class="kpi3_text">KPI2 / Sentiment Analysis<br></p><p class="price_details">{kpi3_b1}</p>',
                unsafe_allow_html=True,
            )

    with col2_metrics:
        # Book 2
        with st.container():
            kpi1 = selected_title_2
            st.markdown(
                f'<p class="kpi1_text">Book Title<br></p><p class="price_details">{kpi1}</p>',
                unsafe_allow_html=True,
            )
        with st.container():
            kpi2_b2 = "{:.2f}".format(df_rbp_filt_title_2["review_score"].mean())
            st.markdown(
                f'<p class="kpi2_text">KPI1 / Review Score<br></p><p class="price_details">{kpi2_b2}</p>',
                unsafe_allow_html=True,
            )
        with st.container():
            kpi3_b2 = "{:.2f}".format(df_rbp_filt_title_2["roberta_compound"].mean())
            st.markdown(
                f'<p class="kpi3_text">KPI2 / Sentiment Analysis<br></p><p class="price_details">{kpi3_b2}</p>',
                unsafe_allow_html=True,
            )

    with col3_benchmark:
        # Benchmark
        with st.container():
            st.markdown(
                f"""<p class="header_title">Book 1 vs Book 2<br></p>""",
                unsafe_allow_html=True,
            )

        with st.container():
            kpi2 = "{:.2f}".format(df_rbp_filt_title_1["review_score"].mean())

            delta = float(kpi2_b1) - float(kpi2_b2)
            delta_color = "green" if delta > 0 else "red"
            delta_text = f'<p class="delta_text">Delta: {delta:.2f}</p>'

            # Display the metric with delta and delta color
            st.markdown(
                f"""<p class="kpi2_text">KPI2 / Review Score<br></p>
                    <p class="price_details">{kpi2}</p>
                    <p class="delta_text" style="color:{delta_color}">{delta:.2f}</p>""",
                unsafe_allow_html=True,
            )

        with st.container():
            kpi3 = "{:.2f}".format(df_rbp_filt_title_1["roberta_compound"].mean())

            # Calculate the delta compared to the previous value
            delta = float(kpi3_b1) - float(kpi3_b2)
            delta_color = "green" if delta > 0 else "red"
            delta_text = f'<p class="delta_text">Delta: {delta:.2f}</p>'

            # Display the metric with delta and delta color
            st.markdown(
                f"""<p class="kpi2_text">KPI2 / Review Score<br></p>
                    <p class="price_details">{kpi3}</p>
                    <p class="delta_text" style="color:{delta_color}">{delta:.2f}</p>""",
                unsafe_allow_html=True,
            )


elif selected_menu == "Book Recommendation System":
    st.write("Book Recommendation System")

    # col1_book_recommendation, col2_book_recommendation = st.columns(2)

    # with col1_book_recommendation:
    selected_user_id_1 = st.selectbox(
        "Select a User to recieve recommendations :smile:",
        unique_user_ids_1,
    )

    if st.button("New Recommendation for User 1"):
        load_dataframe_1.clear()

    (
        top_50_books_for_user_content_1,
        most_liked_1,
        most_recommended_1,
        least_recommended_1,
    ) = recommendation_svd_1(
        model_select=model_SA,
        reader_select=reader_SA,
        metric_score="roberta_compound",
        user_id_select=selected_user_id_1,
    )

    top_50_books_for_user_content_1 = load_dataframe_1(top_50_books_for_user_content_1)

    emp1_1, col1_im_1, col2_im_1, col3_im_1, col4_im_1, col5_im_1, emp2_1 = st.columns(
        (0.5, 1, 1, 1, 1, 1, 0.5)
    )

    with col1_im_1:
        book_rec1_1 = top_50_books_for_user_content_1["image"].iloc[0]
        title_rec1_1 = top_50_books_for_user_content_1["title"].iloc[0]
        st.image(
            book_rec1_1, caption=f"{title_rec1_1} Book Cover", use_column_width=True
        )
    with col2_im_1:
        book_rec2_1 = top_50_books_for_user_content_1["image"].iloc[1]
        title_rec2_1 = top_50_books_for_user_content_1["title"].iloc[1]
        st.image(
            book_rec2_1, caption=f"{title_rec2_1} Book Cover", use_column_width=True
        )
    with col3_im_1:
        book_rec3_1 = top_50_books_for_user_content_1["image"].iloc[2]
        title_rec3_1 = top_50_books_for_user_content_1["title"].iloc[2]
        st.image(
            book_rec3_1, caption=f"{title_rec3_1} Book Cover", use_column_width=True
        )
    with col4_im_1:
        book_rec4_1 = top_50_books_for_user_content_1["image"].iloc[3]
        title_rec4_1 = top_50_books_for_user_content_1["title"].iloc[3]
        st.image(
            book_rec4_1, caption=f"{title_rec4_1} Book Cover", use_column_width=True
        )
    with col5_im_1:
        book_rec5_1 = top_50_books_for_user_content_1["image"].iloc[4]
        title_rec5_1 = top_50_books_for_user_content_1["title"].iloc[4]
        st.image(
            book_rec5_1, caption=f"{title_rec5_1} Book Cover", use_column_width=True
        )

    st.header("Recommended books for User 1")

    col1_user1, col2_user1, col3_user1 = st.columns(3)

    with col1_user1:
        with st.container():
            st.markdown(
                f'<p class="kpi1_text">Most Liked Book by past Ratings<br></p><p class="price_details">{most_liked_1}</p>',
                unsafe_allow_html=True,
            )
    with col2_user1:
        with st.container():
            st.markdown(
                f'<p class="kpi2_text">Most recommended book<br></p><p class="price_details">{most_recommended_1}</p>',
                unsafe_allow_html=True,
            )
    with col3_user1:
        with st.container():
            st.markdown(
                f'<p class="kpi3_text">Least recommended book<br></p><p class="price_details">{least_recommended_1}</p>',
                unsafe_allow_html=True,
            )

    selected_user_id_2 = st.selectbox(
        "Select a User to Compare recommendations  :smile:", unique_user_ids_2
    )

    if st.button("New Recommendation for User 2"):
        load_dataframe_2.clear()

    (
        top_50_books_for_user_content_2,
        most_liked_2,
        most_recommended_2,
        least_recommended_2,
    ) = recommendation_svd_2(
        model_select=model_SA,
        reader_select=reader_SA,
        metric_score="roberta_compound",
        user_id_select=selected_user_id_2,
    )

    top_50_books_for_user_content_2 = load_dataframe_2(top_50_books_for_user_content_2)

    emp1, col1_im, col2_im, col3_im, col4_im, col5_im, emp2 = st.columns(
        (0.5, 1, 1, 1, 1, 1, 0.5)
    )

    with col1_im:
        book_rec1_2 = top_50_books_for_user_content_2["image"].iloc[0]
        title_rec1_2 = top_50_books_for_user_content_2["title"].iloc[0]
        st.image(
            book_rec1_2, caption=f"{title_rec1_2} Book Cover", use_column_width=True
        )
    with col2_im:
        book_rec2_2 = top_50_books_for_user_content_2["image"].iloc[1]
        title_rec2_2 = top_50_books_for_user_content_2["title"].iloc[1]
        st.image(
            book_rec2_2, caption=f"{title_rec2_2} Book Cover", use_column_width=True
        )
    with col3_im:
        book_rec3_2 = top_50_books_for_user_content_2["image"].iloc[2]
        title_rec3_2 = top_50_books_for_user_content_2["title"].iloc[2]
        st.image(
            book_rec3_2, caption=f"{title_rec3_2} Book Cover", use_column_width=True
        )
    with col4_im:
        book_rec4_2 = top_50_books_for_user_content_2["image"].iloc[3]
        title_rec4_2 = top_50_books_for_user_content_2["title"].iloc[3]
        st.image(
            book_rec4_2, caption=f"{title_rec4_2} Book Cover", use_column_width=True
        )
    with col5_im:
        book_rec5_2 = top_50_books_for_user_content_2["image"].iloc[4]
        title_rec5_2 = top_50_books_for_user_content_2["title"].iloc[4]
        st.image(
            book_rec5_2, caption=f"{title_rec5_2} Book Cover", use_column_width=True
        )

    st.header("Recommended Books for User 2")

    col1_user2, col2_user2, col3_user2 = st.columns(3)

    with col1_user2:
        with st.container():
            st.markdown(
                f'<p class="kpi1_text">Most Liked Book by past Ratings<br></p><p class="price_details">{most_liked_2}</p>',
                unsafe_allow_html=True,
            )
    with col2_user2:
        with st.container():
            st.markdown(
                f'<p class="kpi2_text">Most recommended book<br></p><p class="price_details">{most_recommended_2}</p>',
                unsafe_allow_html=True,
            )
    with col3_user2:
        with st.container():
            st.markdown(
                f'<p class="kpi3_text">Least recommended book<br></p><p class="price_details">{least_recommended_2}</p>',
                unsafe_allow_html=True,
            )


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
# // NOTE: streamlit run C:\Users\adria\Documents\MyPortfolio\Book_Recommendation_System\AppBooksAmazon_V3.py --server.headless true
