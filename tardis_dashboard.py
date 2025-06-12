#!/usr/bin/env python3

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib as jl
from sklearn.preprocessing import OrdinalEncoder
import pydeck as pdk

if "page" not in st.session_state:
    st.session_state.page = "home"


def load_model():
    model = jl.load("model.pkl")
    if model is None:
        st.error("Model not found")
        st.stop()
    return model


def load_comments_model():
    model = jl.load("comments_model.pkl")
    if model is None:
        st.error("Model not found")
        st.stop()
    return model


def get_season(month):
    if month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    elif month in [9, 10, 11]:
        return "Autumn"
    else:
        return "Winter"


def load_encoder():
    categorical_cols = [
        "Season",
        "Departure station",
        "Arrival station",
        "Route",
        "Service",
    ]
    encoder = OrdinalEncoder()
    encoder.fit(df[categorical_cols])
    return encoder, categorical_cols


def predict_delay(model, departure_station, arrival_station, date):
    if model is None:
        st.error("Model not found")
        return None
    date_parsed = pd.to_datetime(date, format="%Y-%m", errors="coerce")
    if pd.isna(date_parsed):
        raise ValueError("Invalid date format. Please use 'YYYY-MM'.")
    month = date_parsed.month
    year = date_parsed.year
    season = get_season(month)
    route = f"{departure_station} → {arrival_station}"

    filtered_df = df[
        (df["Departure station"] == departure_station)
        & (df["Arrival station"] == arrival_station)
        & (df["Month"] == month)
    ]

    input_data = {
        "Season": [season],
        "Month": [month],
        "Year": [year],
        "Service": ["National"],
        "Departure station": [departure_station],
        "Arrival station": [arrival_station],
        "Average journey time": [filtered_df["Average journey time"].mean()],
        "Number of scheduled trains": [
            filtered_df["Number of scheduled trains"].mean()
        ],
        "Number of cancelled trains": [
            filtered_df["Number of cancelled trains"].mean()
        ],
        "Number of trains delayed at departure": [
            filtered_df["Number of trains delayed at departure"].mean()
        ],
        "Average delay of late trains at departure": [
            filtered_df["Average delay of late trains at departure"].mean()
        ],
        "Average delay of all trains at departure": [
            filtered_df["Average delay of all trains at departure"].mean()
        ],
        "Number of trains delayed at arrival": [
            filtered_df["Number of trains delayed at arrival"].mean()
        ],
        "Average delay of all trains at arrival": [
            filtered_df["Average delay of all trains at arrival"].mean()
        ],
        "Number of trains delayed > 15min": [
            filtered_df["Number of trains delayed > 15min"].mean()
        ],
        "Average delay of trains > 15min (if competing with flights)": [
            filtered_df[
                "Average delay of trains > 15min (if competing with flights)"
            ].mean()
        ],
        "Number of trains delayed > 30min": [
            filtered_df["Number of trains delayed > 30min"].mean()
        ],
        "Number of trains delayed > 60min": [
            filtered_df["Number of trains delayed > 60min"].mean()
        ],
        "Pct delay due to external causes": [
            filtered_df["Pct delay due to external causes"].mean()
        ],
        "Pct delay due to infrastructure": [
            filtered_df["Pct delay due to infrastructure"].mean()
        ],
        "Pct delay due to traffic management": [
            filtered_df["Pct delay due to traffic management"].mean()
        ],
        "Pct delay due to rolling stock": [
            filtered_df["Pct delay due to rolling stock"].mean()
        ],
        "Pct delay due to station management and equipment reuse": [
            filtered_df[
                "Pct delay due to station management and equipment reuse"
            ].mean()
        ],
        "Pct delay due to passenger handling (crowding, disabled persons, connections)": [
            filtered_df[
                "Pct delay due to passenger handling (crowding, disabled persons, connections)"
            ].mean()
        ],
        "Punctuality": [filtered_df["Punctuality"].mean()],
        "Arrival Delay Gap": [filtered_df["Arrival Delay Gap"].mean()],
        "Departure Delay Gap": [filtered_df["Departure Delay Gap"].mean()],
        "Number of trains non delayed": [
            filtered_df["Number of trains non delayed"].mean()
        ],
        "Percent of trains delayed": [filtered_df["Percent of trains delayed"].mean()],
        "Route": [route],
    }
    input_df = pd.DataFrame(input_data)
    input_df[categorical_cols] = encoder.transform(input_df[categorical_cols])
    prediction = model.predict(input_df)
    return prediction[0] / 10


def extract_date_features(X):
    df_ = X.copy()
    df_["Date"] = pd.to_datetime(df_["Date"], format="%Y-%m")
    df_["Year"] = df_["Date"].dt.year
    df_["Month"] = df_["Date"].dt.month
    return df_[["Year", "Month"]]


def predict_comments(comment_model, departure_station, arrival_station, date):
    if comment_model is None:
        st.error("Comment Model not found")
        return None
    route = f"{departure_station} → {arrival_station}"
    input_df = pd.DataFrame(
        [
            {
                "Date": date,
                "Departure station": departure_station,
                "Arrival station": arrival_station,
                "Route": route,
            }
        ]
    )
    return comment_model.predict(input_df)[0]


def line(df, row_index):
    if row_index >= len(df):
        st.error("Invalid row index")
        return
    row = df.iloc[row_index]
    origin = row["Departure station"]
    destination = row["Arrival station"]
    target_cities_df = pd.DataFrame(target_cities)
    try:
        origin_coords = target_cities_df[target_cities_df["name"] == origin].iloc[0]
        dest_coords = target_cities_df[target_cities_df["name"] == destination].iloc[0]
    except IndexError:
        st.error(f"Could not find coordinates for {origin} or {destination}")
        return
    num_points = 5000
    lats = np.linspace(origin_coords["lat"], dest_coords["lat"], num_points)
    lons = np.linspace(origin_coords["lon"], dest_coords["lon"], num_points)
    route_points = pd.DataFrame({"lat": lats, "lon": lons})
    other_cities = pd.DataFrame(target_cities)
    special_points = pd.DataFrame(
        {
            "lat": [origin_coords["lat"], dest_coords["lat"]],
            "lon": [origin_coords["lon"], dest_coords["lon"]],
            "color": [[0, 0, 255], [0, 0, 255]],
            "name": [origin, destination],
        }
    )
    cities_data = pd.DataFrame(
        {
            "lat": other_cities["lat"],
            "lon": other_cities["lon"],
            "color": [[255, 0, 0, 120]] * len(other_cities),
            "name": other_cities["name"],
        }
    )
    route_data = pd.DataFrame(
        {
            "lat": route_points["lat"],
            "lon": route_points["lon"],
            "color": [[144, 238, 144, 30]] * len(route_points),
            "name": ["TRAJET"] * len(route_points),
        }
    )
    all_points = pd.concat([cities_data, special_points])
    point_layer = pdk.Layer(
        "ScatterplotLayer",
        data=all_points,
        get_position=["lon", "lat"],
        get_color="color",
        get_radius=5000,
        pickable=True,
    )
    path_layer = pdk.Layer(
        "ScatterplotLayer",
        data=route_data,
        get_position=["lon", "lat"],
        get_radius=3000,
        get_color="color",
        pickable=True,
    )
    view_state = pdk.ViewState(
        latitude=(origin_coords["lat"] + dest_coords["lat"]) / 2,
        longitude=(origin_coords["lon"] + dest_coords["lon"]) / 2,
        zoom=5,
        pitch=0,
    )
    deck = pdk.Deck(
        layers=[point_layer, path_layer],
        initial_view_state=view_state,
        tooltip={"text": "{name}"},
    )
    st.pydeck_chart(deck)


def select_itinary():
    st.sidebar.header("Route Selection")
    route_combinations = df[["Departure station"]].drop_duplicates()
    route_names = [
        f"{row['Departure station']}" for _, row in route_combinations.iterrows()
    ]
    selected_departure = st.sidebar.selectbox("From Where ?", route_names)
    route_df = df[(df["Departure station"] == selected_departure)]

    available_arrival_stations = route_df["Arrival station"].unique()
    selected_arrival = st.sidebar.selectbox(
        "To Where ?", available_arrival_stations, index=0
    )
    route_df = route_df[route_df["Arrival station"] == selected_arrival]

    available_dates = ["ALL"] + list(route_df["Date"].unique())
    selected_date = st.sidebar.selectbox("Select date", available_dates, index=0)
    if selected_date == "ALL":
        st.header(
            "Données sur les trajet entre "
            + selected_departure
            + " et "
            + selected_arrival
        )
        selected_index = 1
    else:
        selected_row = route_df[route_df["Date"] == selected_date].iloc[0]
        selected_index = selected_row.name
        line(df, selected_index)
    return selected_index, selected_date, selected_departure, selected_arrival


def select_futur_itinary():
    route_combinations = df[["Departure station"]].drop_duplicates()
    route_names = [
        f"{row['Departure station']}" for _, row in route_combinations.iterrows()
    ]
    selected_departure = st.selectbox("D'où partez vous ?", route_names)
    route_df = df[(df["Departure station"] == selected_departure)]

    available_arrival_stations = route_df["Arrival station"].unique()
    selected_arrival = st.selectbox(
        "Où allez vous ?", available_arrival_stations, index=0
    )
    route_df = route_df[route_df["Arrival station"] == selected_arrival]

    future_start = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
    future_end = pd.to_datetime("2027-12-31")
    selected_date = st.date_input(
        "Quand ?", min_value=future_start, max_value=future_end, value=future_start
    )
    selected_row = route_df.iloc[0]
    selected_index = selected_row.name
    st.markdown("---")
    return selected_departure, selected_arrival, selected_date, selected_index


def format_time(minutes):
    total_seconds = int(round(minutes * 60))
    hours, remainder = divmod(total_seconds, 3600)
    mins, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}h{mins:02d}min{secs:02d}s"
    elif mins > 0:
        return f"{mins:02d}min{secs:02d}s"
    else:
        return f"{secs}s"


def get_comments_for_route(departure_station, arrival_station, date):
    if text_df is None:
        return None, None, None

    comments = text_df[
        (text_df["Departure station"] == departure_station)
        & (text_df["Arrival station"] == arrival_station)
        & (text_df["Date"] == date)
    ]

    if len(comments) == 0:
        return None, None, None

    row = comments.iloc[0]
    return (
        row.get("Cancellation comments"),
        row.get("Departure delay comments"),
        row.get("Arrival delay comments"),
    )


def print_route_details(df, selected_index):
    st.subheader("Month Route Details")
    selected_row = df.iloc[selected_index]
    departure_station = selected_row["Departure station"]
    arrival_station = selected_row["Arrival station"]
    date = selected_row["Date"]

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Date:** {selected_row['Date']}")
        st.write(f"**Season:** {selected_row['Season']}")
        st.write(f"**Service:** {selected_row['Service']}")
        st.write(f"**Departure station:** {departure_station}")
        st.write(f"**Arrival station:** {arrival_station}")
    with col2:
        st.write(
            f"**Number of scheduled trains:** {selected_row['Number of scheduled trains']}"
        )
        st.write(
            f"**Number of cancelled trains:** {selected_row['Number of cancelled trains']}"
        )
        st.write(
            f"**Number of trains at departure:** {selected_row['Number of trains delayed at departure']}"
        )
        st.write(
            f"**Number of trains at arrival:** {selected_row['Number of trains delayed at arrival']}"
        )
        st.write(
            f"**Number of trains delayed > 15min:** {selected_row['Number of trains delayed > 15min']}"
        )
        st.write(
            f"**Number of trains delayed > 30min:** {selected_row['Number of trains delayed > 30min']}"
        )
        st.write(
            f"**Number of trains delayed > 60min:** {selected_row['Number of trains delayed > 60min']}"
        )
    st.subheader("Delay Analysis")
    avg_delay_15min = round(
        selected_row["Average delay of trains > 15min (if competing with flights)"], 2
    )
    st.write(f"**Average delay of trains > 15min:** {format_time(avg_delay_15min)}")
    col1, col2 = st.columns(2)
    with col1:
        avg_delay_dep = round(
            selected_row["Average delay of all trains at departure"], 2
        )
        st.write(f"**Average delay at departure:**  {format_time(avg_delay_dep)}")
        avg_delay_late_dep = round(
            selected_row["Average delay of late trains at departure"], 2
        )
        st.write(
            f"**Average delay of late trains at departure** {format_time(avg_delay_late_dep)}"
        )
    with col2:
        avg_delay_arr = round(selected_row["Average delay of all trains at arrival"], 2)
        st.write(
            f"**Average delay of all trains at arrival:**  {format_time(avg_delay_arr)}"
        )
        avg_delay_late_arr = round(
            selected_row["Average delay of late trains at arrival"], 2
        )
        st.write(
            f"**Average delay of late trains at arrival:** {format_time(avg_delay_late_arr)}"
        )
    st.subheader("Delay Causes")
    causes = {
        "External causes": round(selected_row["Pct delay due to external causes"], 2),
        "Infrastructure": round(selected_row["Pct delay due to infrastructure"], 2),
        "Traffic management": round(
            selected_row["Pct delay due to traffic management"], 2
        ),
        "Rolling stock": round(selected_row["Pct delay due to rolling stock"], 2),
        "Station management": round(
            selected_row["Pct delay due to station management and equipment reuse"], 2
        ),
        "Passenger handling": round(
            selected_row[
                "Pct delay due to passenger handling (crowding, disabled persons, connections)"
            ],
            2,
        ),
    }
    causes_df = pd.DataFrame(
        {"Cause": list(causes.keys()), "Percentage": list(causes.values())}
    )
    causes_df = causes_df.sort_values("Percentage", ascending=False)
    st.bar_chart(causes_df.set_index("Cause"))

    st.subheader("Comments")
    cancellation_comment, departure_comment, arrival_comment = get_comments_for_route(
        departure_station, arrival_station, date
    )

    if cancellation_comment is not None and pd.notna(cancellation_comment):
        st.write(f"**Cancellation comments:** {cancellation_comment}")
    if departure_comment is not None and pd.notna(departure_comment):
        st.write(f"**Departure delay comments:** {departure_comment}")
    if arrival_comment is not None and pd.notna(arrival_comment):
        st.write(f"**Arrival delay comments:** {arrival_comment}")


def print_all_route_infos(df, selected_departure, selected_arrival):
    route_df = df[
        (df["Departure station"] == selected_departure)
        & (df["Arrival station"] == selected_arrival)
    ]
    route_df_comments = comments_df[
        (comments_df["Departure station"] == selected_departure)
        & (comments_df["Arrival station"] == selected_arrival)
    ]
    st.subheader(f"**Nombre de trajets de train:** {len(route_df)}")
    route_df["Total Delays"] = (
        route_df["Number of trains delayed > 15min"]
        + route_df["Number of trains delayed > 30min"]
        + route_df["Number of trains delayed > 60min"]
        + route_df["Number of trains delayed at departure"]
        + route_df["Number of trains delayed at arrival"]
    )
    st.subheader("Distribution des retards (boxplot)")
    boxplot_columns = [
        "Average delay of trains > 15min (if competing with flights)",
        "Average delay of all trains at departure",
        "Average delay of all trains at arrival",
        "Average delay of late trains at departure",
        "Average delay of late trains at arrival",
    ]

    column_display_names = {
        "Average delay of trains > 15min (if competing with flights)": "Trains > 15min",
        "Average delay of all trains at departure": "Tous (départ)",
        "Average delay of all trains at arrival": "Tous (arrivée)",
        "Average delay of late trains at departure": "En retard (départ)",
        "Average delay of late trains at arrival": "En retard (arrivée)",
    }

    boxplot_data = route_df[boxplot_columns].copy()
    boxplot_data.columns = [column_display_names[col] for col in boxplot_columns]

    fig = plt.figure(figsize=(10, 5), facecolor="none")
    ax = fig.add_subplot(111, facecolor="none")
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    boxplot_data.boxplot(
        ax=ax,
        color={
            "boxes": "skyblue",
            "whiskers": "#1f77b4",
            "medians": "red",
            "caps": "#1f77b4",
        },
        boxprops={"alpha": 0.8},
        flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 1},
        patch_artist=True,
    )
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.tick_params(colors="white")
    y_min = max(0, boxplot_data.min().min() * 1)
    y_max = boxplot_data.max().max() * 0.3
    ax.set_ylim(y_min, y_max)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.xlabel("Type de retard")
    plt.ylabel("Minutes de retard")
    plt.title("Distribution des retards par type")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)

    delays_by_date = route_df.groupby("Date")["Total Delays"].sum().reset_index()
    delays_by_date["Date_dt"] = pd.to_datetime(delays_by_date["Date"], format="%Y-%m")
    delays_by_date = delays_by_date.sort_values("Date_dt")

    chart_data = pd.DataFrame(
        {"Total Delays": delays_by_date["Total Delays"].values},
        index=delays_by_date["Date_dt"],
    )

    st.subheader("Nombre de trains retardés en fonction des mois")
    st.line_chart(chart_data, use_container_width=True)

    st.subheader("Moyenne des pourcentages causes")
    average_causes = {
        "External causes": round(
            route_df_comments["Pct delay due to external causes"].mean(), 2
        ),
        "Infrastructure": round(
            route_df_comments["Pct delay due to infrastructure"].mean(), 2
        ),
        "Traffic management": round(
            route_df_comments["Pct delay due to traffic management"].mean(), 2
        ),
        "Rolling stock": round(
            route_df_comments["Pct delay due to rolling stock"].mean(), 2
        ),
        "Station management": round(
            route_df_comments[
                "Pct delay due to station management and equipment reuse"
            ].mean(),
            2,
        ),
        "Passenger handling": round(
            route_df_comments[
                "Pct delay due to passenger handling (crowding, disabled persons, connections)"
            ].mean(),
            2,
        ),
    }
    average_causes_df = pd.DataFrame(
        {
            "Cause": list(average_causes.keys()),
            "Percentage": list(average_causes.values()),
        }
    )
    average_causes_df = average_causes_df.sort_values("Percentage", ascending=False)
    st.bar_chart(average_causes_df.set_index("Cause"))

    route_comments = None
    if text_df is not None:
        route_comments = text_df[
            (text_df["Departure station"] == selected_departure)
            & (text_df["Arrival station"] == selected_arrival)
        ]

    if route_comments is not None and len(route_comments) > 0:
        st.subheader("Commentaires pour cette route")
        for _, comment_row in route_comments.iterrows():
            with st.expander(f"Commentaires pour {comment_row['Date']}"):
                if pd.notna(comment_row.get("Cancellation comments")):
                    st.write(f"**Annulation:** {comment_row['Cancellation comments']}")
                if pd.notna(comment_row.get("Departure delay comments")):
                    st.write(
                        f"**Retard au départ:** {comment_row['Departure delay comments']}"
                    )
                if pd.notna(comment_row.get("Arrival delay comments")):
                    st.write(
                        f"**Retard à l'arrivée:** {comment_row['Arrival delay comments']}"
                    )


def navigate_to(page):
    st.session_state.page = page


def show_analys():
    st.sidebar.write("Bonjour 👋")
    st.sidebar.subheader("Recherchez une destination, un trajet, une date...")
    st.sidebar.write(
        "Ici vous pourrez analyser les anciens trajets, leurs retards et leurs causes."
    )
    st.sidebar.write("Sélectionnez un trajet pour voir les détails.")
    st.sidebar.markdown("---")
    selected_index, selected_date, selected_departure, selected_arrival = (
        select_itinary()
    )
    if selected_date != "ALL":
        print_route_details(df, selected_index)
    else:
        print_all_route_infos(df, selected_departure, selected_arrival)
        line(df, selected_index)


def show_home():
    st.write("Bonjour 👋")
    st.subheader("Recherchez une destination, un trajet, une date...")
    selected_departure, selected_arrival, selected_date, selected_index = (
        select_futur_itinary()
    )
    if selected_date is None:
        return
    prediction = predict_delay(
        model, selected_departure, selected_arrival, selected_date.strftime("%Y-%m")
    )
    st.subheader("Predicted Delay")
    if prediction < 0:
        st.write("# En Avance")
        st.markdown(
            f"Votre train sera en **avance** de: <span style='color:green'>**{format_time(-prediction)}**</span> (train is early)",
            unsafe_allow_html=True,
        )
    else:
        st.write("# En Retard")
        st.markdown(
            f"Votre train sera en **retard** de: <span style='color:red'>**{format_time(prediction)}**</span> (train is late)",
            unsafe_allow_html=True,
        )

    cause = predict_comments(
        comments_model,
        selected_departure,
        selected_arrival,
        selected_date.strftime("%Y-%m"),
    )
    st.write("# Cause")
    st.write(f"La plus probable : **{cause}**")
    st.write("---")
    line(df, selected_index)


def show_dashboard():
    if st.session_state.page == "home":
        show_home()
    elif st.session_state.page == "analysis":
        show_analys()


def load_cities():
    csv_path = "list.csv"
    cities_df = pd.read_csv(csv_path, delimiter=";")
    cities_list = []
    for _, row in cities_df.iterrows():
        cities_list.append(
            {
                "name": row["name"],
                "lat": row["lat"],
                "lon": row["long"] if "long" in cities_df.columns else row["lon"],
            }
        )
    return cities_list


if __name__ == "__main__":
    df = pd.read_csv("cleaned_dataset.csv", delimiter=",")
    comments_df = pd.read_csv("comments_dataset.csv")
    text_df = pd.read_csv("comments.csv")
    target_cities = load_cities()
    model = load_model()
    comments_model = load_comments_model()
    encoder, categorical_cols = load_encoder()
    st.set_page_config(page_title="Train Route Dashboard", layout="wide")
    st.markdown("## Navigation")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("↩️ Home"):
            navigate_to("home")
    with col2:
        if st.button("🔍 Analyse"):
            navigate_to("analysis")
    st.markdown("---")
    show_dashboard()
