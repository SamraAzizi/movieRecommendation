# Netflix Movie Recommender

## Overview

The Netflix Movie Recommender is an interactive application built using Streamlit, allowing users to find movie recommendations based on their preferences or by searching for a specific title. The application leverages a dataset of Netflix titles and provides both content-based and personalized recommendations.

## Features

- **Movie Search**: Users can input a movie title to receive recommendations for similar movies based on genre.
- **Personalized Recommendations**: Users can describe their movie preferences, and the app will suggest movies that match their interests.
- **Explore Movies**: Users can randomly explore a selection of movies from the dataset.
- **CLI Interface**: A command-line interface (CLI) is also available for users who prefer a terminal-based experience.

## Requirements

To run this application, ensure you have the following Python packages installed:

- `streamlit`
- `pandas`
- `langchain_community`

You can install the required packages using pip:

```bash

pip install streamlit pandas langchain_community
```
# Usage

## Running the Streamlit App

To run the Streamlit application, execute the following command in your terminal:

```bash
streamlit run app.py
```
# Code Explanation

## `app.py`

This file contains the Streamlit application logic. It includes:

- **Data Loading**: Loads the dataset using caching for performance.
- **MovieRecommender Class**: Contains methods for generating recommendations based on user input.
- **Tabs**: Organizes the interface into three sections: Movie Search, Personalized Recommendations, and Explore Movies.

## `cli.py`

This file provides a command-line interface for the recommender system. Users can interact with the recommender through a text-based menu.

## `recommender.py`

This file contains the core logic for the `MovieRecommender` class, which handles:

- **Loading the dataset**
- **Generating content-based recommendations**
- **Generating personalized recommendations based on user preferences**
