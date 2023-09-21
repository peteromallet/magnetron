from magnetron import display_rows_with_editing, fetch_and_filter_data
import streamlit as st
import pandas as pd
import json
import yaml


st.set_page_config(layout="wide") 

st.header("Steerable Motion Dataset")

st.markdown("***")

with st.sidebar:

    st.subheader("Download dataset")

    st.write("This is the output of the Steerable Motion training data collection process. You can read more about Steerable Motion [here](https://steerablemotion.com) and find the repo for the data collection tool [here](https://github.com/banodoco/magnetron).")

    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    df = fetch_and_filter_data(statuses=['approved'])[['video_id', 'uuid', 'image_0_location', 'image_1_location', 'caption', 'image_0_frame_number', 'image_1_frame_number']]
    csv = convert_df(df)

    st.download_button("Download Data", csv, "file.csv", "text/csv", key='download-csv')

display_rows_with_editing(status='approved', allow_action=False)