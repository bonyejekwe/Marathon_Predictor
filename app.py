
import plotly.express as px
from shiny.express import input, render, ui
from shinywidgets import render_widget

import numpy as np
import pandas as pd
from bayes_model import initialize, _prior_dist
from visualize import plot_from_data

nucr_filename = "processed_data/nucr_runners.csv"
nucr = pd.read_csv(nucr_filename)
train_data, train_info, test_data, test_info, marks, s2_matrix, max_finish = initialize(test_file=nucr_filename)
test_sample = test_data[np.random.choice(range(test_data.shape[0]), 10)]
informed_prior = _prior_dist(informed=True, max_time=max_finish)

# with ui.sidebar():
#     ui.input_select("var", "Select variable", choices=["total_bill", "tip"])

ui.page_opts(title="Quantifying Uncertainty in Live Marathon Finish Time Predictions", fillable=True)

with ui.nav_panel("NUCR Plots"):
    "View the plots of NUCR runners!"

    ui.input_selectize("var", "Select Runner", choices=list(test_info["Name"]))
    ui.input_checkbox_group("var3", "Select Splits", choices=marks[1:], selected=marks[1:], inline=True)

    @render.plot
    def nucr_hist():
        map = {name: i for i, name in enumerate(test_info["Name"])}
        name = input.var()
        i = map[name]
        row = test_data[i]
        testing = {dist: mark for dist, mark in zip(marks, row)}
        plot_from_data(testing, name=test_info.iloc[i]["Name"], marks=list(input.var3()), prior=informed_prior,
                        lk_data=train_data, s2=s2_matrix, plot_range=40)


with ui.nav_panel("My Plot"):
    "Move the sliders to create your own plot!"

    with ui.layout_column_wrap(gap="2rem"):
        ui.input_slider("s1", "5K", min=15, max=30, value=24, step=0.1),
        ui.input_slider("s2", "10K", min=30, max=60, value=48, step=0.1)
        ui.input_slider("s3", "15K", min=45, max=90, value=72, step=0.1)
        ui.input_slider("s4", "20K", min=60, max=120, value=96, step=0.1)
        ui.input_slider("s5", "HALF", min=63, max=126, value=100, step=0.1)
        ui.input_slider("s6", "25K", min=75, max=150, value=118, step=0.1),
        ui.input_slider("s7", "30K", min=90, max=180, value=142, step=0.1)
        ui.input_slider("s8", "35K", min=105, max=210, value=168, step=0.1)
        ui.input_slider("s9", "40K", min=120, max=240, value=192, step=0.1)


    @render.plot
    def my_hist():
        testing = {"5K": input.s1(), "10K": input.s2(), "15K": input.s3(), "20K": input.s4(), "HALF": input.s5(),
                    "25K": input.s6(), "30K": input.s7(), "35K": input.s8(), "40K": input.s9()}
        plot_from_data(testing, name="My", marks=["5K", "10K", "15K", "20K", "HALF", "25K", "30K", "35K", "40K"], prior=informed_prior,
                        lk_data=train_data, s2=s2_matrix, plot_range=40)


with ui.nav_panel("Table"):
    @render.data_frame
    def table():
        return px.data.tips()
    

