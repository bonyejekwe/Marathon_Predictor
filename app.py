
import faicons as fa
import matplotlib.pyplot as plt
from shiny import reactive
from shiny.express import input, render, ui

import pandas as pd
from utils import int_to_str_time
from race_class import RaceSplits, get_from_info, get_race_for_person

nucr_filename = "processed_data/nucr_runners.csv"
nucr = pd.read_csv(nucr_filename)
names = nucr["Name"]
marks = ["5K", "10K", "15K", "20K", "25K", "30K", "35K", "40K"]
race_splits = RaceSplits()

card_style = "color:white; background:#2A2A2A !important;"
ui.page_opts(title="Marathon Finish Time Predictor", fillable=True)

with ui.sidebar(id="sidebar", open="open"):
    ui.input_radio_buttons("radio", "Input Type",  {"0": "Total Time"}, inline=True)  # , "1": "Last Split"
    ui.input_radio_buttons("radio2", "Marathon Race",  {"0": "Boston", "1": "New York", "2": "Chicago"}, inline=True)  
    ui.input_select("select", label="Next split to enter", choices=["5K"], selected="5K")
    ui.input_text("runner_split1", value="", label="Enter time here:", placeholder="MM:SS or HH:MM:SS")
    ui.input_action_button("bttn", "Update", class_="btn-success")
    ui.input_action_button("reset", "Reset", class_="btn-success")

    with ui.card(full_screen=True):
        ui.card_header("Last Action", style=card_style)
        @render.text
        @reactive.event(input.bttn, input.reset)       
        def textSidebar():
            if input.bttn() > race_splits.bttn_count:
                try:
                    split = True if input.radio() == "1" else False
                    race_splits.add_pace(dist=input.select(), time=input.runner_split1(), split=split)
                    race_splits.bttn_count += 1

                    next_dist = marks[len(race_splits.stored_times)] if len(race_splits.stored_times) < len(marks) else None
                    if next_dist is not None:
                        ui.update_select("select", choices=[next_dist], selected=next_dist)
                    else:
                        ui.update_select("select", choices=[], selected=None)

                    ui.update_text("runner_split1", value="")
                    return f"Updated {input.select()}"
                except Exception as exc:
                    return f"Error: {exc}"
            else:
                race_splits.reset_race()
                ui.update_select("select", choices=["5K"], selected="5K")
                ui.update_text("runner_split1", value="")
                return "Reset"

with ui.nav_panel("My Plot"):
    with ui.layout_columns(fill=False):
        with ui.card(full_screen=True):
            "View your finish time estimates in real time! Sequentially input your race splits in the sidebar text box, click update, and view your live prediction at that stage of the race. Input your times in 5km increments [5K, 10K, ..., 40K] using the total time."

    def getInfoInput():
        empty_fig = plt.figure(figsize=(10, 6))
        empty_table = pd.DataFrame(columns=["Split", "Input Time", "Median", "50% CI", "80% CI", "95% CI"])

        if not race_splits.stored_times:
            return empty_fig, empty_table

        race_splits.city = {"0": "bos", "1": "nyc", "2": "chi"}[input.radio2()]
        fig, table = get_from_info(race_splits, show=marks)
        return fig, table

    with ui.layout_columns(col_widths=[6, 6, 12]):
                                     
        with ui.card(full_screen=True):
            ui.card_header("Live Prediction: Table", style=card_style)

            with ui.popover(title="Filter credible intervals", placement="top"):        
                fa.icon_svg("gear")
                ui.input_checkbox_group("intervals1", "Credible Intervals", choices=["50% CI", "80% CI", "95% CI"], 
                                        selected=["50% CI", "95% CI"], inline=True)
            
            @render.data_frame
            @reactive.event(input.bttn, input.reset)
            def tableInput():
                _, info_table = getInfoInput()
                if info_table is None or info_table.empty:
                    return pd.DataFrame(columns=["Split", "Input Time", "Median"] + list(input.intervals1()))
                return info_table[["Split", "Input Time", "Median"] + list(input.intervals1())]

        with ui.card(full_screen=True):
            ui.card_header("Live Prediction: Plot", style=card_style)

            @render.plot
            @reactive.event(input.bttn, input.reset)
            def histInput():
                fig, _ = getInfoInput()
                return fig


with ui.nav_panel("NUCR Plots"):
    with ui.layout_columns(fill=False):
        with ui.card(full_screen=True):
            "The Northeastern University Club Running team has multiple Boston Marathon qualifiers every year. View the plots of some NUCR runners that ran the 2023 race. The vertical dotted black line in the plot shows their actual finish time."

    ui.input_selectize("runner_name0", "", choices=list(names))
    ui.input_checkbox_group("splits_list0", "", choices=marks, selected=marks, inline=True)

    with ui.layout_columns(col_widths=[6, 6, 12]):

        @reactive.calc
        def getInfoNUCR():
            mapping = {name: i for i, name in enumerate(names)}
            name = input.runner_name0()
            i = mapping[name]

            race, actual = get_race_for_person(i, nucr)
            race_splits.city = {"0": "bos", "1": "nyc", "2": "chi"}[input.radio2()]
            fig, table = get_from_info(race, show=input.splits_list0(), actual=actual)
            return fig, table, actual
                                     
        with ui.card(full_screen=True):
            ui.card_header("Live Prediction: Table", style=card_style)

            @render.text        
            def textNUCR():     
                actual_time = getInfoNUCR()[2]
                return f"Actual finish time: {int_to_str_time(actual_time)}" #.txt()
            
            with ui.popover(title="Filter credible intervals", placement="top"):        
                fa.icon_svg("gear")
                ui.input_checkbox_group("intervals0", "Credible Intervals", choices=["50% CI", "80% CI", "95% CI"], 
                                        selected=["50% CI", "95% CI"], inline=True)

            @render.data_frame
            def tableNUCR():
                info_table = getInfoNUCR()[1]
                return info_table[["Split", "Input Time", "Median"] + list(input.intervals0())]

        with ui.card(full_screen=True):
            ui.card_header("Live Prediction: Plot", style=card_style)

            @render.plot
            def histNUCR():
                return getInfoNUCR()[0]
            

with ui.nav_panel("Project Info"):    
    with ui.layout_columns(fill=False):
        with ui.card(full_screen=True):
            ui.card_header("Quantifying Uncertainty in Marathon Finish Time Predictions", style=card_style)
            "During a marathon, a runner’s expected finish time is commonly estimated by extrapolating the average pace covered so far, assuming it is held constant for the rest of the race. Two problems arise when predicting finish times this way: the estimates do not consider the in-race context that can determine if a runner is likely to finish faster or slower than expected, and the prediction is simply a point estimate with no information about uncertainty. To address these issues, we implement a hierarchical Bayesian linear regression model that incorporates information from all splits in a race and allows quantification of uncertainty around the predicted finish times. Multiple models under this Bayesian approach are compared to the traditional baseline model using data from the Boston Marathon over four years (2021-2024), and we find a marked improvement in predictive accuracy. We also briefly explore the same implementation for the New York and Chicago marathons, and develop an app that allows runners to visualize their estimated finish time distribution in real time."
