
# plots.py: functionality to generate miscellaneous plots

import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from utils import get_data, binning, int_to_str_time, mae_table, rmse_table, group_data, other_stats, all_tests, get_test_preds, add_intervals_to_test, marathon_map
random.seed(2025)

def get_plot_race_dists(races=["bos", "nyc", "chi"]):
    """Plot distribution of finish times for a list of races in specified years"""
    ticks = (60, 120, 180, 240, 300, 360, 420, 480, 540, 600)
    train_list = [(r, pd.read_csv(f"processed_data/full_data_{r}.csv")) for r in races]
    cmap = {"bos": "darkgreen", "nyc": "blue", "chi": "crimson"}
    nmap = {"bos": "Boston", "nyc": "New York", "chi": "Chicago"}

    fig, ax = plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(12)
    for lbl, train_data in train_list:
        minutes_dist = train_data["Finish Net"] // 60
        bins = binning(minutes_dist)
        minutes_dist.hist(bins=bins, alpha=0.2, density=True, color=cmap[lbl], ax=ax)
        d2 = np.bincount(minutes_dist)
        ax.plot(range(len(d2)), d2 / sum(d2), color=cmap[lbl], linewidth=1, label=nmap[lbl])
    
    # ax.set_facecolor(("orange", 0.05))
    labels = [int_to_str_time(60 * t, no_secs=True) for t in ticks]
    
    plt.xticks(ticks, labels=labels, fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim(ticks[1] - 15, ticks[-2] - 45)
    plt.xlabel(f"Time (HH:MM)", fontsize=20)
    plt.ylabel("Frequency", fontsize=20)
    plt.title(f"Distribution of Marathon Finish Times", fontsize=22)
    plt.legend(fontsize=18)
    plt.savefig(f"analysis/plots/plot_dists.jpg", dpi=300)
    plt.close()

def get_plot_race_subgroup(race="bos"):
    """Plot distribution of finish times for a list of races in specified years"""
    fig, ax = plt.subplots(ncols=2, sharey=True)
    fig.set_figheight(8)
    fig.set_figwidth(20)

    ticks = (60, 120, 180, 240, 300, 360, 420, 480, 540)
    df = pd.read_csv(f"processed_data/full_data_{race}.csv")

    train_list = [(r, df[df["M/F"] == r]) for r in ["M", "F"]]
    cmap = {"M": "blue", "F": "crimson"}
    nmap = {"M": "Male", "F": "Female"}

    for lbl, train_data in train_list:
        minutes_dist = train_data["Finish Net"] // 60
        bins = binning(minutes_dist)
        minutes_dist.hist(bins=bins, alpha=0.2, density=True, color=cmap[lbl], ax=ax[0])
        d2 = np.bincount(minutes_dist)
        ax[0].plot(range(len(d2)), d2 / sum(d2), color=cmap[lbl], linewidth=1, label=nmap[lbl])
        
    train_list = [(r, df[(df["Age"] >= lower) & (df["Age"] <= higher)]) for r, lower, higher in [["1", 0, 30], ["2", 31, 40], ["3", 41, 50], ["4", 51, 100]]]
    nmap = {"1": "Under 30", "2": "31 - 40", "3": "41 - 50", "4": "Above 50"}
    sns.set_palette("crest", n_colors=4, desat=0.8)
    colors1 = sns.color_palette()
    cmap = {str(i+1): colors1[i] for i in range(4)}

    for lbl, train_data in train_list:
        minutes_dist = train_data["Finish Net"] // 60
        bins = binning(minutes_dist)
        minutes_dist.hist(bins=bins, alpha=0.2, density=True, color=cmap[lbl], ax=ax[1])
        d2 = np.bincount(minutes_dist)
        ax[1].plot(range(len(d2)), d2 / sum(d2), color=cmap[lbl], linewidth=1, label=nmap[lbl])
    
    # ax.set_facecolor(("orange", 0.05))
    labels = [int_to_str_time(60 * t, no_secs=True) for t in ticks]
    yticks=ax[0].get_yticks()[::2]
    ax[0].set_ylabel("Frequency", fontsize=20)
    for i, group in [[0, "Age"], [1, "Gender"]]:
        ax[i].set_xticks(ticks, labels=labels, fontsize=18)
        ax[i].set_xlim(ticks[1] - 15, ticks[-2] - 45)
        ax[i].set_xlabel(f"Time (HH:MM)", fontsize=20)
        ax[i].set_yticks(yticks, labels=yticks, fontsize=18)
        ax[i].set_title(f"Distribution by {group} - {marathon_map.get(race, race)}", fontsize=22)
        ax[i].legend(fontsize=18)

    plt.savefig(f"analysis/plots/plot_dist_{race}_both.jpg", dpi=300)
    plt.close()

def get_extrap_scatter(racename="bos", sample_size=1000, yrs=[2021, 2022, 2023], mks=["10K", "20K", "30K"]):
    """Get scatter plot comparing extrapolated finish times to actual finish times for a given race"""
    train_data, _ = get_data(racename=racename, size_train=sample_size, train_lis=yrs)
    table = train_data[train_data["dist"].isin(mks)].copy()
    table[["total_pace", "finish"]] = table[["total_pace", "finish"]].apply(lambda x: (42195 / 60) / x)
    
    if racename == "bos":
        ticks = (90, 150, 210, 270, 330, 390, 450)
    else:
        ticks = (90, 150, 210, 270, 330, 390, 450, 510)

    sns.set_palette("mako", n_colors=len(mks)) #crest #mako
    sns.jointplot(data=table, x="total_pace", y="finish", hue="dist", alpha=0.8)# hueorder=["red", "blue", "green"])
    plt.plot([ticks[0] - 30, ticks[-1] + 30], [ticks[0] - 30, ticks[-1] + 30], color="red", label='', alpha=0.6)
    plt.xlabel("Finish Estimate Extrapolated from Total Pace (HH:MM)", fontsize=12)
    plt.ylabel("True Finish Time (HH:MM)", fontsize=12)

    labels = [int_to_str_time(60 * t, no_secs=True) for t in ticks]
    plt.xticks(ticks, labels=labels, rotation=0)
    plt.yticks(ticks, labels=labels, rotation=0)

    for i, dist in enumerate(mks):
        small_data = table[table["dist"] == dist]
        m, b = np.polyfit(small_data["total_pace"], small_data["finish"], 1)
        x = np.array([ticks[0] - 30, ticks[-1] + 30])
        plt.plot(x, m*x + b, alpha=0.8, label=f"{dist} fit", linestyle="--", color=sns.color_palette()[i])

    plt.xlim(ticks[0] - 30, ticks[-1] + 30)
    plt.ylim(ticks[0] - 30, ticks[-1] + 30)
    # plt.title("Comparing Extrapolated Estimates to True Finish At Different Stages of Race")
    plt.grid()
    plt.legend() # ncols=2)
    filename = f"analysis/plots/{racename}_data_scatter.jpg"
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    print(f"File saved: {filename}")

def plot_error(test_data: pd.DataFrame, models: list, baseline: str, save_name: str = "bos", bar=True, rnd=3, other=True, pr=False):
    """Create table and plot (line or bar) to compare the MAE for multiple mdoels. Labels
      specifies the models to be shown."""
    fig, ax = plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(11)

    labels = models + [baseline]
    sns.reset_defaults()
    colors = [f"C{i}" for i in range(len(labels))]
    # sns.set_palette("viridis", n_colors=len(colors), desat=0.8)
    table_group = mae_table(test_data, labels).sort_index(axis=1)
    if bar:
        table_group.plot(width=0.6, alpha=0.8, edgecolor="black", linewidth=0.3, ax=ax, legend=False, kind="bar") #, color=colors)
        suff = "_bar"
    else:
        table_group.plot(label=table_group.columns, style='.-', linewidth=2, grid=True, alpha=0.8, color=colors, ax=ax)
        suff = "_line"
        
    # fig.patch.set_facecolor(('yellow', 0.05)) # This changes the grey to white
    #ax.set_facecolor(("orange", 0.05))
    plt.xticks(fontsize=12, rotation=0)
    plt.yticks(fontsize=12)
    plt.xlabel("Distance Into Race (km)", fontsize=15)
    plt.ylabel("Prediction Error (MAE), in minutes", fontsize=15)
    plt.title(f"Average Error For Each Model - {marathon_map.get(save_name, save_name)}", fontsize=18)
    plt.legend(fontsize=15)
    plt.grid(True)
    if save_name != "":
        filename = f"analysis/plots/{save_name}_MAE_error{suff}.jpg"
        plt.savefig(filename, bbox_inches="tight", dpi=300)
        if pr == False:
            print(f"File saved: {filename}")
    plt.close()

    for lbl in models:
        table_group[f"pcnt_{lbl}"] = 1 - (table_group[lbl] / table_group[baseline])

    if other:
        other_tbl = other_stats(test_data[[baseline] + models], test_data["finish"])
        table_group = pd.concat([table_group, other_tbl])

    filename = f"analysis/tables/{save_name}_MAE_error.csv"
    table_group.round(rnd).to_csv(filename)
    if pr == False:
        print(f"File saved: {filename}")
    return table_group

def plot_error2(test_data: pd.DataFrame, models: list, baseline: str, save_name: str = "bos", bar=True, rnd=3, other=True, pr=False):
    """Create table and plot (line or bar) to compare the RMSE for multiple mdoels. Labels
      specifies the models to be shown."""
    fig, ax = plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(11)

    labels = models + [baseline]
    sns.reset_defaults()
    colors = [f"C{i}" for i in range(len(labels))]
    # sns.set_palette("viridis", n_colors=len(colors), desat=0.8)
    table_group = rmse_table(test_data, labels).sort_index(axis=1)
    if bar:
        table_group.plot(width=0.6, alpha=0.8, edgecolor="black", linewidth=0.3, ax=ax, legend=False, kind="bar") #, color=colors)
        suff = "_bar"
    else:
        table_group.plot(label=table_group.columns, style='.-', linewidth=2, grid=True, alpha=0.8, color=colors, ax=ax)
        suff = "_line"
        
    # fig.patch.set_facecolor(('yellow', 0.05)) # This changes the grey to white
    #ax.set_facecolor(("orange", 0.05))
    plt.xticks(fontsize=12, rotation=0)
    plt.yticks(fontsize=12)
    plt.xlabel("Distance Into Race (km)", fontsize=15)
    plt.ylabel("Prediction Error (RMSE), in minutes", fontsize=15)
    plt.title(f"Average Error For Each Model - {marathon_map.get(save_name, save_name)}", fontsize=18)
    plt.legend(fontsize=15)
    plt.grid(True)
    if save_name != "":
        filename = f"analysis/plots/{save_name}_RMSE_error{suff}.jpg"
        plt.savefig(filename, bbox_inches="tight", dpi=300)
        if pr == False:
            print(f"File saved: {filename}")
    plt.close()

    for lbl in models:
        table_group[f"pcnt_{lbl}"] = 1 - (table_group[lbl] / table_group[baseline])

    if other:
        other_tbl = other_stats(test_data[[baseline] + models], test_data["finish"])
        table_group = pd.concat([table_group, other_tbl])

    filename = f"analysis/tables/{save_name}_RMSE_error.csv"
    table_group.round(rnd).to_csv(filename)
    if pr == False:
        print(f"File saved: {filename}")
    return table_group


def plot_finish_groups(test_data, model: str, baseline: str, num=4, overall=True, save_name: str = "bos", 
                       palette="inferno", grouping="finish", pr=False): # grouping="age"
    """Create error plot grouped by finish time"""
    fig, ax = plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(11)
    sns.set_palette(palette, n_colors=num, desat=0.8)
    colors1 = sns.color_palette()
    # colors1 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'][:num]
    colors2 = [to_rgba(c, alpha=0.3) for c in colors1]
    mixed_colors = [val for pair in zip(colors2, colors1) for val in pair]

    table_group = group_data(test_data, group_feat=grouping, num_groups=num, pref="Q", group_name="group", lbls=[baseline, model])
    table_group.plot(style='.-', width=0.8, color=mixed_colors, edgecolor="black", linewidth=0.3, ax=ax, legend=False, kind="bar")

    if overall:
        error = mae_table(test_data, [baseline, model])
        plt.plot(range(8), error[baseline], color="black", alpha=0.4, linestyle=':',  marker=".", label=f"TOTAL_{baseline}")
        plt.plot(range(8), error[model], color="black", alpha=0.4, linestyle='-',  marker=".", label=f"TOTAL_{model}")
    # fig.patch.set_facecolor(('yellow', 0.05)) # This changes the grey to white
    # ax.set_facecolor(("orange", 0.05))
    plt.grid(alpha=0.8)
    plt.legend(fontsize=15)
    plt.xticks(range(8), rotation=0, fontsize=16)
    plt.yticks(range(0, int(table_group.max(axis=None) + 5), 5), fontsize=16)
    plt.xlabel("Distance Into Race (km)", fontsize=18)
    plt.ylabel("Prediction Error (MAE), in minutes", fontsize=18)
    plt.title(f"Average Error By Finish Groups - {marathon_map.get(save_name, save_name)}", fontsize=22)
    filename = f"analysis/plots/{save_name}_MAE_error_groups.jpg"
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    if pr == False:
        print(f"File saved: {filename}")
    sns.reset_defaults()
    plt.close()
    return table_group


def plot_finish_age_gender(test_data, model: str, baseline: str, num=4, overall=True, 
                           save_name: str = "bos", palette="crest", grouping="age", bins=None, pr=False):
    """Create MAE plot grouped by age group, for each gender"""
    fig, ax = plt.subplots(ncols=2, sharey=True)
    fig.set_figheight(10)
    fig.set_figwidth(20)
    mks = ["5K", "10K", "15K", "20K", "25K", "30K", "35K", "40K"]

    sns.set_palette(palette, n_colors=num, desat=0.8)
    colors1 = sns.color_palette()
    colors2 = [to_rgba(c, alpha=0.3) for c in colors1]
    mixed_colors = [val for pair in zip(colors2, colors1) for val in pair]

    for i, g in enumerate(["F", "M"]):
        test_data2 = test_data[test_data["male"] == i]
        table_group = group_data(test_data2, group_feat=grouping, num_groups=num, pref="G", group_name="group", lbls=[baseline, model], bins=bins)
        table_group.plot(style='.-', width=0.8, #alpha=1, 
            color=mixed_colors,edgecolor="black", linewidth=0.3,
            ax=ax[i], legend=False, kind="bar")

        if overall:
            error = mae_table(test_data2, [baseline, model])
            ax[i].plot(range(8), error[baseline], color="black", alpha=0.4, linestyle=':',  marker=".", label=f"TOTAL_{baseline}")
            ax[i].plot(range(8), error[model], color="black", alpha=0.4, linestyle='-',  marker=".", label=f"TOTAL_{model}")

        #ax[i].set(xlabel="Distance Into Race (km)", ylabel="Prediction Error (MAE), in minutes", title=f"Average Error By Finish Groups - {g}") # , ylim=(0.2, 1)
        yrange = range(0, int(table_group.max(axis=None) + 5), 5)
        ax[i].set_xticks(ticks=range(8), labels=mks, fontsize=16, rotation=0)
        ax[i].set_yticks(ticks=yrange, labels=yrange, fontsize=16)
        ax[i].set_xlabel("Distance Into Race (km)", fontsize=18)
        ax[i].set_ylabel("Prediction Error (MAE), in minutes", fontsize=18)
        ax[i].set_title(f"Average Error By Age Groups - {g}", fontsize=22)
        ax[i].legend(fontsize=16)
        ax[i].grid(alpha=0.8)
        ax[i].set_xticklabels(mks, rotation=0)

    filename = f"analysis/plots/{save_name}_error_gender_age.jpg"
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    if pr == False:
        print(f"File saved: {filename}")
    sns.reset_defaults()
    plt.close()
    return

def plot_interval_checks(itbl: pd.DataFrame, pred_names: list, intervals: list = [50, 80, 95], 
                        linestyles = ["-.", "--.", ":."], save_name: str = "bos", rnd=3):
    """Plot the interval check and sizes for each model specified in pred_names"""
    fig, ax = plt.subplots(ncols=2)
    fig.set_figheight(12)
    fig.set_figwidth(20)

    n = len(pred_names)
    colors = [f"C{i}" for i in range(len(pred_names))] * len(intervals)
    mks = ["5K", "10K", "15K", "20K", "25K", "30K", "35K", "40K"]
    tables0, tables1 = [], []
    for conf in intervals:
        
        sublabels = [f"{p}-size{conf}" for p in pred_names]
        result_tbl = itbl.groupby(["dist"])[sublabels].apply(lambda x: x.sum() / len(x)).loc[mks]
        tables0.append(result_tbl)

        sublabels = [f"{p}-in{conf}" for p in pred_names]
        result_tbl = itbl.groupby(["dist"])[sublabels].apply(lambda x: x.sum() / len(x)).loc[mks]
        tables1.append(result_tbl)

    big_table0, big_table1 = pd.concat(tables0, axis=1), pd.concat(tables1, axis=1)

    styles = []
    for s in linestyles:
        styles += [s] * n
    ax0 = big_table0.plot(label=big_table0.columns,  style=styles , linewidth=2, grid=True, alpha=0.75, color=colors, ax=ax[0])
    ax1 = big_table1.plot(label=big_table1.columns,  style=styles , linewidth=2, grid=True, alpha=0.75, color=colors, ax=ax[1])

    ax0.set(xlabel="Distance Into Race", ylabel="Credible Interval Sizes",
            title="Credible Interval Sizes") #, facecolor=("orange", 0.05)
    ax1.set(xlabel="Distance Into Race", ylabel="Proportion of Actual Finish Times Within Credible Interval", 
            title="Proportion Within Interval Over Different Distances") # , ylim=(0.2, 1)

    leg01 = ax0.legend([Line2D([0,1],[0,1],linestyle=s, color='black') for s in ["-.", "--", ":"]], ["50", "80", "95"], loc="upper center")
    ax[0].add_artist(leg01)
    leg02 = ax0.legend([Line2D([0,1],[0,1], color=c) for c in colors[:n]], pred_names, loc="upper right")
    ax[0].add_artist(leg02)

    leg11 = ax1.legend([Line2D([0,1],[0,1],linestyle=s, color='black') for s in ["-.", "--", ":"]], ["50", "80", "95"], loc="lower center")
    ax[1].add_artist(leg11)
    leg12 = ax1.legend([Line2D([0,1],[0,1], color=c) for c in colors[:n]], pred_names, loc="lower right")
    ax[1].add_artist(leg12)

    filename = f"analysis/plots/{save_name}_intervals"
    plt.savefig(filename + ".jpg", bbox_inches="tight", dpi=300)
    plt.close()

    filename = f"analysis/tables/{save_name}_int"
    big_table0.round(rnd).to_csv(filename + "_sizes.csv")
    big_table1.round(rnd).to_csv(filename + "_check.csv")
    print(f"File saved for {save_name}: intervals.jpg, sizes.csv, check.csv")
    return big_table0, big_table1

def plots_for_race(test_pred_data, race, models, baseline, comp_model):
    """Function for all plots except interval data"""
    tbl = plot_error(test_pred_data, models, baseline, save_name=race, bar=True, other=True, pr=True)
    a = plot_finish_groups(test_pred_data, model=comp_model, baseline=baseline, save_name=race, pr=True)
    plot_finish_age_gender(test_pred_data, model=comp_model, baseline=baseline, save_name=race, grouping="age", bins=[0, 30, 40, 50], pr=True)
    print(f"File saved for {race}: error_bar.jpg, error.csv, error_groups.jpg, error_gender_age.jpg")    
    return tbl

def tests_and_plots(
        test_data_full, races_list, models, baseline, full_preds=True, comp_model="", tests_list=["KS", "CVM", "AD", "Wilcoxon"]
    ):
    """Function to run all tests and plots for each race in `races_list`.
        If a specific race is specified, it will subset only that race from the data.
        If `full_preds` is True, will run interval checks and plots using full array.
        Otherwise, will run error plots and tests using prediction means.
        Return a dictionary with keys as each race and values as the generated tables."""
    tables = {}
    for race in races_list:
        if race != "all":
            test_df = test_data_full[test_data_full["Race"] == race]
        else:
            test_df = test_data_full
        
        if full_preds: # full array, for interval checks and plots
            preds_full = get_test_preds(test_df, "all", baseline=baseline, full=True)
            intervals_tbl = add_intervals_to_test(test_df, preds_full, models)
            i_check, i_sizes = plot_interval_checks(intervals_tbl, models, save_name=race)
            tables[race] = (i_check, i_sizes)
        else: # prediction means, for error plots and tests
            tests_tbl = all_tests(data=test_df, models_list=models, savename=race, tests_list=tests_list)
            error_tbl = plots_for_race(test_pred_data=test_df, race=race, models=models, baseline=baseline, comp_model=comp_model)
            tables[race] = (tests_tbl, error_tbl)

    return tables

def plot_params(param_table_name):
    params_tbl = pd.read_csv(f"analysis/tables/{param_table_name}.csv")
    for col in params_tbl.columns:
      if 'mean' in col:
        plt.plot(params_tbl[col], label=col)
        
    plt.legend()
    plt.show()
    return

if __name__ == "__main__":
    print('start')
    get_plot_race_dists()
    get_plot_race_subgroup(race="bos")
    print('a')
    # size, mks = 10000, ["10K", "20K", "30K"]
    # get_extrap_scatter("bos", sample_size=size, mks=mks)
    # get_extrap_scatter("nyc", sample_size=size, mks=mks)
    # get_extrap_scatter("chi", sample_size=size, mks=mks)
    # print('all_plots')

    # for race in ["bos", "nyc", "chi"]:
    #     # get_extrap_scatter(race, sample_size=size, mks=mks)
    #     size1, size2 = 2000, 20000
    #     train_yr, test_yr = [2021, 2022, 2023], [2024]
    #     train, test = get_data(racename=race, size_train=size1, size_test=size2, train_lis=train_yr, test_lis=test_yr, save=False)

    #     models, baseline = ["M1", "M2", "M3"], "BL"
    #     test_preds = get_test_preds(test, race, baseline=baseline)
    #     error_tbl = plots_for_race(test_preds, race, models, baseline, comp_model="M2")
    #     print(f"Plotted:  {race}")

    print('fin')
