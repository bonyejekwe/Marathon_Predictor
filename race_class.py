
from typing import List
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import str_to_int_time, int_to_str_time, time_to_pace, conv1, last_map, get_predictions

np.random.seed(2024)
marks = ["5K", "10K", "15K", "20K", "25K", "30K", "35K", "40K"]
stan_path = "stan_results/params_all3.csv"

class RaceSplits():

    marks = ["5K", "10K", "15K", "20K", "25K", "30K", "35K", "40K"]
    last_dist =  {"5K": "5K",  "10K": "5K",  "15K": "10K",  "20K": "15K",  "25K": "20K",  
                  "30K": "25K", "35K": "30K", "40K": "35K", "Finish": "40K", "Finish": "Finish"}

    def __init__(self):
        self.stored_times = {}
        self.stored_paces = []
        self.prop = 0
        self.bttn_count = 0
        self.city = "bos"

    def add_pace(self, dist: str, time: str, split=False):
        assert dist in marks
        num_time = str_to_int_time(time)
        if num_time is None:
            raise ValueError("Enter a valid time in MM:SS or HH:MM:SS format")
        if (split) and (dist != "5K"):
            assert self.last_dist[dist] in self.stored_times.keys(), "error here"
            last_time = self.stored_times[self.last_dist[dist]]
            num_time += last_time
                
        self.stored_times[dist] = num_time
        self.load_paces()

    def load_paces(self):
        self.stored_paces = []
        for m in marks:
            last_m = self.last_dist[m]
            if (m in self.stored_times.keys()) and (last_m in self.stored_times.keys()):
                curr_time, last_time = self.stored_times[m], self.stored_times[last_m]
                total_pace = conv1[m] / curr_time
                if m == "5K":
                    curr_pace = conv1["5K"] / curr_time
                else:
                    curr_pace = conv1["5K"] / (curr_time - last_time)
                prop = conv1[m] / conv1["Finish Net"]
                self.stored_paces.append((m, curr_pace, total_pace, prop))

    def get_stored_paces(self):
        return pd.DataFrame(self.stored_paces, columns=["dist", "curr_pace", "total_pace", "prop"])
    
    def get_person_dict(self):
        person = self.get_stored_paces()
        person[["total_pace", "curr_pace"]] = time_to_pace(person[["total_pace", "curr_pace"]], "Finish Net")
        person_dict = {row["dist"]: (row["dist"], row["total_pace"], row["curr_pace"]) for _, row in person.iterrows()}
        return person_dict
    

    def posterior_array(self, show: list = ["10K", "15K"]):
        info = self.get_stored_paces()
        info = info[info["dist"].isin(show)]
        info['alpha'] = 1
        info['lvl'] = (info['dist'].str[:-1].astype(int) / 5).astype(int)
        feats_lis = ["alpha", "total_pace"]
        info['race'] = self.city
        info['Race'] = self.city
        info = pd.get_dummies(info, columns=["race"], dtype=int)
        preds = (42195 / 60) / get_predictions(info, stan_path, feats_lis=feats_lis, full=True, curr_pace=True)
        return preds
    
    def stored_dists(self):
        return list(self.stored_times.keys())
    
    def reset_race(self):
        # self.__init__()
        self.stored_times = {}
        self.stored_paces = []
        self.prop = 0
        self.bttn_count = 0
        self.city = "bos"


def table_info(info: pd.DataFrame, show = ["5K", "10K"]):
    percentiles = np.percentile(info, [2.5, 10, 25, 50, 75, 90, 97.5], axis=1)
    percentile_names = ["lower95", "lower80", "lower50", "median", "upper50", "upper80", "upper95"]
    table = pd.DataFrame(percentiles, index=percentile_names, columns=show)
    for col in table:
        table[col] = table[col].apply(lambda x: int_to_str_time(60 * x, no_secs=True))
    return table


def get_from_info(
    race: RaceSplits,
    name: str = "",
    actual=None,
    show: list = ["5K", "10K", "15K", "20K", "25K", "30K", "35K", "40K"]
):
    """Returns figure and table"""
    plt.close("all")
    marks = ["5K", "10K", "15K", "20K", "25K", "30K", "35K", "40K"]
    stored_paces = race.get_stored_paces()
    shows = [m for m in marks if m in list(stored_paces["dist"])] if not stored_paces.empty else []
    shows = [m for m in shows if m in show]
    fig = plt.figure(figsize=(10, 6))

    if not shows:
        plt.title(f"{name} Live Prediction")
        plt.xlabel("Time (HH:MM)")
        plt.ylabel("Probability")
        plt.savefig("analysis/plots/plot_live.jpg", dpi=300)
        empty_table = pd.DataFrame(columns=["Split", "Input Time", "Median", "50% CI", "80% CI", "95% CI"])
        return fig, empty_table

    p_array = race.posterior_array(shows)
    percentile_info = table_info(p_array, show=shows)
    
    table = []
    colors = plt.get_cmap()(np.linspace(0.2, 0.8, len(shows)))
    for i, dist in enumerate(shows):
        info = percentile_info[dist]
        # print('info', race.stored_times[dist], int_to_str_time(race.stored_times[dist]))
        input_time = int_to_str_time(race.stored_times[dist])
        range50 = f'{info["lower50"]}-{info["upper50"]}'
        range80 = f'{info["lower80"]}-{info["upper80"]}'
        range95 = f'{info["lower95"]}-{info["upper95"]}'
        info_strings = (dist, input_time, f'{info["median"]}', range50, range80, range95)
        table.append(info_strings)
        sns.kdeplot(p_array[i], color=colors[i], label=dist, linewidth=2, alpha=0.9, bw_adjust=1.5)


    x_labels = plt.xticks()[0]
    y_labels = plt.yticks()[0]
    plt.xticks(x_labels, [int_to_str_time(60 * t, no_secs=True) for t in x_labels])#, fontsize=16)
    plt.yticks(y_labels, [round(float(y), 3) for y in y_labels])#, fontsize=16)
    
    plt.xlabel("Time (HH:MM)")#, fontsize=18)
    plt.ylabel("Probability")#, fontsize=18)
    plt.title(f"{name} Live Prediction")#, fontsize=20)
    if actual:
        plt.vlines(actual / 60, 0, plt.yticks()[0].max(), linestyles="dashed", color="black", label="actual")

    plt.legend()#fontsize=16)
    plt.savefig("analysis/plots/plot_live.jpg", dpi=300)
    print('done')

    table_df = pd.DataFrame(table, columns=["Split", "Input Time", "Median", "50% CI", "80% CI", "95% CI"])
    return fig, table_df


def get_race_for_person(num, train_data,) -> RaceSplits:
    mlis = ["5K", "10K", "15K", "20K", "25K", "30K", "35K", "40K"]
    # train_data = pd.read_csv("processed_data/nucr_runners.csv")
    person = train_data.iloc[num]
    times = list(person[mlis].apply(lambda x: int_to_str_time(x)))

    person_race = RaceSplits()
    for m, t in zip(mlis, times):
        person_race.add_pace(m, t)

    person_race.load_paces()
    actual = person["Finish Net"]
    return person_race, actual


if __name__ == "__main__":
    nucr_filename = "processed_data/nucr_runners.csv"
    nucr = pd.read_csv(nucr_filename)
    race, act = get_race_for_person(1, train_data=nucr)
    print('a', act)
    print(race)
    fig, table = get_from_info(race, actual=act)
    print(table)

    print('hi')
    r = RaceSplits()
    r.add_pace("5K", "19:31")
    r.add_pace("10K", "38:15")
    print(r.get_stored_paces())

    marks = ["5K", "10K", "15K", "20K", "25K", "30K", "35K", "40K"]
    print(marks.index("15K"))

    rs = RaceSplits()
    # print('paces', rs.get_paces())
    print('curr_dist', rs.stored_dists())
    rs.add_pace("5K", "15:45")

    # print('paces', rs.get_paces())
    print('curr_dist', rs.stored_dists())
    rs.add_pace("10K", "32:12")
    rs.add_pace("15K", "49:57")
    rs.get_person_dict()
    rs.add_pace("20K", "64:46")
    rs.get_person_dict()
    print('s', rs.get_stored_paces())

    print(rs.stored_times)
    # print(rs.stored_times_table())

    # nucr_filename = "processed_data/nucr_runners.csv"
    # nucr = pd.read_csv(nucr_filename)
    # r = RaceSplits()
    # t = get_race_for_person(0, nucr)
    # print(t)
    # p = get_from_info(t[0])
    # print(p)
