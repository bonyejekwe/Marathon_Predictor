
# utils.py: utility functions
from typing import Union
import numpy as np
import pandas as pd
from scipy import stats

np.random.seed(2024)

def str_to_int_time(time: str)  -> Union[int, None]:
    """Convert string time (HH:MM:SS) to int time"""
    try:
        times = time.split(":")[::-1]
        seconds = sum(x * int(time) for x, time in zip([1, 60, 3600], times))
        return int(seconds)
    except ValueError:
        return None
    except AttributeError:
        return None

def int_to_str_time(time: int, no_secs: bool = False)  -> Union[str, None]:
    """Convert int time (in mins) to str time (MM:SS) or (HH:MM:SS)"""
    secs = int(time % 60)
    time2 = int((time - secs) / 60)  # should be int
    if time2 < 60:
        mins, secs = str(time2).zfill(2), str(secs).zfill(2)
        return "00" + ":" + mins + ":" + secs
        # return mins + ":" + secs
    else:
        mins = int((time2 % 60))
        hrs = int((time2 - mins) / 60)
        hrs, mins, secs = str(hrs).zfill(2), str(mins).zfill(2), str(secs).zfill(2)

        if no_secs:
            return hrs + ":" + mins
        return hrs + ":" + mins + ":" + secs

def binning(data: pd.Series):
    """Return number of bins for series of data (for each integer)"""
    return int(data.max()) - int(data.min()) + 1

marks = ["5K", "10K", "15K", "20K", "25K", "30K", "35K", "40K"]

last_map = {
    "10K": "5K", "15K": "10K", "20K": "15K",
    "25K": "20K", "30K": "25K", "35K": "30K",
    "40K": "35K", "Finish Net": 42_195,
}

conv1 = {
    "5K": 5_000, "10K": 10_000, "15K": 15_000, 
    "20K": 20_000, "25K": 25_000, "30K": 30_000,
    "35K": 35_000, "40K": 40_000, "Finish Net": 42_195,
}

def time_to_pace(time, dist):
    """Convert time to pace"""
    secs = time * 60
    return conv1[dist] / secs


def pace_to_time(pace, dist):
    """Convert pace to time"""
    secs = conv1[dist] / pace
    return secs / 60

def process_df(data):
    """Processes data. Input dataframe has columns=[Age, M/F, 5K, 10K, ... 40K, HALF, Finish Net, Year], while
    output dataframe has columns=[dist, curr_pace, total_pace, finish, age, gender, year]"""
    new_idx, new_dist, new_mark, new_fin, new_last = [], [], [], [], []
    new_age, new_gender, new_year = [], [], []
    for dist in marks:
        new_idx.extend(list(data.index))
        new_dist.extend([dist] * len(data))
        new_mark.extend(conv1[dist] / data[dist])
        new_fin.extend(conv1["Finish Net"] / data["Finish Net"])
        if dist == "5K":
            last = data["5K"]
        else:
            last = data[dist] - data[last_map[dist]]
        new_last.extend(conv1["5K"] / last)
        new_age.extend(data["Age"])
        new_gender.extend(data["M/F"])
        new_year.extend(data["Year"])

    new_df = pd.DataFrame({
        "id": new_idx, "dist": new_dist, "curr_pace": new_last, "total_pace": new_mark, "finish": new_fin,
        "age": new_age, "gender": new_gender, "year": new_year
    })
    new_df["male"] = (new_df['gender'] == "M").astype(int)
    new_df["malexage"] = new_df["male"] * new_df["age"]
    new_df["alpha"] = 1
    new_df['lvl'] = (new_df['dist'].str[:-1].astype(int) / 5).astype(int)
    return new_df

def get_data(racename="bos", size_train=50, size_test=50, train_lis=[2022], test_lis=[2023], save=False, seed=2025):
    """Get and process data from filepath. After processing data, this function samples the training and test data
    based on training and test data specifications for size and years. Returns train and test data"""
    d = pd.read_csv(f"processed_data/full_data_{racename}.csv")
    train_years, test_years =  d[d["Year"].isin(train_lis)], d[d["Year"].isin(test_lis)]
    xtrain = process_df(train_years)
    xtest = process_df(test_years)

    if size_train != None:
        xtrain = xtrain.sample(n=size_train, random_state=seed).sort_values("lvl")
    
    if size_test != None:
        xtest = xtest.sample(n=size_test, random_state=seed).sort_values("lvl")

    if save:
        # break perfect equality for 5K rows so curr_pace != total_pace (avoids collinearity/identifiability)
        ix_train = xtrain['dist'] == '5K'
        if ix_train.any():
            xtrain.loc[ix_train, 'curr_pace'] = xtrain.loc[ix_train, 'total_pace'] * (1 + 1e-8)
        ix_test = xtest['dist'] == '5K'
        if ix_test.any():
            xtest.loc[ix_test, 'curr_pace'] = xtest.loc[ix_test, 'total_pace'] * (1 + 1e-8)

        xtrain.to_csv(f"processed_data/train_{racename}.csv")
        xtest.to_csv(f"processed_data/test_{racename}.csv")
    return xtrain, xtest

def save_data(race_list, size_train=50, size_test=50, train_lis=[2022], test_lis=[2023], seed=2025):
    for race in race_list:
        get_data(racename=race, size_train=size_train, size_test=size_test, train_lis=train_lis, test_lis=test_lis, save=True, seed=seed)
    return

#######

def get_preds(test_data, stan_data, feats_lis, name="stan_pred", propleft=False, full=False):
    """Get predictions from test data using stan results. The feat_lis columns in test_data correspond
    with the beta_lis columns in stan_data."""
    beta_lis = [f"beta.{i+1}" for i in range(len(feats_lis))]

    d1 = test_data[feats_lis]
    d2 = stan_data[beta_lis].T.values

    norm_mean = stan_data["alpha"] + d1.dot(d2)

    if full:
        if propleft: 
            norm_std = np.outer(test_data["propleft"], stan_data["sigma"])
        else:
            norm_std = stan_data["sigma"]
        preds = np.random.normal(norm_mean, norm_std)
        return preds
    else:
        preds = norm_mean.mean(axis=1) #preds.mean(axis=1)
        return preds
    

def _get_lvl_params(stan_data, lvl, num_feats):
    betas = stan_data[[f"beta.{lvl}.{num+1}" for num in range(num_feats)]].T.values
    sigma = stan_data[f"sigma.{lvl}"].values
    return betas, sigma

def _preds(x, feats, params, full=True):
    betas, sigma = _get_lvl_params(params, x["lvl"].iloc[0], len(feats))
    xfeats = x[feats].values
    pred_means = xfeats.dot(betas)
    if full:
        preds = np.random.normal(pred_means, sigma)
        return preds
    else:
        preds = pred_means.mean(axis=1)
        return preds
    
def get_predictions(test_data, stan_path, feats_lis, full=False):
    stan_data = pd.read_csv(stan_path)
    group = test_data.groupby("lvl", group_keys=False)
    result = group.apply(lambda x: _preds(x, feats_lis, stan_data, full))
    return np.concatenate(list(result))

def other_stats(data, finish, rnd=3):
    """Return overall MAE and R-squared values for specified columns in data"""
    ftime = (42195/60) / finish
    tss = (((ftime) - (ftime).mean()) ** 2).sum()
    tbl = data.apply(lambda x: (x.abs().mean(), 1 - ((x ** 2).sum()/ tss)))  # overall MAE, R^2
    tbl = tbl.set_index([["Overall MAE", "Overall $R^2$"]])
    return tbl.round(rnd)

def get_table(test_data, model_preds, baseline_name="extrap"):
    """Get table that outpus all information to compare models"""
    test_new = test_data.copy()
    y_true = (42195 / 60) / test_new["finish"]
    extrap = (42195 / 60) / test_new["total_pace"]
    
    test_new[baseline_name] = extrap - y_true
    for name, pred in model_preds.items():
        test_new[name] = ((42195 / 60) / pred) - y_true

    return test_new

def error_table(test_data, labels: list):
    """table of MAE values from predictions"""
    mks = ["5K", "10K", "15K", "20K", "25K", "30K", "35K", "40K"]
    table_group = test_data.groupby(["dist"])[labels].apply(lambda x: x.abs().mean()).loc[mks]
    return table_group

def group_data(test_data, group_feat, lbls:list, num_groups=4, pref="G", group_name="group", bins=None):
    data = test_data.copy()
    mks = ["5K", "10K", "15K", "20K", "25K", "30K", "35K", "40K"]
    if bins == None:
        bins = np.percentile(data[group_feat], [100 * i / num_groups for i in range(num_groups)])
    data[group_name] = [f"{pref}{g}" for g in np.digitize(data[group_feat], bins=bins)]
    group = data.groupby(["dist", group_name])[lbls].apply(lambda x: x.abs().mean()).unstack().loc[mks].swaplevel(0, 1, axis=1)
    group2 = group.set_axis([f"{a}_{b}" for a, b in group.columns], axis=1).sort_index(axis=1)
    return group2

def add_intervals_to_test(data_tbl, m_preds, pred_names):
    data = data_tbl.copy()
    for pred_name in pred_names:
        for conf, lower, upper in [(50, 25, 75), (80, 10, 90), (95, 2.5, 97.5)]:
            test_true = (42195 / 60) / data["finish"]
            b1, b2 = np.percentile(m_preds[pred_name], [lower, upper], axis=1)
            data[f"{pred_name}-lower{conf}"] = b1
            data[f"{pred_name}-upper{conf}"] = b2
            data[f"{pred_name}-size{conf}"] = b2 - b1
            data[f"{pred_name}-in{conf}"] = (test_true < b2) & (test_true > b1)
    return data

def marathons_table():
    pd.concat([pd.read_csv(f"analysis/tables/{race}_error.csv", index_col="dist").rename({"BL": f"BL_{race}", "M2": f"M2_{race}"}, axis=1)[[f"BL_{race}", f"M2_{race}"]] for race in ["bos", "nyc", "chi"]], axis=1).reset_index().to_csv("analysis/tables/marathons.csv", index=False)
    
def get_race_info():
    race_dfs = []
    for race, name in [("bos", "Boston"), ("nyc", "New York"), ("chi", "Chicago")]:
        df = pd.read_csv(f"processed_data/full_data_{race}.csv")
        df["Marathon"] = name
        race_dfs.append(df)

    big_df = pd.concat(race_dfs)
    arr1 = big_df.groupby(["Marathon", "Year"])["Name"].count()
    arr2 = big_df[big_df["M/F"] == "M"].groupby(["Marathon", "Year"])["Name"].count()
    arr3 = big_df[(big_df["M/F"] == "F") | (big_df["M/F"] == "W")].groupby(["Marathon", "Year"])["Name"].count()#.values

    arrs = {"Total": arr1, "M": arr2, "W": arr3}

    for age, lower, upper in [["A1", 0, 30], ["A2", 31, 40], ["A3", 41, 50], ["A4", 51, 101]]:
    # for age, lower, upper in [["A1", 0, 34], ["A2", 35, 45], ["A3", 46, 101]]:
        arrs[f"AG {lower}-{upper}"] = big_df[(big_df["Age"] >= lower) & (big_df["Age"] <= upper)].groupby(["Marathon", "Year"])["Name"].count()

    pd.DataFrame(arrs).to_csv("analysis/tables/race_info.csv")
    return big_df

def all_tests(data, test_list, savename):
    tbl = []
    for lbl1, lbl2 in test_list:
        row = []
        arr1, arr2 = data[lbl1], data[lbl2]
        value = stats.ks_2samp(arr1, arr2).pvalue
        row.append(value)
        value = stats.wilcoxon(arr1, arr2).pvalue
        row.append(value)
        value = stats.cramervonmises_2samp(arr1, arr2).pvalue
        row.append(value)
        value = stats.anderson_ksamp([arr1, arr2]).pvalue
        row.append(value)
        tbl.append(row)

    idx = [f"{lbl1}-{lbl2}" for lbl1, lbl2 in test_list]
    df = pd.DataFrame(tbl, index=idx, columns=["KS", "Wilcoxon", "CVM", "AD"]).round(4).replace(0.0000, "<0.0001")
    df.to_csv(savename)
    return df

def save_param_values(race, rnd=4):
    """Save paramater values into analysis/tables"""
    param_lists = []
    mks = ["5K", "10K", "15K", "20K", "25K", "30K", "35K", "40K"]
    params = pd.read_csv(f"stan_results/model2/params_{race}.csv", index_col="Unnamed: 0")
    p1 = params.describe().T["mean"]
    p2 = params.describe().T["std"]
    param_lists.append(p1[[f"beta.{i}.1" for i in range(1, 9)]].round(rnd).set_axis(mks).rename('alpha_mean'))
    param_lists.append(p2[[f"beta.{i}.1" for i in range(1, 9)]].round(rnd).set_axis(mks).rename('alpha_std'))
    param_lists.append(p1[[f"beta.{i}.2" for i in range(1, 9)]].round(rnd).set_axis(mks).rename('total_pace_mean'))
    param_lists.append(p2[[f"beta.{i}.2" for i in range(1, 9)]].round(rnd).set_axis(mks).rename('total_pace_std'))
    param_lists.append(p1[[f"beta.{i}.3" for i in range(1, 9)]].round(rnd).set_axis(mks).rename('curr_pace_mean'))
    param_lists.append(p2[[f"beta.{i}.3" for i in range(1, 9)]].round(rnd).set_axis(mks).rename('curr_pace_std'))
    param_lists.append(p1[[f"sigma.{i}" for i in range(1, 9)]].round(rnd).set_axis(mks).rename('sigma_mean'))
    param_lists.append(p2[[f"sigma.{i}" for i in range(1, 9)]].round(rnd).set_axis(mks).rename('sigma_std'))
    param_df = pd.concat(param_lists, axis=1)
    param_df.to_csv(f"analysis/tables/{race}_params.csv")
    return param_df

def get_test_preds(test_data, race: str, baseline = "BL", full=False):
    model_info = [
        ("M1", f"stan_results/model1/params_{race}.csv", ["alpha", "total_pace"]),
        ("M2", f"stan_results/model2/params_{race}.csv", ["alpha", "total_pace", "curr_pace"]),
        ("M3", f"stan_results/model3/params_{race}.csv", ["alpha", "total_pace", "curr_pace", "male", "age"]),
    ]
    mpreds = {name: get_predictions(test_data, path, feats_lis=feats, full=full) for (name, path, feats) in model_info}
    if full:
        mpreds2 = {k: (42195 / 60) / v for k, v in mpreds.items()}
        return mpreds2
    else:
        test2 = get_table(test_data, mpreds, baseline_name=baseline)
        return test2

if __name__ == '__main__':
    pass
