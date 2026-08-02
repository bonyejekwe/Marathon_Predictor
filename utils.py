
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

marathon_map = {
    "bos": "Boston Marathon", 
    "nyc": "New York Marathon", 
    "chi": "Chicago Marathon", 
    "all": "All Marathons"
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
        # if dist == "5K":
        #     new_last.extend([0] * len(data))
        # else:
        #     new_last.extend(conv1["5K"] / last)
        new_last.extend(conv1["5K"] / last)
        new_age.extend(data["Age"])
        new_gender.extend(data["M/F"])
        new_year.extend(data["Year"])

    new_df = pd.DataFrame({
        "id": new_idx, "dist": new_dist, "curr_pace": new_last, "total_pace": new_mark, "finish": new_fin,
        "age": new_age, "gender": new_gender, "year": new_year
    })
    new_df["male"] = (new_df['gender'] == "M").astype(int)
    new_df["alpha"] = 1
    new_df["age_t"] = new_df["age"] / 100
    new_df["agegroup"] = np.digitize(new_df["age"], bins=[0, 30, 40, 50, 100])
    new_df['lvl'] = (new_df['dist'].str[:-1].astype(int) / 5).astype(int)
    return new_df

def get_data(size_train=50, size_test=50, train_lis=[2022], test_lis=[2023], save=False, seed=2025):
    """Get and process data from filepath. After processing data, this function samples the training and test data
    based on training and test data specifications for size and years. Returns train and test data"""
    
    train_data_list, test_data_list = [], []
    for race in ["bos", "nyc", "chi"]:
        d = pd.read_csv(f"processed_data/full_data_{race}.csv")
        train_years, test_years =  d[d["Year"].isin(train_lis)], d[d["Year"].isin(test_lis)]
        xtrain = process_df(train_years)
        xtest = process_df(test_years)
        xtrain["race"], xtest["race"] = race, race
        xtrain["Race"], xtest["Race"] = race, race

        if size_train != None:
            xtrain = xtrain.sample(n=size_train, random_state=seed).sort_values("lvl")
        
        if size_test != None:
            xtest = xtest.sample(n=size_test, random_state=seed).sort_values("lvl")

        train_data_list.append(xtrain)
        test_data_list.append(xtest)

    xtrain = pd.concat(train_data_list)
    xtest = pd.concat(test_data_list)

    xtrain = pd.get_dummies(xtrain, columns=['race'], dtype=int)
    xtest = pd.get_dummies(xtest, columns=['race'], dtype=int)
    
    if save:
        train_path, test_path = "processed_data/train_all.csv", "processed_data/test_all.csv"
        xtrain.to_csv(train_path)
        print(f"File saved: {train_path} ({xtrain.shape[0]} rows)")
        xtest.to_csv(test_path)
        print(f"File saved: {test_path} ({xtest.shape[0]} rows)")

    return xtrain, xtest

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
    

def _get_lvl_race_params(stan_data, lvl, race, num_feats, curr_pace=False):
    # print(race, lvl) 
    if race == "nyc":
        col_nums = list(range(num_feats + 1))
    elif race == "chi":
        col_nums = list(range(num_feats)) + [num_feats + 1]
    else: # race == "bos"
        col_nums = list(range(num_feats))

    beta_cols = [f"beta.{lvl}.{num+1}" for num in col_nums]
    if (curr_pace) and (lvl != 1):
        beta_cols += [f"gamma.{lvl-1}"]
    
    betas = stan_data[beta_cols].T.values
    sigma = stan_data[f"sigma.{lvl}"].values
    return betas, sigma

def _preds(x, feats, params, full=True, curr_pace=False):
    race_name = x["Race"].iloc[0]
    race_lvl = x["lvl"].iloc[0]
    
    betas, sigma = _get_lvl_race_params(params, race_lvl, race_name, len(feats), curr_pace=curr_pace)
    
    if race_name == "nyc":
        feats_ls = feats + ["race_nyc"]
    elif race_name == "chi":
        feats_ls = feats + ["race_chi"]
    else: # race_name == "bos"
        feats_ls = feats

    if (curr_pace) and (race_lvl != 1):
        feats_lis = feats_ls + ["curr_pace"]
    else:
        feats_lis = feats_ls

    xfeats = x[feats_lis].values
    pred_means = xfeats.dot(betas)
    if full:
        preds = np.random.normal(pred_means, sigma)
        return preds
    else:
        preds = pred_means.mean(axis=1)
        return preds
    
def get_predictions(test_data, stan_path, feats_lis, full=False, curr_pace=False):
    stan_data = pd.read_csv(stan_path)
    result = []
    for _, group in test_data.groupby(["lvl", "Race"], group_keys=True):
        result.append(_preds(group, feats_lis, stan_data, full, curr_pace))
    return np.concatenate(result)

def other_stats(data, finish, rnd=3):
    """Return overall MAE, RMSE and R-squared values for specified columns in data"""
    ftime = (42195/60) / finish
    tss = (((ftime) - (ftime).mean()) ** 2).sum()
    tbl = data.apply(lambda x: (x.abs().mean(), np.sqrt((x ** 2).sum()/ tss), 1 - ((x ** 2).sum()/ tss)))  # overall MAE, RMSE, R^2
    tbl = tbl.set_index([["Overall MAE", "Overall RMSE", "Overall $R^2$"]])
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

def mae_table(test_data, labels: list):
    """table of MAE values from predictions"""
    mks = ["5K", "10K", "15K", "20K", "25K", "30K", "35K", "40K"]
    table_group = test_data.groupby(["dist"])[labels].apply(lambda x: x.abs().mean()).loc[mks]
    return table_group

def rmse_table(test_data, labels: list):
    """table of RMSE values from predictions"""
    mks = ["5K", "10K", "15K", "20K", "25K", "30K", "35K", "40K"]
    table_group = test_data.groupby(["dist"])[labels].apply(lambda x: np.sqrt((x ** 2).sum() / len(x))).loc[mks]
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

def all_tests(data, models_list, savename, tests_list=["KS", "CVM", "AD", "Wilcoxon"]):
    """Test all unique pairs of models using comparison tests."""
    model_pairs = []
    for i in range(len(models_list)):
        for j in range(i + 1, len(models_list)):
            model_pairs.append([models_list[i], models_list[j]])

    tbl = []
    for lbl1, lbl2 in model_pairs:
        row = []
        arr1, arr2 = data[lbl1], data[lbl2]
        if "KS" in tests_list:
            value = stats.ks_2samp(arr1, arr2).pvalue
            row.append(value)
        if "CVM" in tests_list:
            value = stats.cramervonmises_2samp(arr1, arr2).pvalue
            row.append(value)
        if "AD" in tests_list:
            value = stats.anderson_ksamp([arr1, arr2]).pvalue
            row.append(value)
        if "Wilcoxon" in tests_list:
            value = stats.wilcoxon(arr1, arr2).pvalue
            row.append(value)
        tbl.append(row)

    idx = [f"{lbl1}-{lbl2}" for lbl1, lbl2 in model_pairs]
    df = pd.DataFrame(tbl, index=idx, columns=tests_list).round(4).replace(0.0000, "<0.0001")
    filename = f"analysis/tables/{savename}_stattest.csv"
    df.to_csv(filename)
    print(f"File saved: {filename}")
    return df

def save_param_values(savename="params3", rnd=4):
    """Save paramater values into analysis/tables/{savename}.csv"""
    params = pd.read_csv(f"stan_results/params_all3.csv", index_col="Unnamed: 0")
    mks = ["5K", "10K", "15K", "20K", "25K", "30K", "35K", "40K"]

    feats_map = [
        ("alpha", [f"beta.{i+1}.1" for i in range(8)]),
        ("total_pace", [f"beta.{i+1}.2" for i in range(8)]),
        ("race_nyc", [f"beta.{i+1}.3" for i in range(8)]),
        ("race_chi", [f"beta.{i+1}.4" for i in range(8)]),
        ("curr_pace", [f"gamma.{i+1}" for i in range(7)]),
        ("sigma", [f"sigma.{i+1}" for i in range(8)]),
    ]

    plis = []
    for name, cols in feats_map:
        if name == "curr_pace":
            plis.append(params[cols].agg(["mean", "std"]).T.round(rnd).set_axis(mks[1:]).rename(columns={"mean": f"{name}_mean", "std": f"{name}_std"}))
        else:   
            plis.append(params[cols].agg(["mean", "std"]).T.round(rnd).set_axis(mks).rename(columns={"mean": f"{name}_mean", "std": f"{name}_std"}))

    param_df = pd.concat(plis, axis=1)
    param_df.to_csv(f"analysis/tables/{savename}.csv")
    return param_df

def get_test_preds(test_data, race: str, baseline = "BL", full=False):
    model_info = [
        ("S1", f"stan_results/params_{race}1.csv", ["alpha", "total_pace"], False),
        ("S2", f"stan_results/params_{race}2.csv", ["alpha", "total_pace", "male", "age_t"], False),
        ("S3", f"stan_results/params_{race}3.csv", ["alpha", "total_pace"], True),
        ("S4", f"stan_results/params_{race}4.csv", ["alpha", "total_pace", "male", "age_t"], True),
    ]
    mpreds = {name: get_predictions(test_data, path, feats_lis=feats, full=full, curr_pace=cp) for (name, path, feats, cp) in model_info}
    if full:
        mpreds2 = {k: (42195 / 60) / v for k, v in mpreds.items()}
        return mpreds2
    else:
        test2 = get_table(test_data, mpreds, baseline_name=baseline)
        return test2

if __name__ == '__main__':
    size = 10000 # size1, size2 = 10000, 10000
    train_yr, test_yr = [2021, 2022, 2023], [2024]
    train, test = get_data(size_train=size, size_test=size, train_lis=train_yr, test_lis=test_yr, save=True)
    print('done')
