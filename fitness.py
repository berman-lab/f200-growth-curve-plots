import pandas as pd
from growth_curves import reorder_indices, xss
from itertools import product
import numpy as np

def get_curveball_fitness(df, with_tqdm=False, models=None):
    import curveball
    import curveball.models as cb_models

    ix_wrapper = lambda i: i
    if with_tqdm:
        from tqdm.notebook import tqdm
        ix_wrapper = tqdm

    df = df.copy()
    df["Time"] = df["Time (s)"] / 60 / 60

    all_models = {
        "ix": [],
        "lag": [],
        "min doubling time": [],
        "max OD": [],
        "models": [],
        "model name": [],
    }

    for ix in ix_wrapper(df.index.unique()):
        all_models["ix"].append(ix)
       
        model_fits = curveball.models.fit_model(df.loc[ix], PLOT=False, PRINT=False, models=models)
        all_models["models"].append(model_fits)
        best_model = model_fits[0]

        all_models["lag"].append(cb_models.find_lag(best_model))
        all_models["min doubling time"].append(cb_models.find_min_doubling_time(best_model))
        all_models["max OD"].append(df.loc[ix, "OD"].max())
        all_models["model name"].append(best_model.model.name)

    return pd.DataFrame(
        all_models,
        pd.MultiIndex.from_tuples(all_models["ix"], names=df.index.names)
    ).drop(["ix"], axis=1)

def move_columns_to_axis(df, columns, level_name, value_col_name):
    dfs_to_concat = []
    for col in columns:
        other_cols = [c for c in columns if c != col]
        subdf = df.loc[:,[c for c in df.columns if c not in other_cols]].\
            rename({col: value_col_name}, axis=1)
        subdf[level_name] = col
        subdf.set_index(level_name, inplace=True, append=True)

        dfs_to_concat.append(subdf)

    return reorder_indices(pd.concat(dfs_to_concat).sort_index())

# TODO: Older fitness computation + glucose calculations for Rhodamine 6G experiments.
# See if still needed and/or if should be improved upon.
"""
Every well gets a nominal fitness computation using a function (e.g., AUC).

Then we need to normalize. The normalization function can take the index of
a well, and return the index relative to which it should be normalized.

Optionally, we can avg out the wells.
"""

def auc_fitness(data):
    from numpy import trapz
    
    return trapz(data["OD"], data["Time (s)"])

def get_nominal_fitness(df, fit_func):
    # AUC caclulations are more interpretable if the baseline is zero, and not
    # some high arbitrary number.
    # TODO: allow normalizing against a sterile control.
    df = df.copy()
    df["OD"] = df["OD"] - df["OD"].min()
    data = {"Nominal": []}
    index = []
    
    for row_ix in df.index.unique():
        nominal_fit = fit_func(df.loc[row_ix])
        data["Nominal"].append(nominal_fit)
        index.append(row_ix)
    
    return pd.DataFrame(data, pd.MultiIndex.from_tuples(index, names=df.index.names)).sort_index()
    
def normalize_nominal_fit_df(nominal_fit_df, norm_func):
    # Average out the fitnesses:
    # nominal_mean_df = nominal_fit_df.groupby(level=nominal_fit_df.index.names[:-1]).mean()
    # result = nominal_mean_df.rename({"Nominal": "Normalized"}, axis="columns")
    result = nominal_fit_df.groupby(level=nominal_fit_df.index.names[:-1]).mean().rename({"Nominal": "Nominal mean"}, axis="columns")
    result["Normalized"] = -1
    
    for ix in result.index.unique():
        result.loc[ix, "Normalized"] = result.loc[ix]["Nominal mean"] / result.loc[norm_func(ix)]["Nominal mean"]
    
    return result

def norm_by_od(df, od_df):
    """Normalize the values in a given DataFrame by ODs from another DataFrame.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to normalize in the format returned by `read_experiment`-like
        functions.
    od_df : pandas.DataFrame
        DataFrame of ODs. Rows should be indexed A-to-F, and columns 1-to-12.
        `df` will be normalized by using the 'Well' index level.
        
    Returns
    -------
    pandas.DataFrame
        A normalized copy of `df`.
    """
    
    result = df.copy()

    pre_well_ixs = (slice(None),) * (result.index.nlevels-1)
    for r, c in product(od_df.index, od_df.columns):
        ix = pre_well_ixs + (f"{r}{c}",)
        
        if not pd.isna(od_df.loc[r, c]) and \
            ix[-1] in df.index.get_level_values("Well"):
            result.loc[ix, "OD"] /= od_df.loc[r, c]
    
    return result

def normalize_glucose(df):
    """For R6G efflux experiments - normalize the readings from +Glu to -Glu
    condition.
    
    Assumes that every -/+ Glu condition has exactly two wells - so if there
    are any repeats, they either need to be separated by a key in the index
    or handled separately.
    
    Only the OD and Time (s) columns will be kept. The "Well" index will be
    chosen arbitrarily from the two -/+ Glu wells.
    
    Parameters
    ----------
    df : pandas.DataFrame
    
    Returns
    -------
    pandas.DataFrame
        A normalized version of `df`. The "Glucose" index level will not be
        part of the index.
    """
    
    diff_dict = {"OD": [], "Time (s)": []}
    index = []
    
    no_gluc_df = xss(df, [[0]], ["Glucose"], drop_singleton_levels=True)
    add_gluc_df = xss(df, [[1]], ["Glucose"], drop_singleton_levels=True)

    for ix in df.index.droplevel(["Glucose", "Well"]).unique():
        without_gluc = no_gluc_df.loc[ix].sort_values("Time (s)")
        with_gluc = add_gluc_df.loc[ix].sort_values("Time (s)")

        well = with_gluc.index[0]

        delta = with_gluc["OD"].reset_index(drop=True) - without_gluc["OD"].reset_index(drop=True)

        diff_dict["OD"] += list(delta)
        diff_dict["Time (s)"] += list(without_gluc["Time (s)"])

        index += [ix + (well,)]*len(delta)

    index = pd.MultiIndex.from_tuples(index, names=df.index.droplevel(["Glucose"]).names)
    return pd.DataFrame(diff_dict, index=index).sort_index()

################################################################################
# Utilities
################################################################################

# Sanitizing the growth curves before fitness estimations is strongly recommended.
# Artifacts in reading the OD can really mess with naive estimators.

def sanitize_growth_curves(df):
    result = df.copy()
    
    # For some reason, sometimes a few of the first readings have a high OD, which then collapses back to the baseline.
    # We should filter them out:
    ### This version looks for the first cutoff after which the ODs are monotonically rising.
    ### This can cut most of the 'EMPTY' curves, which can create problems for the fitness estimations.
    ### We could filter them out, but it's best to think of another method of filtering.
    # def filter_spikes(df):
    #     min_od = df["OD"].min()
    #     ix_cutoff = 0
    #     for od in df["OD"]:
    #         # if od > min_od * 1.5:
    #         if od > min_od:
    #             ix_cutoff += 1
    #         else:
    #             break
    #     if ix_cutoff == 0:
    #         return df
    #     else:
    #         # if ix_cutoff > 2:
    #         #     print(df.index[0], ix_cutoff)
    #         return df.iloc[ix_cutoff:]

    # This method cuts the first monotonically decreasing portion:
    def filter_spikes(df):
        diff = df["OD"].iloc[:-1] > df["OD"].iloc[1:]
        ix_cutoff = diff.argmin()
        return df.iloc[ix_cutoff:]

    # TODO For some reason, an index level is added which is just the tuple of all other indices?!
    result = result.groupby(result.index).apply(filter_spikes).reset_index(level=0, drop=True)
    
    return result

################################################################################
# Model-free fitness
################################################################################

# Nominal fitness 

def get_model_free_fitness_single(df, lag_threshold=2, plot=False):
    # Assumes `df` contains a single growth curve.
    # Returns the following columns:
    # * Max OD
    # * Lag (s)
    # * Lag (hr) # Given in hours, rounded to 2 decimal points.
    # * Max slope
    # * Max slope (smoothed) # The max slope of the growth curve after smoothing (see code for reference).
    result = {}
    
    result["Lag (s)"] = result["Lag (hr)"] = None
    cutoff_ods = df["OD"] >= df["OD"].min() * lag_threshold
    if cutoff_ods.any():
        # We want the first timepoint so that all subsequent ODs will be bigger than the cutoff.
        # result["Lag"] = df[ df["OD"] >= media_od_cutoffs[media] ]["Time (s)"].iloc[0] /60/60
        s = (~cutoff_ods).reset_index()["OD"]

        # The first numerical index of the timepoint where the OD is always larger than the cutoff:
        tt_od_index = s.where(s).last_valid_index()+1
        if tt_od_index < len(df):
            result["Lag (s)"] = df.iloc[tt_od_index]["Time (s)"]
            result["Lag (hr)"] = round(result["Lag (s)"] / 60 / 60, 2) # In hours, rounded, for convenience.

    result["Max OD"] = df["OD"].max()

    slope = np.gradient(df["OD"])
    result["Max slope"] = slope.max()

    # Because of potential "jumps" in the growth curve, we smooth it and recalculate the slope.
    # Parameters chosen empirically.
    from scipy.signal import savgol_filter
    try:
        smooth_od = savgol_filter(df["OD"], 10, 3)
    except:
        print(df.index.unique())
        raise
    result["Max slope (smoothed)"] = np.gradient(smooth_od).max()

    if plot:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(nrows=3, figsize=(10, 10))
        axs[0].set_title(f"{df.index.unique().get_level_values(0)[0]} {df.index.unique().get_level_values(1)[0]}")
        axs[0].plot(range(len(df)), df["OD"], color="blue")
        axs[0].plot(range(len(df)), smooth_od, color="red")
        axs[1].plot(range(len(slope)), slope, color="blue")
        axs[1].plot(range(len(slope)), np.gradient(smooth_od), color="red")
        axs[2].plot(range(len(slope)), np.gradient(slope))
    
    return pd.Series(result)

def get_model_free_fitness_df(df):
    return df.groupby(df.index.names).apply(get_model_free_fitness_single)


################################################################################
# MICs
################################################################################

def get_mic_distributions(df, timepoint=24, inhibition=0.5):
    df_flc0 = df.xs(0, level="FLC")["OD"]
    gb_obj = df_flc0.groupby(df_flc0.index.names)
    target_ods_series = (gb_obj.max() + gb_obj.min()) * (1-inhibition)
    target_ods_series = target_ods_series.groupby(level=target_ods_series.index.names.difference(["_Source"])).mean()
    
    timepoints_series = df[["Time (s)"]].groupby(df.index.names).agg( (lambda x: x[x > timepoint*60*60].iloc[0]) ).set_index("Time (s)", append=True)
    ods_at_timepoint_series = df.set_index("Time (s)", append=True)["OD"].loc[timepoints_series.index].reset_index(level="Time (s)", drop=True)
    # TODO: we have to remove the "FLC" column when accessing target_ods_series, currently assuming it's the second column.
    is_below_target_od_series = ods_at_timepoint_series.to_frame().apply(lambda x: x[0] < target_ods_series.loc[x.name[:1] + x.name[2:-1]], axis=1)
    
    gb_obj = is_below_target_od_series.groupby(is_below_target_od_series.index.names.difference(["_Source"]))
    mics_df = (gb_obj.sum() / gb_obj.count()).unstack(level="FLC")

    return mics_df

def get_mic_ranges(df, threshold=1, agg_col=None):
    flcs = df.columns
    max_flc = max(flcs)
    min_flc = flcs[1]
    over_flc = max_flc+1
        
    result = df.apply(lambda x: x.idxmax() if x.max() >= threshold else over_flc, axis=1)

    def flc_to_mic(flc):
        return "0?" if flc == 0 else f"<{flc}" if flc == min_flc else f">{max_flc}" if flc == over_flc else flc
    
    if agg_col is not None:
        result = result.groupby(result.index.names.difference([agg_col])).\
            agg(
                from_mic=lambda x: flc_to_mic(x.min()),
                to_mic=lambda x: flc_to_mic(x.max())
            )
    else:
        result = result.apply(flc_to_mic)
    
    return result