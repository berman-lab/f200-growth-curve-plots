import pandas as pd
from growth_curves import reorder_indices, xss
from itertools import product

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
