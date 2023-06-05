import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from openpyxl import load_workbook
from itertools import product

def read_plate_key(in_file, sheet_name="Keys", converters=None):
    """Read the plate keys from a given sheet in an Excel file.
    
    The key sheet should consist of one or more **tables** separated by at
    least one empty row, followed by an uppercase "STOP" (also separated by at
    least one empty row). Rows after the "STOP" are ignored. Everything should
    be aligned to the first column.
    
    The **table** format is as follows:
    The first cell of the first row should store the name of the key, e.g.
    "Strain" or "Media". The next row should be a N-by-12 grid, with every cell
    storing the key of the corresponding well (where the top-left cell
    corresponds to the A1 well). No empty rows are allowed between the key name
    and the key grid. N can be less than 8 - leading empty rows should contain
    at least one cell with '_' in it (which is always interpreted as "ignore 
    this cell"), otherwise all cells can be empty. Trailing empty rows are not
    necessary - the first empty row will indicate an end of the table.
    
    An example with a 3x3 layout (instead of 8x12) where the first row is to be
    ignored:
    
    +------+---+---+
    |Strain|   |   |
    +------+---+---+
    |   _  |   |   |
    +------+---+---+
    |   1  | 2 | 3 |
    +------+---+---+
    |   1  | 2 | 3 |
    +------+---+---+
    |      |   |   |
    +------+---+---+
    | Media|   |   |
    +------+---+---+
    |  _   |   |   |
    +------+---+---+
    | YPD  |YPD|YPD|
    +------+---+---+
    | SDC  |SDC|SDC|
    +------+---+---+
    |      |   |   |
    +------+---+---+
    | STOP |   |   |
    +------+---+---+
    
    Parameters
    ----------
    in_file : str
        The path to the Excel file holding the keys.
    sheet_name : str, default: ``"Keys"``
        The sheet name of the sheet that holds the keys.
    converters : dict of str to callable, optional
        A dictionary of converters for the key values. The dict keys are key
        names, while the dict values are callables that take the string value
        from the sheet and convert it to the correct value for further
        processing in Python.
    
    Returns
    -------
    keys : list
        The keys parsed in the key sheet (list of strings). In the above example
        this would be ``('Strain', 'Media')``.
    key_values : dict of str to tuple
        A dictionary mapping well names to a tuple of
        key values (with the same length as the lengths of the returned `keys`).
        In the above example this would be ``{'A1': (1, 'YPD'), 'A2': (2, 'YPD'),
        'A3': (3, 'YPD'), 'B1': (1, 'SDC'), 'B2': (2, 'SDC'), 'B3': (3, 'SDC')}``.
    """
    
    rows = list("ABCDEFGH")
    cols = list(range(1, 13))
    
    cells = [f"{r}{c}" for r, c in product(rows, cols)]
    result = {c: [] for c in cells}
    
    wb = load_workbook(filename=in_file)
    sheet = wb[sheet_name]
    
    keys = []
    start_ix = stop_ix = -1 # The start and stop ixes of a single key table
    for row_ix, row in enumerate(sheet.iter_rows()):
        key = row[0].value
        if key is None or not str(key).strip():
            # This is an empty row
            if stop_ix < start_ix:
                # We've reached the end of a table
                stop_ix = row_ix - 1
                df_rows = rows[:stop_ix-start_ix+1]
                
                df = pd.read_excel(
                    in_file,
                    sheet_name=sheet_name,
                    skiprows=start_ix,
                    header=None,
                    names=cols,
                    nrows=stop_ix-start_ix+1,
                    engine="openpyxl"
                )
                df["Rows"] = df_rows
                df.set_index("Rows", inplace=True)
                
                converter = converters.get(keys[-1], lambda x: x)
                for row, col in product(df_rows, cols):
                    if pd.isna(df.loc[row, col]) or "_" == df.loc[row, col]:
                        continue
                    result[f"{row}{col}"].append( converter(df.loc[row, col]) )
            else:
                # This is just an empty row, which we allow and simply skip
                continue
        elif stop_ix >= start_ix:
            # We're either expecting the start of a key table or the end of the table list
            key = str(key).strip()
            
            if key.upper() == "STOP":
                break
            
            keys.append(key)
            start_ix = row_ix+1
            
    for key in list(result.keys()):
        if not result[key]:
            del result[key]
        else:
            result[key] = tuple(result[key])
        
    return keys, result

def read_experiment(in_file, sheet_name, key_sheet_name="Keys", converters=None):
    """Parse plate growth curve data into a DataFrame.
    
    Reads the plate data along with the relevant keys (expected to be in the
    same workbook). The DataFrame will have a MultiIndex over the rows as
    parsed from the `key_sheet_name` sheet, with the last level called "Well"
    and storing the well information. The columns are "Time (s)", "OD" and
    "Temp" (temperature).
    
    Parameters
    ----------
    in_file : str
        The path to the Excel input file.
    sheet_name : str
        The name of the sheet holding the plate data.
    key_sheet_name : str, default: ``"Keys"``
        The name of the sheet holding the key definitions.
    converters : dict, optional
        Converters for the key values, see the same parameter in the
        `read_plate_key` function.
    
    Returns
    -------
    pandas.DataFrame
        The parsed OD measurements for the plate.
    """
    
    index_names, cell_indexes = read_plate_key(in_file, key_sheet_name, converters)
    
    wb = load_workbook(filename=in_file)
    sheet = wb[sheet_name]
    
    for row_ix, row in enumerate(sheet.iter_rows()):
        if "Time [s]" in str(row[0].value):
            break
    
    df = pd.read_excel(
        in_file,
        sheet_name=sheet_name,
        skiprows=row_ix,
        header=None,
    ).dropna(axis=0, how="all").dropna(axis=1, how="all").iloc[:-1,:]
    
    time_series = list(df.iloc[0,1:])
    temp_series = list(df.iloc[1,1:])
    
    new_data = {
        "OD": [],
        "Time (s)": [],
        "Temp": [],
    }
    
    index = []
    for well_ix, well in enumerate(df.iloc[2:,0]):
        ods = list(df.iloc[well_ix+2,1:])
        
        new_data["OD"] += ods
        new_data["Time (s)"] += time_series
        new_data["Temp"] += temp_series
        
        index += [cell_indexes[well] + (well,)] * len(ods)
    
    index = pd.MultiIndex.from_tuples(index, names=list(index_names)+["Well"])
    
    new_df = pd.DataFrame(new_data, index=index).sort_index()
    
    return new_df

def plot_ods(
    df,
    x_index=None, y_index=None, x_title=None, y_title=None,
    mean_indices=None,
    std_dev=None, # None, "bar", "area" -- must be used with mean_indices!
    style_func=None,
    legend="last col",
    title_all_axes=False,
    cmap="viridis",
    dpi=150,
    alpha=1,
):
    """Plot the ODs from a DataFrame returned by `read_experiment`.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame with the growth curves to plot.
    x_index : str, optional
        The name of the level in `df`'s index to plot along the X axis
        (columns) of the figure. The values of this level will be used as the
        titles for the first row of Axes in the figure.
    y_index : str, optional
        The name of the level in `df`'s index to plot along the Y axis (rows)
        of the figure. The values of this level will be used as the title of
        the y axis for the first column of the Axes in the figure.
    x_title : str, optional
        If x_index is not specified, this will be used as the title of the
        first Axes in the figure.
    y_title: str, optional
        If y_index is not specified, this will be used as the title of the y
        axis of the first Axes in the figure.
    mean_indices : sequence of str, optional
        The index levels which will be used to group the growth curves for
        averaging. For example, if ``("Strain", "Media")`` is passed, every unique
        strain-media combination will be grouped, and all remaining sub-levels
        will be averaged over to yield a single growth curve, which will then
        be plotted. If different experiments within the group have different time stamps,
        the "missing" measurements will be interpolated before averaging.
    std_dev : {None, 'bar', 'area'}
        If `mean_indices` is specified, this will set the method for displaying
        the standard deviation:
            
        - **bar** - show the standard deviation as whiskers.
        - **area** - show the standard deviation as a semi-transparent area.
    style_func : callable, optional
        A callable that accepts a tuple of ``(x_label, y_label, ix)`` where ix is
        the index of the curve to draw. It should return a dict of kwargs for
        the `Axes.plot` method.
    legend : {"last col", "every axes", "none"}
        - ``"last col"`` will add a legend at the end of each row.
        - ``"every axes"`` will add a legend to every axes.
        - ``"none"`` will not show any legends.
    title_all_axes : bool, default=False
        If True, will set the title for all Axes (not just the top row).
    cmap : str or `matplotlib.colors.Colormap`, default='viridis'
        The colormap to use for the growth curves.
    
    Returns
    -------
    matplotlib.figure.Figure
        The figure with the plots.
    """
    
    if std_dev is not None and mean_indices is None:
        raise ValueError("std_dev is not None but mean_indices was not given!")
    
    if isinstance(cmap, str):
        cmap = cm.get_cmap(cmap)
    
    if mean_indices:
        df = avg_over_ixs(df, mean_indices)
    else:
        # Work on a copy of the df, in case we'll need to modify it:
        df = df.copy()
    
    # If the x/y indices are not specified, create singleton "dummy" levels to
    # allow for the subequent code to be agnostic of these effects.
    if x_index is None:
        x_index = "_dummy_x"
        df[x_index] = ""
        df.set_index(x_index, append=True, drop=True, inplace=True)
    if y_index is None:
        y_index = "_dummy_y"
        df[y_index] = ""
        df.set_index(y_index, append=True, drop=True, inplace=True)
        
    # The code expects at least one level beyond the x/y levels, and if it
    # doesn't exist, create it as we did in case the x/y levels.
    if len(df.index.levels) == 2:
        z_index = "_dummy_z"
        df[z_index] = ""
        df.set_index(z_index, append=True, drop=True, inplace=True)
    
    x_labels = df.index.get_level_values(x_index).unique()
    y_labels = df.index.get_level_values(y_index).unique()
    rows = len(y_labels)
    cols = len(x_labels)
    
    fig, axs = plt.subplots(
        rows, cols, figsize=(cols*3, rows*2),
        sharex=True, sharey=True, dpi=dpi
        )
    if rows == 1 and cols == 1:
        axs = [[axs]]
    elif rows == 1:
        axs = [axs]
    elif cols == 1:
        axs = [[axs[i]] for i in range(len(axs))]
    
    for ix, label in enumerate(x_labels):
        axs[0][ix].set_title(x_title or label)
        
    for ix, label in enumerate(y_labels):
        axs[ix][0].set_ylabel(y_title or label)
        
    for row_ix, y_label in enumerate(y_labels):
        for col_ix, x_label in enumerate(x_labels):
            ax = axs[row_ix][col_ix]
            ax.xaxis.set_tick_params(which='both', labelbottom=True)
            
            # Not all indexes are guaranteed to be in every axes. Skip them if
            # they don't exist.
            if x_label not in df.index.get_level_values(x_index) or \
                y_label not in df.index.get_level_values(y_index):
                continue
            ax_df = df.xs((x_label, y_label), level=(x_index, y_index))
            
            # The same condition can be run in several wells, and this gets
            # special treatment in the condition loop. To set up the condition
            # loop, we need to get all of the condition indexes, but without the
            # well information.
            if "Well" in ax_df.index.names:
                # TODO: this assumes that the Well index is the last one
                # (otherwise, the .loc won't work).
                condition_ixs = ax_df.index.droplevel("Well").unique()
            else:
                condition_ixs = ax_df.index.unique()
                
            for con_ix_count, con_ix in enumerate(condition_ixs):
                con_df = ax_df.loc[con_ix]
                
                # TODO: colors break when an index is missing from one of the axes
                if len(condition_ixs) == 1:
                    color = "black"
                else:
                    color = cmap(1.0*(con_ix_count/(len(condition_ixs)-1)))
                
                if mean_indices is not None:
                    con_df = con_df.copy() # To prevent SettingWithCopyWarning
                    if std_dev is not None:
                        std_dev_df = con_df["OD std"].reset_index()
                        std_dev_df["OD"] = std_dev_df["OD std"]
                    con_df["OD"] = con_df["OD mean"]
                    con_df["Well"] = ""
                    # TODO: the append=False assumes there are no more index levels.
                    con_df.set_index("Well", append=False, drop=True, inplace=True)
                    
                wells = con_df.index.get_level_values("Well").unique()
                
                for well_ix, well in enumerate(wells):
                    well_df = con_df.loc[well]
                    style = {
                        "label": con_ix,
                        "color": color,
                        "alpha": alpha,
                    }
                    if style_func is not None:
                        user_style = style_func(x_label, y_label, con_ix)
                        style.update(user_style)
                    if well_ix > 0:
                        # This will hide this entry from the legend:
                        style["label"] = f"_{style['label']}"
                    
                    xs = well_df["Time (s)"] / 60 / 60
                    ys = well_df["OD"]
                    if not std_dev or std_dev == "area":
                        ax.plot(xs, ys, **style)
                        if std_dev == "area":
                            # reset_index() is required here because the index
                            # of ys is a running number while the index of
                            # std_dev_data is an empty string (due to how it's created)
                            ax.fill_between(
                                xs,
                                ys.reset_index(drop=True) - std_dev_df["OD"],
                                ys.reset_index(drop=True) + std_dev_df["OD"],
                                color=style["color"], alpha=0.2
                            )
                    elif std_dev == "bar":
                        ax.errorbar(
                            xs, ys, yerr=std_dev_df["OD"], errorevery=4,
                            **style
                        )
            
            if len(condition_ixs) == 1 and title_all_axes:
                prev_title = ax.get_title()
                if prev_title:
                    prev_title += "\n"
                ax.set_title(prev_title + str(condition_ixs[0]))
                
            if legend == "every axes":
                legend_without_duplicate_labels(
                    ax,
                    fontsize=8, loc='upper left'
                )
    
    if legend == "last col":
        for ax_row in axs:
            # TODO: this gives a warning if some of the Axes on the row have no
            # values, and in any case will only display the legend for the
            # rightmost Axes.
            legend_without_duplicate_labels(
                ax_row[-1],
                bbox_to_anchor=(1.05, 1),
                loc='upper left', borderaxespad=0., fontsize=6
            )
    
    return fig

def avg_over_ixs(df_in, avg_levels, x_col="Time (s)", y_col="OD", interpolation_method="from_derivatives"):
    """Return a DataFrame that averages all levels over a given set of levels.
    
    If the timestamps differ between any measurement series, measurements will
    be interpolated for the missing timestamps.
    
    The new DataFrame will have `avg_levels` as the row MultiIndex and three
    columns: `x_col`, `y_col mean` and `y_col std` (for standard deviation).
    
    Parameters
    ----------
    df_in : pandas.DataFrame
        The input DataFrame.
    avg_levels : sequence of str
        The index levels to group by for averaging purposes.
    x_col : str, default: "Time (s)"
        The independent variable column name.
    y_col : str, default: "OD"
        The dependent variable column name.
    interpolation_method : str, default: "from_derivatives"
        The interpolation method.
    
    Returns
    -------
    pandas.DataFrame
        A new DataFrame, with `avg_levels` as the new MultiIndex, and three 
    """
    dfs_to_concat = []
    for exp_ix in df_in.index.droplevel([i for i in df_in.index.names if i not in avg_levels]).unique():
        exp_df = df_in.xs(exp_ix, level=avg_levels)[[x_col, y_col]]
        
        avg_dfs = []
        # No assignments are made here, but pandas still gives a warning, hence
        # we suppress it.
        pd.set_option('mode.chained_assignment', None)
        for avg_ix in exp_df.index.unique():
            avg_dfs.append( exp_df.loc[avg_ix].set_index(x_col).sort_index() )
        pd.set_option('mode.chained_assignment','warn')
        
        joined_df = avg_dfs[0]
        for col_suffix, df in enumerate(avg_dfs[1:]):
            joined_df = joined_df.join(df, how="outer", sort=True, rsuffix=str(col_suffix))
            
        for col_name in joined_df.columns:
            joined_df.loc[:,col_name] = joined_df.loc[:,col_name].interpolate(method=interpolation_method)
            
        joined_df.dropna(inplace=True) # For the extreme NaNs that don't get interpolated
        
        mean_s = joined_df.mean(axis=1)
        std_s = joined_df.std(axis=1).fillna(0) # Std. dev. will be NaN if there's only one value to average
        joined_df[f"{y_col} mean"] = mean_s
        joined_df[f"{y_col} std"] = std_s
        
        joined_df = pd.DataFrame({f"{y_col} mean": mean_s, f"{y_col} std": std_s}).reset_index()
        # TODO: is there a better way to set a single multi-index tuple into a dataframe?
        for level_name, level_value in zip(avg_levels, exp_ix):
            joined_df[level_name] = level_value
        
        joined_df.set_index(list(avg_levels), inplace=True)
        dfs_to_concat.append(joined_df)
        
    return pd.concat(dfs_to_concat)

# Example of a style function:
def style_func(x_label, y_label, ax_index):
    return {"label": "", "color": "", "linestyle": ""}

def xss(df, keys, levels, drop_singleton_levels=False):
    level_ixs = [df.index.names.index(l) for l in levels]
    index_dict = {l_ix: vs for (l_ix, vs) in zip(level_ixs, keys)}
    
    def test_index(index):
        for index_ix, index_value in enumerate(index):
            if index_ix in index_dict:
                if index_value not in index_dict[index_ix]:
                    return False
        
        return True
    
    result = df.loc[df.index.map(test_index)]
    
    if drop_singleton_levels:
        for level_ix in range(result.index.nlevels-1, -1, -1):
            if len(result.index.get_level_values(level_ix).unique()) <= 1:
                result = result.droplevel(level_ix)
    
    return result

def add_row_col(df):
    result = df.copy()
    well_values = result.index.get_level_values("Well")
    result["Row"] = [w[0] for w in well_values]
    result["Column"] = [w[1:] for w in well_values]
    result.set_index("Row", inplace=True, append=True)
    result.set_index("Column", inplace=True, append=True)
    
    return result

def plot_plate(df):
    return plot_ods(
        add_row_col(df), x_index="Column", y_index="Row", legend="every axes"
    )

# Adapted from https://stackoverflow.com/a/56253636
def legend_without_duplicate_labels(ax, **kws):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique), **kws)
