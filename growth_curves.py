import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from openpyxl import load_workbook
from itertools import product

def read_plate_key(in_file, sheet_name="Keys", converters=None):
    rows = list("ABCDEFGH")
    cols = list(range(1, 13))
    
    cells = [f"{r}{c}" for r, c in product(rows, cols)]
    result = {c: [] for c in cells}
    
    wb = load_workbook(filename=in_file)
    sheet = wb[sheet_name]
    
    keys = []
    next_ix = -1
    for row_ix, row in enumerate(sheet.iter_rows()):
        if row_ix < next_ix:
            continue
        
        key = row[0].value
        if not key:
            continue
        key = str(key).strip()
            
        if key.upper() == "STOP":
            break
        
        keys.append(key)
        next_ix = row_ix + 9
        
        df = pd.read_excel(
            in_file,
            sheet_name=sheet_name,
            skiprows=row_ix+1,
            header=None,
            names=cols,
            nrows=8
        )
        df["Rows"] = rows
        df.set_index("Rows", inplace=True)
        
        converter = converters.get(key, lambda x: x)
        for row, col in product(rows, cols):
            result[f"{row}{col}"].append( converter(df.loc[row, col]) )
            
    for key in result:
        result[key] = tuple(result[key])
        
    return keys, result

def read_experiment(in_file, sheet_name, key_sheet_name="Keys", converters=None):
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
    
    new_df = pd.DataFrame(new_data, index=index)
    
    return new_df

# This only supports an index of no more than 3 levels, and each axes can only support an index of one level.
# multi_well can be "mean", BUT THIS REQUIRES THE TIME COLUMN TO BE SYNCED ACROSS ALL MEASUREMENTS!
def plot_ods(
    df, x_index=None, y_index=None, x_title=None, y_title=None, cmap=None, style_func=None, multi_well=None, mean_indices=None,
    std_dev=None, # None, "bar", "area"
    dpi=150,
    legend=True,
    title_all_axes=False,
    alpha=1,
):
    if std_dev is not None and multi_well is None:
        multi_well = "mean"
    
    if cmap is None:
        cmap = cm.get_cmap("viridis")
    elif isinstance(cmap, str):
        cmap = cm.get_cmap(cmap)
    
    df = df.copy()
    if x_index is None:
        x_index = "_dummy_x"
        df[x_index] = ""
        df.set_index(x_index, append=True, drop=True, inplace=True)
    if y_index is None:
        y_index = "_dummy_y"
        df[y_index] = ""
        df.set_index(y_index, append=True, drop=True, inplace=True)
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
            
            try:
                df_slice = df.xs((x_label, y_label), level=(x_index, y_index))
            except KeyError:
                # Not all indexes are guaranteed to be in every axes.
                # TODO: is this safe? Should we log this?
                continue
            
            # TODO: this assumes that the Well index is the last one
            # (otherwise, the .loc won't work).
            ixs = df_slice.index.droplevel("Well").unique()
            for ix_ix, ix in enumerate(ixs):
                data = df_slice.loc[ix]
                
                # TODO: colors break when an index is missing from one of the axes
                if len(ixs) == 1:
                    color = "black"
                else:
                    color = cmap(1.0*(ix_ix/(len(ixs)-1)))
                
                if multi_well == "mean":
                    if std_dev is not None:
                        std_dev_data = data.groupby("Time (s)").std().reset_index()
                    data = data.copy().groupby("Time (s)").mean().reset_index()
                    data["Well"] = ""
                    # TODO: the append=False assumes there are no more index levels.
                    data.set_index("Well", append=False, drop=True, inplace=True)
                    
                wells = data.index.get_level_values("Well").unique()
                
                for well_ix, well in enumerate(wells):
                    well_data = data.loc[well]
                    style = {
                        "label": ix,
                        "color": color,
                        "alpha": alpha,
                    }
                    if style_func is not None:
                        user_style = style_func(x_label, y_label, ix)
                        style.update(user_style)
                    if well_ix > 0:
                        style["label"] = f"_{style['label']}"
                    
                    xs = well_data["Time (s)"] / 60 / 60
                    ys = well_data["OD"]
                    if not std_dev or std_dev == "area":
                        ax.plot(xs, ys, **style)
                        if std_dev == "area":
                            ax.fill_between(xs, ys-std_dev_data["OD"], ys+std_dev_data["OD"], color=style["color"], alpha=0.2)
                    elif std_dev == "bar":
                        ax.errorbar(xs, ys, yerr=std_dev_data["OD"], errorevery=4, **style)
            
            if len(ixs) == 1 and title_all_axes:
                prev_title = ax.get_title()
                if prev_title:
                    prev_title += "\n"
                ax.set_title(prev_title + str(ixs[0]))
    
    if legend:
        for ax_row in axs:
            # TODO: this gives a warning if some of the Axes on the row have no
            # values, and in any case will only display the legend for the
            # rightmost Aaxes.
            ax_row[-1].legend(bbox_to_anchor=(1.05, 1),
                             loc='upper left', borderaxespad=0., fontsize=6)
    
    return fig

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

