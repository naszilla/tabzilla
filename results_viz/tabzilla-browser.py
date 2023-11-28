""" An interactivate categorized chart based on a movie dataset.
This example shows the ability of Bokeh to create a dashboard with different
sorting options based on a given dataset.

"""
from os.path import dirname, join
from pathlib import Path

import numpy as np
import pandas as pd
from bokeh.io import curdoc, output_file, save
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Div, Select
from bokeh.plotting import figure

this_file = Path(__file__)

output_file(this_file.parent / f"{this_file.stem}.html")


results_df = pd.read_csv(
    Path(__file__).parent / "cleaned_results/tuned_aggregated_results.csv"
)
metafeatures_df = pd.read_csv(
    Path(__file__).parent / "cleaned_results/agg_metafeatures.csv"
)

# strip the prefix "openml__" from all dataset names
clean_dataset_name = (
    lambda x: x.split("__")[1].replace("_", "-") + " (" + x.split("__")[2] + ")"
)
metafeatures_df.loc[:, "dataset_basename"] = metafeatures_df["dataset_basename"].apply(
    clean_dataset_name
)

results_df.loc[:, "dataset_name"] = results_df["dataset_name"].apply(clean_dataset_name)

# add a dataset indicator variable to cross-reference datasets
dataset_to_index = {
    d: i for i, d in enumerate(metafeatures_df["dataset_basename"].unique())
}

metafeatures_df.loc[:, "dataset_id"] = metafeatures_df["dataset_basename"].apply(
    lambda x: dataset_to_index[x]
)
results_df.loc[:, "dataset_id"] = results_df["dataset_name"].apply(
    lambda x: dataset_to_index[x]
)


# set color and marker for each point
MARKER_DICT = {
    "XGBoost": "star",
    "CatBoost": "circle",
    "LightGBM": "square",
    "DANet": "hex",
    "SAINT": "triangle",
    "DeepFM": "cross",
    "ResNet": "inverted_triangle",
    "FTTransformer": "square",
    "TabTransformer": "asterisk",
    "TabNet": "plus",
    "MLP-rtdl": "circle",
    "MLP": "cross",
    "NODE": "diamond",
    "STG": "star",
    "NAM": "square",
    "VIME": "circle",
    "DecisionTree": "square",
    "KNN": "circle",
    "RandomForest": "cross",
    "LinearModel": "diamond",
    "SVM": "triangle",
    "TabPFN": "star",
}

results_df.loc[:, "marker"] = "x"
for alg, mark in MARKER_DICT.items():
    results_df.loc[results_df["alg_name"] == alg, "marker"] = mark

assert ~results_df["marker"].isna().any()

# baselines = gray, neural = red, gbdt = green
results_df.loc[:, "color"] = "gray"
results_df.loc[results_df["alg_type"] == "neural", "color"] = "red"
results_df.loc[results_df["alg_type"] == "gbdt", "color"] = "green"

# prepare column names for plotting
rename_plot_cols = {}
for metric in ["Log Loss", "Accuracy", "F1", "AUC"]:
    rename_plot_cols[metric + "__test_mean"] = metric
    rename_plot_cols["normalized_" + metric + "__test_mean"] = (
        metric + " (Normalized 0-1)"
    )

rename_plot_cols["time__train_median"] = "Train time (s)"
rename_plot_cols["train_per_1000_inst_median_Accuracy"] = "Train time per 1000 inst."

results_df.rename(columns=rename_plot_cols, inplace=True)

PERFORMANCE_AXIS_OPTIIONS = list(rename_plot_cols.values())

# curate a list of dataset properties to plot
rename_metafeature_cols = {
    "f__pymfe.general.attr_to_inst": "Ratio: Features to Instances",
    "f__pymfe.general.cat_to_num": "Ratio: Cat. Features to Num. Features",
    "f__pymfe.general.freq_class.max": "Max. class frequency",
    "f__pymfe.general.freq_class.mean": "Mean class frequency",
    "f__pymfe.general.freq_class.median": "Median class frequency",
    "f__pymfe.general.freq_class.min": "Min class frequency",
    "f__pymfe.general.freq_class.range": "Range of class frequency",
    "f__pymfe.general.nr_attr": "Num. features",
    "f__pymfe.general.nr_bin": "Num. binary features",
    "f__pymfe.general.nr_cat": "Num. categorical features",
    "f__pymfe.general.nr_class": "Num. target classes",
    "f__pymfe.general.total_num_instances": "Num. instances",
    "f__pymfe.general.nr_num": "Num. numerical features",
}

metafeatures_df.rename(columns=rename_metafeature_cols, inplace=True)

DATAEST_AXIS_OPTIONS = list(rename_metafeature_cols.values())

# merge these axes into the results df
results_df = results_df.merge(
    metafeatures_df[["dataset_basename"] + DATAEST_AXIS_OPTIONS],
    left_on="dataset_name",
    right_on="dataset_basename",
    how="left",
)

#############################################################
# bokeh app
#############################################################

# dimensions

CONTROL_WIDTH = 600
CONTROL_HEIGHT = 200
TOTAL_HEIGHT = 700
HEADER_HEIGHT = 200

TABLE_HEIGHT = TOTAL_HEIGHT - CONTROL_HEIGHT - HEADER_HEIGHT
TOTAL_WIDTH = 1400


##########################################################
# data structures
##########################################################

##################
# dataset table

from bokeh.models import DataTable, TableColumn

# original data for datasets
dataset_source = ColumnDataSource(metafeatures_df)

##############
# scatter plot

# original source data
# initialze "selected" column to false - this column controls plotting
results_source = ColumnDataSource(results_df)

# data structure for selected results, for scatter plot. include position and plotting information
results_selected = ColumnDataSource(
    data=dict(
        dataset_name=[],
        alg_name=[],
        color=[],
        marker=[],
        x=[],
        y=[],
        # ll=[],
        # acc=[],
    ),
)

##########################################################


##########################################################
# display objects
##########################################################

# header: description of the app
header = Div(
    text=(Path(__file__).parent / "description.html").read_text("utf8"),
    width=CONTROL_WIDTH,
    height=HEADER_HEIGHT,
)

reczilla_img = Div(
    text=f"""
    <img src="https://github.com/naszilla/tabzilla/blob/main/img/tabzilla_logo.png?raw=true" width={CONTROL_WIDTH//1.5} class="img-center"/>
    """,
    margin=10,
    sizing_mode="stretch_width",
)


plot_control_header = Div(
    text="""
    <h2>1. Select Axes</h2>
    Use the controls below to select which datasets are included in the results.
    """,
    sizing_mode="stretch_width",
)

dataset_control_header = Div(
    text="""
    <h2>2. Select Datasets</h2>
    Select datasets to plot their results.
    """,
    sizing_mode="stretch_width",
)

# table for displaying all datasets (not just selected datasets)
# table for displaying datasets
dataset_display_columns = [
    TableColumn(field="dataset_basename", title="Dataset"),
    TableColumn(field="Num. instances", title="Size"),
    TableColumn(field="Num. features", title="# Feats"),
    TableColumn(field="Num. target classes", title="# Classes"),
]
dataset_table = DataTable(
    source=dataset_source,
    columns=dataset_display_columns,
    height=TABLE_HEIGHT,
    selectable="checkbox",
    index_position=None,
    # sizing_mode="stretch_both",
)

from bokeh.models import CustomJS

# selectors for x/y axis values
x_axis = Select(
    title="X Axis (dataset properties)",
    options=DATAEST_AXIS_OPTIONS,
    value="Num. instances",
)

y_axis = Select(
    title="Y Axis (performance)",
    options=PERFORMANCE_AXIS_OPTIIONS,
    value="Log Loss (Normalized 0-1)",
)


##########################################################
# display objects
##########################################################

TOOLTIPS = [
    ("Alg", "@alg_name"),
    ("Dataset", "@dataset_name"),
    ("x", "@x"),
    ("y", "@y"),
]

# note: the below works for adding stylesheets...
# from bokeh.models import InlineStyleSheet, Styles
# fig_stylesheet = InlineStyleSheet(css=":host { min-width: 400px; min-height: 400px}")

scatter_fig = figure(
    # height=TOTAL_HEIGHT - HEADER_HEIGHT,
    # width=600,
    title="",
    toolbar_location="above",
    tools="pan,wheel_zoom,box_zoom,reset",
    tooltips=TOOLTIPS,
    margin=20,
    width=TOTAL_WIDTH - CONTROL_WIDTH,
    height=TOTAL_HEIGHT - HEADER_HEIGHT,
    sizing_mode="stretch_both",
    # stylesheets=[fig_stylesheet], # use this to add a stylesheet (see above)
    min_height=400,
    min_width=400,
)

from bokeh.models import Scatter

# plot points
scatter_plot = scatter_fig.scatter(
    x="x",
    y="y",
    source=results_selected,
    size=10,
    fill_color="color",
    marker="marker",
    line_color="color",
    # fill_alpha="alpha",
    legend_field="alg_name",
)


# when the source is changed, or x or y axes change, update the plotting dataset

scatter_update_callback = CustomJS(
    args=dict(
        results_source=results_source,  # all results, from the original metadataset. column "selected" controls whether each row is plotted
        results_selected=results_selected,  # all results that will be plotted
        dataset_source=dataset_source,  # all datasets. those selected will be included in the selected results
        y_axis=y_axis,
        x_axis=x_axis,
        plot_x_axis=scatter_fig.xaxis,  # update the axes
        plot_y_axis=scatter_fig.yaxis,
        plot_view=scatter_plot.view,
    ),
    code="""

        // iterate through the entire source and select...
        // - the rows corresponding to a selected dataset
        // - the columns corresponding to the selected axis values
        
        const y_col = y_axis.value;
        const x_col = x_axis.value;

        // axes are returned as a list, so take the first one
        plot_x_axis[0].axis_label = x_col;
        plot_y_axis[0].axis_label = y_col; 

        const results_source_data = results_source.data;
        const results_select_data = results_selected.data;

        // reset the plotting data
        results_select_data['dataset_name'] = [];
        results_select_data['alg_name'] = [];
        results_select_data['color'] = [];
        results_select_data['marker'] = [];
        results_select_data['x'] = [];
        results_select_data['y'] = [];

        const selected_datasets = dataset_source.selected.indices;
        
        const selected_dataset_ids = [];
        for (let i = 0; i < selected_datasets.length; i++) {
            selected_dataset_ids.push(dataset_source.data['dataset_id'][selected_datasets[i]])
        }   

        // for each point in the source data, include the row if it is one of the selected points
        for (let i = 0; i < results_source_data['dataset_name'].length; i++) {
            if (selected_dataset_ids.includes(results_source_data['dataset_id'][i])) {
                results_select_data['dataset_name'].push(results_source_data['dataset_name'][i])
                results_select_data['alg_name'].push(results_source_data['alg_name'][i])
                results_select_data['color'].push(results_source_data['color'][i])
                results_select_data['marker'].push(results_source_data['marker'][i])
                results_select_data['x'].push(results_source_data[x_col][i])
                results_select_data['y'].push(results_source_data[y_col][i])
                // console.log('dataset: ' + results_select_data['dataset_name'][i] + '. alg: ' + results_select_data['alg_name'][i] + '. marker: ' + results_select_data['marker'][i] ) 
            }
        }
        results_selected.change.emit();
        plot_view.reset()
        // below from bokeh forums - this also works
        // document.querySelectorAll('.bk-tool-icon-reset[title="Reset"]').forEach(d => d.click())
    """,
)

# add callback to each UI element
y_axis.js_on_change("value", scatter_update_callback)
x_axis.js_on_change("value", scatter_update_callback)
dataset_source.selected.js_on_change("indices", scatter_update_callback)

####################
# define UI elements

xy_axes = row(
    y_axis,
    x_axis,
)
control_col = column(
    header,
    plot_control_header,
    xy_axes,
    dataset_control_header,
    dataset_table,
    width=CONTROL_WIDTH,
    height=TOTAL_HEIGHT,
)

plot_col = column(
    reczilla_img,
    scatter_fig,
    sizing_mode="stretch_both",
    height=TOTAL_HEIGHT,
)

layout = row(
    control_col,
    plot_col,
    height=TOTAL_HEIGHT,
    width=TOTAL_WIDTH,
    sizing_mode="stretch_both",
)

curdoc().add_root(layout)
curdoc().title = "TabZilla Results Browser"


from bokeh.plotting import save

# save the results to a file
save(layout, title="TabZilla Results Browser")
