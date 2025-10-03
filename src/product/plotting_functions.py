"""
Plotting functions
"""

import os
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path
import contextily as cx
import numpy as np


def create_hexbin_density_map(df, lat_col, lon_col, value_col=None, title="Geographic Density Map", 
                             gridsize=90, cmap="Reds", figsize=(12, 12)):
    """
    Create a hexbin density map for geographic data with NYC borough context.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data to plot
    lat_col : str
        Name of the latitude column
    lon_col : str
        Name of the longitude column
    value_col : str, optional
        Name of column to use for values (if None, uses count)
    title : str
        Title for the plot
    gridsize : int
        Size of hexagonal grid
    cmap : str
        Colormap name
    figsize : tuple
        Figure size (width, height)
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    
    # -------------------
    # 1) Load & prep NYC boundaries (TIGER block groups -> filter -> dissolve to boroughs)
    # -------------------
    bg_shapefile = os.path.abspath(os.path.join(os.getcwd(), "..", "src", "resources", "tl_2022_36_bg"))
    gdf_bg = gpd.read_file(bg_shapefile)

    # Ensure we only take NY State (36) and NYC counties
    gdf_bg = gdf_bg[gdf_bg["STATEFP"] == "36"]
    nyc_counties = {"005": "Bronx", "047": "Brooklyn", "061": "Manhattan", "081": "Queens", "085": "Staten Island"}
    gdf_nyc = gdf_bg[gdf_bg["COUNTYFP"].isin(nyc_counties.keys())].copy()

    # Dissolve block groups into one polygon per county for clean borders
    gdf_boros = gdf_nyc[["COUNTYFP", "geometry"]].dissolve(by="COUNTYFP", as_index=False)
    gdf_boros["borough"] = gdf_boros["COUNTYFP"].map(nyc_counties)

    # -------------------
    # 2) Prep complaint points & project everything to Web Mercator (EPSG:3857)
    # -------------------
    # Remove rows with missing lat/lon values
    df_plot = df.dropna(subset=[lat_col, lon_col]).copy()

    gdf_pts = gpd.GeoDataFrame(
        df_plot,
        geometry=gpd.points_from_xy(df_plot[lon_col], df_plot[lat_col]),
        crs="EPSG:4326",  # WGS84
    )

    # Project to 3857 so basemap/hexbin use same units
    gdf_boros_3857 = gdf_boros.to_crs(3857)
    gdf_pts_3857 = gdf_pts.to_crs(3857)

    # Optional: clip points to NYC (with a small buffer to keep edge points)
    nyc_union = gdf_boros_3857.union_all()
    gdf_pts_3857 = gdf_pts_3857[gdf_pts_3857.within(nyc_union.buffer(2000))]

    # -------------------
    # 3) Build a nicer hexbin map
    # -------------------
    fig, ax = plt.subplots(figsize=figsize)

    # Set extent to NYC bounds for consistent hex size & color scaling
    xmin, ymin, xmax, ymax = gdf_boros_3857.total_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    
    # Add a Google Maps-style basemap with roads and labels (after setting extent)
    cx.add_basemap(ax, crs="EPSG:3857", source=cx.providers.CartoDB.Voyager)

    # Hexbin in projected meters
    x = gdf_pts_3857.geometry.x.values
    y = gdf_pts_3857.geometry.y.values
    
    # Use values if provided, otherwise use count
    if value_col is not None and value_col in df_plot.columns:
        C = gdf_pts_3857[value_col].values
        reduce_C_function = np.sum  # sum the values in each bin
    else:
        C = None
        reduce_C_function = None

    hb = ax.hexbin(
        x, y,
        C=C,
        reduce_C_function=reduce_C_function,
        gridsize=gridsize,
        extent=(xmin, xmax, ymin, ymax),
        cmap=cmap,
        mincnt=1,               # show all bins including those with 0 count
        linewidths=0
    )
    
    # Clip hexbin to NYC borough boundaries only
    # Convert the NYC polygon to a matplotlib Path and apply as clip
    if nyc_union.geom_type == 'Polygon':
        # Single polygon - use exterior coordinates
        clip_path = Path(list(nyc_union.exterior.coords))
    else:
        # MultiPolygon - create compound path from all polygon exteriors
        clip_path = Path.make_compound_path(*[
            Path(list(geom.exterior.coords)) for geom in nyc_union.geoms
        ])
    
    clip_patch = PathPatch(clip_path, transform=ax.transData, facecolor='none', edgecolor='none')
    ax.add_patch(clip_patch)
    hb.set_clip_path(clip_patch)


    # Borough labels at dissolved centroids
    label_gdf = gdf_boros_3857.copy()
    label_gdf["centroid"] = label_gdf.geometry.representative_point()  # safer than centroid for oddly shaped polygons
    for _, r in label_gdf.iterrows():
        ax.annotate(
            r["borough"],
            xy=(r["centroid"].x, r["centroid"].y),
            ha="center", va="center",
            fontsize=8, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="lightblue", ec="none", alpha=0.6)
        )

    # Colorbar with better label
    cbar = plt.colorbar(hb, ax=ax, fraction=0.035, pad=0.02)
    if value_col is not None:
        cbar.set_label(f"Sum of {value_col}")
    else:
        cbar.set_label("Count")

    # Title & tidy styling
    ax.set_title(title, fontsize=14)
    ax.set_axis_off()  # hides ticks/frames; cleaner with basemap

    plt.tight_layout()
    return fig, ax