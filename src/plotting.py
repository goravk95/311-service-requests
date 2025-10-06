"""
Plotting functions for NYC 311 data visualization.
"""

import os
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path
import contextily as cx
import numpy as np
import pandas as pd
import h3
from shapely.geometry import Polygon
from datetime import datetime, timedelta


def create_hexbin_density_map(
    df,
    lat_col,
    lon_col,
    value_col=None,
    title="Geographic Density Map",
    gridsize=90,
    cmap="Reds",
    figsize=(12, 12),
):
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
    bg_shapefile = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "resources", "tl_2022_36_bg")
    )
    gdf_bg = gpd.read_file(bg_shapefile)

    # Ensure we only take NY State (36) and NYC counties
    gdf_bg = gdf_bg[gdf_bg["STATEFP"] == "36"]
    nyc_counties = {
        "005": "Bronx",
        "047": "Brooklyn",
        "061": "Manhattan",
        "081": "Queens",
        "085": "Staten Island",
    }
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
        x,
        y,
        C=C,
        reduce_C_function=reduce_C_function,
        gridsize=gridsize,
        extent=(xmin, xmax, ymin, ymax),
        cmap=cmap,
        mincnt=1,  # show all bins including those with 0 count
        linewidths=0,
    )

    # Clip hexbin to NYC borough boundaries only
    # Convert the NYC polygon to a matplotlib Path and apply as clip
    if nyc_union.geom_type == "Polygon":
        # Single polygon - use exterior coordinates
        clip_path = Path(list(nyc_union.exterior.coords))
    else:
        # MultiPolygon - create compound path from all polygon exteriors
        clip_path = Path.make_compound_path(
            *[Path(list(geom.exterior.coords)) for geom in nyc_union.geoms]
        )

    clip_patch = PathPatch(clip_path, transform=ax.transData, facecolor="none", edgecolor="none")
    ax.add_patch(clip_patch)
    hb.set_clip_path(clip_patch)

    # Borough labels at dissolved centroids
    label_gdf = gdf_boros_3857.copy()
    label_gdf["centroid"] = (
        label_gdf.geometry.representative_point()
    )  # safer than centroid for oddly shaped polygons
    for _, r in label_gdf.iterrows():
        ax.annotate(
            r["borough"],
            xy=(r["centroid"].x, r["centroid"].y),
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="lightblue", ec="none", alpha=0.6),
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


def plot_h3_counts_for_week(
    df,
    week,
    complaint_family,
    value_col='count',
    title=None,
    cmap="YlOrRd",
    figsize=(14, 14),
    edge_color="white",
    edge_width=0.5,
    alpha=0.7,
):
    """
    Plot H3 hexagon counts for a specific week and complaint type at resolution 8.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data to plot with 'week', 'complaint_family', and 'hex8' columns
    week : str or datetime
        Start date of the week to plot (e.g., '2024-01-01' or datetime object)
    complaint_family : str
        Complaint family to filter for (e.g., 'food_safety', 'vector_control')
    value_col : str
        Name of column to use for values (default: 'count')
    title : str, optional
        Title for the plot. If None, auto-generates based on parameters
    cmap : str
        Colormap name (default: 'YlOrRd')
    figsize : tuple
        Figure size (width, height) (default: (14, 14))
    edge_color : str
        Color of hexagon edges (default: 'white')
    edge_width : float
        Width of hexagon edges (default: 0.5)
    alpha : float
        Transparency of hexagons (default: 0.7)
    
    Returns:
    --------
    fig : matplotlib figure object
    ax : matplotlib axes object
    counts_df : pandas.DataFrame
        DataFrame with hexagon IDs and counts
    
    Examples:
    ---------
    fig, ax, counts = plot_h3_counts_for_week(
        df, '2024-01-01', 'food_safety', value_col='y'
    )
    """
    df_filtered = df[(df['week'] == week) & (df['complaint_family'] == complaint_family)]
    df_filtered['h3_cell'] = df_filtered['hex8']
    final_resolution = 8
    
    # Count records per hexagon
    counts_df = df_filtered.groupby('h3_cell')[value_col].sum().reset_index(name='count')
    
    # Convert H3 cells to polygons
    def h3_to_polygon(h3_cell):
        """Convert H3 cell to shapely Polygon"""
        boundary = h3.cell_to_boundary(h3_cell)
        # boundary is a list of (lat, lng) tuples - need to convert to (lng, lat) for shapely
        coords = [(lng, lat) for lat, lng in boundary]
        return Polygon(coords)
    
    counts_df['geometry'] = counts_df['h3_cell'].apply(h3_to_polygon)
    
    # Create GeoDataFrame
    gdf_hexagons = gpd.GeoDataFrame(
        counts_df,
        geometry='geometry',
        crs='EPSG:4326'
    )
    
    # -------------------
    # Load NYC boundaries for context
    # -------------------
    bg_shapefile = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "resources", "tl_2022_36_bg")
    )
    gdf_bg = gpd.read_file(bg_shapefile)
    
    gdf_bg = gdf_bg[gdf_bg["STATEFP"] == "36"]
    nyc_counties = {
        "005": "Bronx",
        "047": "Brooklyn",
        "061": "Manhattan",
        "081": "Queens",
        "085": "Staten Island",
    }
    gdf_nyc = gdf_bg[gdf_bg["COUNTYFP"].isin(nyc_counties.keys())].copy()
    gdf_boros = gdf_nyc[["COUNTYFP", "geometry"]].dissolve(by="COUNTYFP", as_index=False)
    gdf_boros["borough"] = gdf_boros["COUNTYFP"].map(nyc_counties)
    
    # Project to Web Mercator for basemap
    gdf_boros_3857 = gdf_boros.to_crs(3857)
    gdf_hexagons_3857 = gdf_hexagons.to_crs(3857)
    
    # -------------------
    # Create plot
    # -------------------
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set extent to NYC bounds
    xmin, ymin, xmax, ymax = gdf_boros_3857.total_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    
    # Add basemap
    cx.add_basemap(ax, crs="EPSG:3857", source=cx.providers.CartoDB.Voyager)
    
    # Plot hexagons with color based on count
    gdf_hexagons_3857.plot(
        ax=ax,
        column='count',
        cmap=cmap,
        edgecolor=edge_color,
        linewidth=edge_width,
        alpha=alpha,
        legend=True,
        legend_kwds={
            'label': 'Count',
            'orientation': 'vertical',
            'shrink': 0.8,
            'pad': 0.02
        }
    )
    
    # Clip to NYC boundaries
    nyc_union = gdf_boros_3857.union_all()
    if nyc_union.geom_type == "Polygon":
        clip_path = Path(list(nyc_union.exterior.coords))
    else:
        clip_path = Path.make_compound_path(
            *[Path(list(geom.exterior.coords)) for geom in nyc_union.geoms]
        )
    
    clip_patch = PathPatch(clip_path, transform=ax.transData, facecolor="none", edgecolor="none")
    ax.add_patch(clip_patch)
    
    # Add borough labels
    label_gdf = gdf_boros_3857.copy()
    label_gdf["centroid"] = label_gdf.geometry.representative_point()
    for _, r in label_gdf.iterrows():
        ax.annotate(
            r["borough"],
            xy=(r["centroid"].x, r["centroid"].y),
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", ec="none", alpha=0.7),
        )
    
    # Generate title if not provided
    if title is None:
        title = f"311 Service Requests - {week}"
        title += f"\n{complaint_family}"
        title += f" (H3 Resolution {final_resolution})"
    
    ax.set_title(title, fontsize=10, fontweight='bold', pad=20)
    ax.set_axis_off()
    
    # Add summary stats as text
    total_count = counts_df['count'].sum()
    max_count = counts_df['count'].max()
    n_hexagons = len(counts_df)
    
    stats_text = f"Total Requests: {total_count:,}\n"
    stats_text += f"Max per Hexagon: {max_count:,}\n"
    stats_text += f"Active Hexagons: {n_hexagons:,}"
    
    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    plt.tight_layout()
    
    return fig, ax, counts_df
