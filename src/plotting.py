"""Plotting functions for NYC 311 data visualization.

This module provides functions for creating geographic visualizations of NYC 311 service
request data, including hexbin density maps and H3 hexagon-based choropleth maps.
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


def create_hexbin_density_map(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    value_col: str | None = None,
    title: str = "Geographic Density Map",
    gridsize: int = 90,
    cmap: str = "Reds",
    figsize: tuple[int, int] = (12, 12),
) -> tuple:
    """Create a hexbin density map for geographic data with NYC borough context.

    Args:
        df: DataFrame containing the data to plot.
        lat_col: Name of the latitude column.
        lon_col: Name of the longitude column.
        value_col: Name of column to use for values (if None, uses count).
        title: Title for the plot.
        gridsize: Size of hexagonal grid.
        cmap: Colormap name.
        figsize: Figure size (width, height).

    Returns:
        Tuple of matplotlib figure and axes objects.
    """
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
    df_plot = df.dropna(subset=[lat_col, lon_col]).copy()

    gdf_pts = gpd.GeoDataFrame(
        df_plot,
        geometry=gpd.points_from_xy(df_plot[lon_col], df_plot[lat_col]),
        crs="EPSG:4326",
    )

    gdf_boros_3857 = gdf_boros.to_crs(3857)
    gdf_pts_3857 = gdf_pts.to_crs(3857)

    nyc_union = gdf_boros_3857.union_all()
    gdf_pts_3857 = gdf_pts_3857[gdf_pts_3857.within(nyc_union.buffer(2000))]
    fig, ax = plt.subplots(figsize=figsize)

    xmin, ymin, xmax, ymax = gdf_boros_3857.total_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    cx.add_basemap(ax, crs="EPSG:3857", source=cx.providers.CartoDB.Voyager)

    x = gdf_pts_3857.geometry.x.values
    y = gdf_pts_3857.geometry.y.values
    if value_col is not None and value_col in df_plot.columns:
        C = gdf_pts_3857[value_col].values
        reduce_C_function = np.sum
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
        mincnt=1,
        linewidths=0,
    )

    if nyc_union.geom_type == "Polygon":
        clip_path = Path(list(nyc_union.exterior.coords))
    else:
        clip_path = Path.make_compound_path(
            *[Path(list(geom.exterior.coords)) for geom in nyc_union.geoms]
        )

    clip_patch = PathPatch(clip_path, transform=ax.transData, facecolor="none", edgecolor="none")
    ax.add_patch(clip_patch)
    hb.set_clip_path(clip_patch)

    label_gdf = gdf_boros_3857.copy()
    label_gdf["centroid"] = label_gdf.geometry.representative_point()
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

    cbar = plt.colorbar(hb, ax=ax, fraction=0.035, pad=0.02)
    if value_col is not None:
        cbar.set_label(f"Sum of {value_col}")
    else:
        cbar.set_label("Count")

    ax.set_title(title, fontsize=14)
    ax.set_axis_off()

    plt.tight_layout()
    return fig, ax


def plot_h3_counts_for_week(
    df: pd.DataFrame,
    week: str,
    complaint_family: str,
    value_col: str = 'count',
    title: str | None = None,
    cmap: str = "YlOrRd",
    figsize: tuple[int, int] = (14, 14),
    edge_color: str = "white",
    edge_width: float = 0.5,
    alpha: float = 0.7,
) -> tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """Plot H3 hexagon counts for a specific week and complaint type.

    Args:
        df: DataFrame containing the data to plot.
        week: Start date of the week to plot.
        complaint_family: Complaint family to filter for.
        value_col: Name of column to use for values.
        title: Title for the plot. If None, auto-generates.
        cmap: Colormap name.
        figsize: Figure size (width, height).
        edge_color: Color of hexagon edges.
        edge_width: Width of hexagon edges.
        alpha: Transparency of hexagons.

    Returns:
        Tuple of (figure, axes, counts DataFrame).
    """
    df_filtered = df[(df['week'] == week) & (df['complaint_family'] == complaint_family)]
    df_filtered['h3_cell'] = df_filtered['hex8']
    final_resolution = 8
    
    counts_df = df_filtered.groupby('h3_cell')[value_col].sum().reset_index(name='count')
    
    def h3_to_polygon(h3_cell):
        boundary = h3.cell_to_boundary(h3_cell)
        coords = [(lng, lat) for lat, lng in boundary]
        return Polygon(coords)
    
    counts_df['geometry'] = counts_df['h3_cell'].apply(h3_to_polygon)
    gdf_hexagons = gpd.GeoDataFrame(
        counts_df,
        geometry='geometry',
        crs='EPSG:4326'
    )
    
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
    
    gdf_boros_3857 = gdf_boros.to_crs(3857)
    gdf_hexagons_3857 = gdf_hexagons.to_crs(3857)
    fig, ax = plt.subplots(figsize=figsize)
    
    xmin, ymin, xmax, ymax = gdf_boros_3857.total_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    
    cx.add_basemap(ax, crs="EPSG:3857", source=cx.providers.CartoDB.Voyager)
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
    
    nyc_union = gdf_boros_3857.union_all()
    if nyc_union.geom_type == "Polygon":
        clip_path = Path(list(nyc_union.exterior.coords))
    else:
        clip_path = Path.make_compound_path(
            *[Path(list(geom.exterior.coords)) for geom in nyc_union.geoms]
        )
    
    clip_patch = PathPatch(clip_path, transform=ax.transData, facecolor="none", edgecolor="none")
    ax.add_patch(clip_patch)
    
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

    if title is None:
        title = f"311 Service Requests - {week}"
        title += f"\n{complaint_family}"
        title += f" (H3 Resolution {final_resolution})"
    
    ax.set_title(title, fontsize=10, fontweight='bold', pad=20)
    ax.set_axis_off()
    
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
