"""NYC 311 Service Requests Data Loader.

This module provides asynchronous functionality to fetch NYC 311 service request data
from the Socrata Open Data API and save it as partitioned Parquet files.
Also fetches ACS census population data and NOAA weather data.
"""

import os
import asyncio
import gc
from datetime import datetime
import pandas as pd
from sodapy import Socrata
import censusdata

from . import config


def _create_socrata_client(
    api_endpoint: str, app_token: str, username: str, password: str
) -> Socrata:
    """Create and configure a Socrata API client.

    Args:
        api_endpoint: The Socrata API endpoint.
        app_token: Application token for API authentication.
        username: API username/key ID.
        password: API password/key secret.

    Returns:
        Configured Socrata client instance.
    """
    client = Socrata(
        api_endpoint, app_token=app_token, username=username, password=password, timeout=600
    )
    return client


def _convert_dataframe_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convert DataFrame column types to match the schema.

    Args:
        df: The DataFrame with raw string data from Socrata API.

    Returns:
        DataFrame with properly typed columns.
    """
    timestamp_cols = ["created_date", "closed_date", "due_date", "resolution_action_updated_date"]
    for col in timestamp_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    float_cols = ["x_coordinate_state_plane", "y_coordinate_state_plane", "latitude", "longitude"]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    special_cols = ["location"]
    for col in special_cols:
        if col in df.columns:
            df[col] = df[col].astype("string")

    return df


def _add_missing_schema_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add missing columns from schema as NA values.

    Args:
        df: The DataFrame to add missing columns to.

    Returns:
        DataFrame with all schema columns present.
    """
    for col in config.SCHEMA.names:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def _process_dataframe(df: pd.DataFrame, use_full_schema: bool = True) -> pd.DataFrame:
    """Process the DataFrame to match the schema.

    Args:
        df: The DataFrame to process.
        use_full_schema: If True, add missing columns from schema and enforce full schema.

    Returns:
        Processed DataFrame.
    """
    df = _convert_dataframe_types(df)
    if use_full_schema:
        df = _add_missing_schema_columns(df)
    return df


def _fetch_data_for_month(
    year: int,
    month: int,
    save: bool = True,
    columns: list[str] | None = None,
    additional_filter: str | None = None,
) -> None:
    """Fetch and save 311 service request data for a specific month.

    Args:
        year: The year to fetch data for.
        month: The month to fetch data for (1-12).
        save: Whether to save the data to disk.
        columns: List of column names to select. If None, selects all columns.
        additional_filter: Additional WHERE clause filter.
    """
    client = _create_socrata_client(
        api_endpoint=config.API_ENDPOINT,
        app_token=config.APP_TOKEN,
        username=config.API_KEY_ID,
        password=config.API_KEY_SECRET,
    )

    start = f"{year}-{month:02d}-01T00:00:00"
    if month == 12:
        end = f"{year+1}-01-01T00:00:00"
    else:
        end = f"{year}-{month+1:02d}-01T00:00:00"

    where_clause = f"{config.DATE_COLUMN} >= '{start}' AND {config.DATE_COLUMN} < '{end}'"
    if additional_filter:
        where_clause = f"{where_clause} AND {additional_filter}"

    print(f"Fetching {year}-{month:02d} ...")
    if columns:
        results = list(
            client.get_all(config.DATASET_ID, where=where_clause, select=",".join(columns))
        )
    else:
        results = list(client.get_all(config.DATASET_ID, where=where_clause))

    if results:
        df = pd.DataFrame.from_records(results)

        use_full_schema = columns is None

        df = _process_dataframe(df, use_full_schema=use_full_schema)

        if save:
            file_path = (
                config.LANDING_DATA_PATH + f"/year={year}/month={month:02d}/part-0000.parquet"
            )
            if use_full_schema:
                df.to_parquet(file_path, index=False, schema=config.SCHEMA)
            else:
                df.to_parquet(file_path, index=False)
            print(f"Saved {file_path} ({len(df):,} rows, {len(df.columns)} columns)")
        else:
            print(
                f"Fetched {year}-{month:02d} ({len(df):,} rows, {len(df.columns)} columns) - not saved"
            )

        del df
        del results
        gc.collect()
    else:
        print(f"No data for {year}-{month:02d}")


async def fetch_and_save_month(
    year: int,
    month: int,
    sem: asyncio.Semaphore,
    save: bool = True,
    columns: list[str] | None = None,
    additional_filter: str | None = None,
) -> None:
    """Asynchronously fetch and save data for a specific month with concurrency control.

    Args:
        year: The year to fetch data for.
        month: The month to fetch data for (1-12).
        sem: Semaphore to limit concurrent API requests.
        save: Whether to save the data to disk.
        columns: List of column names to select. If None, selects all columns.
        additional_filter: Additional WHERE clause filter.
    """
    async with sem:
        await asyncio.to_thread(
            _fetch_data_for_month, year, month, save, columns, additional_filter
        )


async def fetch_all_service_requests(
    save: bool = True, columns: list[str] | None = None, additional_filter: str | None = None
) -> None:
    """Fetch all NYC 311 service request data across the configured date range.

    Args:
        save: Whether to save the data to disk.
        columns: List of column names to select. If None, selects all columns.
        additional_filter: Additional WHERE clause filter.
    """
    sem = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS)

    years = range(config.DATA_START_YEAR, config.DATA_END_YEAR)
    months = range(1, 13)
    tasks = [
        fetch_and_save_month(y, m, sem, save, columns, additional_filter)
        for y in years
        for m in months
    ]
    await asyncio.gather(*tasks)


async def fetch_current_month_service_requests(
    save: bool = True, columns: list[str] | None = None, additional_filter: str | None = None
) -> None:
    """Fetch and overwrite data for the current month only.

    Args:
        save: Whether to save the data to disk.
        columns: List of column names to select. If None, selects all columns.
        additional_filter: Additional WHERE clause filter.
    """
    now = datetime.now()
    current_year = now.year
    current_month = now.month

    sem = asyncio.Semaphore(1)

    print(f"Fetching current month: {current_year}-{current_month:02d}")
    await fetch_and_save_month(current_year, current_month, sem, save, columns, additional_filter)
    print("Current month data updated successfully!")


def fetch_acs_census_population_data(
    start_year: int = 2013, end_year: int = 2023, save: bool = True
) -> pd.DataFrame:
    """Fetch ACS 5-year population data for NYC block groups.

    Args:
        start_year: First year to fetch data for.
        end_year: Last year to fetch data for (inclusive).
        save: Whether to save the data to disk.

    Returns:
        DataFrame with columns ['GEOID', 'population', 'year'].
    """
    nyc_counties_list = ["005", "047", "061", "081", "085"]
    acs_var = ["B01003_001E"]
    years = list(range(start_year, end_year + 1))

    all_data = []

    for year in years:
        print(f"Downloading ACS 5-year data for {year}...")
        year_data = []

        for county in nyc_counties_list:
            try:
                print(f"  - County {county}...")
                data = censusdata.download(
                    src="acs5",
                    year=year,
                    geo=censusdata.censusgeo(
                        [("state", "36"), ("county", county), ("block group", "*")]
                    ),
                    var=acs_var,
                )
                year_data.append(data)
            except Exception as e:
                print(f"    Error downloading county {county} for year {year}: {e}")
                continue

        if year_data:
            combined = pd.concat(year_data)
            combined = combined.reset_index()
            combined["year"] = year
            all_data.append(combined)

    df_pop = pd.concat(all_data, ignore_index=True)

    def geo_to_geoid(geo):
        params = geo.params()
        return params[0][1] + params[1][1] + params[2][1] + params[3][1]

    df_pop["GEOID"] = df_pop["index"].apply(geo_to_geoid)
    df_pop.rename(columns={"B01003_001E": "population"}, inplace=True)
    df_pop = df_pop[["GEOID", "population", "year"]]

    if save:
        output_path = config.DATA_DIR / "acs-population" / "combined_population_data.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_pop.to_csv(output_path, index=False)
        print(f"Saved ACS census data to {output_path}")

    return df_pop


def fetch_noaa_weather_data(
    start_year: int = 2010, end_year: int = 2025, save: bool = True
) -> pd.DataFrame:
    """Fetch NOAA NCLIMGRID daily weather data for NYC counties.

    Args:
        start_year: First year to fetch data for.
        end_year: Last year to fetch data for (inclusive).
        save: Whether to save the data to disk.

    Returns:
        DataFrame with weather data columns.
    """
    nyc_fips = ["36005", "36047", "36061", "36081", "36085"]
    years = range(start_year, end_year + 1)
    months = range(1, 13)

    base_url = "https://noaa-nclimgrid-daily-pds.s3.amazonaws.com/EpiNOAA/v1-0-0/parquet/cty/YEAR={year}/STATUS=scaled/{yyyymm}.parquet"

    all_dfs = []

    for year in years:
        for month in months:
            yyyymm = f"{year}{month:02d}"
            url = base_url.format(year=year, yyyymm=yyyymm)

            try:
                df = pd.read_parquet(url)
                df_nyc = df[df["fips"].isin(nyc_fips)]
                df_nyc = df_nyc[["date", "fips", "tmax", "tmin", "tavg", "prcp"]]
                all_dfs.append(df_nyc)
                print(f"Loaded {yyyymm}, rows: {len(df_nyc)}")
            except Exception as e:
                print(f"Error loading {url}: {e}")
                continue

    df_weather = pd.concat(all_dfs, ignore_index=True)
    for col in ["tmax", "tmin", "tavg", "prcp"]:
        df_weather[col] = df_weather[col].astype(float)

    df_weather["date"] = pd.to_datetime(df_weather["date"])
    df_weather["year"] = df_weather["date"].dt.year
    df_weather["month"] = df_weather["date"].dt.month
    df_weather["day"] = df_weather["date"].dt.day

    if save:
        output_path = config.DATA_DIR / "noaa-nclimgrid-daily" / "nyc_fips_weather_data.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_weather.to_csv(output_path, index=False)
        print(f"Saved weather data to {output_path}")

    return df_weather
