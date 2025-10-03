"""NYC 311 Service Requests Data Loader.

This module provides asynchronous functionality to fetch NYC 311 service request data
from the Socrata Open Data API and save it as partitioned Parquet files.

The data is organized by year and month partitions for efficient querying and storage.
Concurrent requests are controlled to respect API rate limits.
"""

import os
import asyncio
import gc
from datetime import datetime
import pandas as pd
from sodapy import Socrata
import censusdata

from .config import (
    API_ENDPOINT,
    APP_TOKEN,
    API_KEY_ID,
    API_KEY_SECRET,
    DATASET_ID,
    DATE_COLUMN,
    S3_OUTPUT_DIR,
    LOCAL_OUTPUT_DIR,
    DATA_START_YEAR,
    DATA_END_YEAR,
    MAX_CONCURRENT_REQUESTS,
    SCHEMA,
)


def _create_socrata_client(api_endpoint, app_token, username, password):
    """Create and configure a Socrata API client.

    Args:
        api_endpoint (str): The Socrata API endpoint (e.g., "data.cityofnewyork.us").
        app_token (str): Application token for API authentication.
        username (str): API username/key ID.
        password (str): API password/key secret.

    Returns:
        Socrata: Configured Socrata client instance.
    """
    client = Socrata(
        api_endpoint, app_token=app_token, username=username, password=password, timeout=600
    )
    return client


def _convert_dataframe_types(df):
    """Convert DataFrame column types to match the schema.

    Args:
        df (pd.DataFrame): The DataFrame with raw string data from Socrata API.

    Returns:
        pd.DataFrame: DataFrame with properly typed columns.
    """
    # Convert timestamp columns
    timestamp_cols = ["created_date", "closed_date", "due_date", "resolution_action_updated_date"]
    for col in timestamp_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Convert float columns
    float_cols = ["x_coordinate_state_plane", "y_coordinate_state_plane", "latitude", "longitude"]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    special_cols = ["location"]
    for col in special_cols:
        if col in df.columns:
            df[col] = df[col].astype("string")

    return df


def _add_missing_schema_columns(df):
    """Add missing columns from schema as NA values.

    Args:
        df (pd.DataFrame): The DataFrame to add missing columns to.

    Returns:
        pd.DataFrame: DataFrame with all schema columns present.
    """
    for col in SCHEMA.names:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def _process_dataframe(df):
    """Process the DataFrame to match the schema."""
    df = _convert_dataframe_types(df)
    df = _add_missing_schema_columns(df)
    return df


def _fetch_data_for_month(year: int, month: int, save: bool = True):
    """Fetch and save 311 service request data for a specific month.

    This function runs synchronously in a worker thread. It creates its own
    Socrata client to avoid sharing connections across threads.

    Args:
        year (int): The year to fetch data for.
        month (int): The month to fetch data for (1-12).
        save (bool): Whether to save the data to disk. Defaults to True.

    Notes:
        - Data is saved as Parquet files in year=YYYY/month=MM partition structure.
        - Empty results are logged but no file is created.
        - Memory is explicitly freed after processing each month.
    """
    # Create per-thread client to avoid connection sharing issues
    client = _create_socrata_client(
        api_endpoint=API_ENDPOINT, app_token=APP_TOKEN, username=API_KEY_ID, password=API_KEY_SECRET
    )

    start = f"{year}-{month:02d}-01T00:00:00"
    if month == 12:
        end = f"{year+1}-01-01T00:00:00"
    else:
        end = f"{year}-{month+1:02d}-01T00:00:00"

    where_clause = f"{DATE_COLUMN} >= '{start}' AND {DATE_COLUMN} < '{end}'"
    print(f"Fetching {year}-{month:02d} ...")

    # client.get_all handles paging internally (blocking)
    results = list(client.get_all(DATASET_ID, where=where_clause))

    if results:
        df = pd.DataFrame.from_records(results)

        # Convert types to match schema
        df = _process_dataframe(df)

        if save:
            # Write to year/month partition
            file_path = os.path.join(
                S3_OUTPUT_DIR, f"year={year}/month={month:02d}/part-0000.parquet"
            )
            df.to_parquet(file_path, index=False, schema=SCHEMA)
            print(f"Saved {file_path} ({len(df):,} rows)")
        else:
            print(f"Fetched {year}-{month:02d} ({len(df):,} rows) - not saved")

        # Clean up memory
        del df
        del results
        gc.collect()
    else:
        print(f"No data for {year}-{month:02d}")


async def fetch_and_save_month(year: int, month: int, sem: asyncio.Semaphore, save: bool = True):
    """Asynchronously fetch and save data for a specific month with concurrency control.

    Args:
        year (int): The year to fetch data for.
        month (int): The month to fetch data for (1-12).
        sem (asyncio.Semaphore): Semaphore to limit concurrent API requests.
        save (bool): Whether to save the data to disk. Defaults to True.

    Notes:
        The semaphore ensures we don't exceed MAX_CONCURRENT_REQUESTS simultaneous
        API calls in case of RAM limitations.
    """
    async with sem:
        # Run the synchronous worker in a thread pool
        await asyncio.to_thread(_fetch_data_for_month, year, month, save)


async def fetch_all_service_requests(save: bool = True):
    """Fetch all NYC 311 service request data across the configured date range.

    This function orchestrates the parallel download of data for all months
    in the configured year range (DATA_START_YEAR to DATA_END_YEAR).

    Args:
        save (bool): Whether to save the data to disk. Defaults to True.

    Concurrency is controlled by MAX_CONCURRENT_REQUESTS to respect API limits.
    Data is saved as partitioned Parquet files organized by year and month.

    Example:
        >>> import asyncio
        >>> asyncio.run(fetch_all_service_requests())

    Raises:
        Any exceptions from individual month fetches will propagate up.
    """
    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    years = range(DATA_START_YEAR, DATA_END_YEAR)
    months = range(1, 13)
    tasks = [fetch_and_save_month(y, m, sem, save) for y in years for m in months]
    await asyncio.gather(*tasks)


async def fetch_current_month_service_requests(save: bool = True):
    """Fetch and overwrite data for the current month only.

    This function fetches NYC 311 service request data for the current month
    and overwrites any existing data in the data directory. This is useful for
    keeping the current month's data up-to-date without re-fetching all historical data.

    Args:
        save (bool): Whether to save the data to disk. Defaults to True.

    The function automatically determines the current year and month from the system clock.

    Example:
        >>> import asyncio
        >>> asyncio.run(fetch_current_month())

    Notes:
        - The existing file for the current month (if any) will be overwritten.
        - Only requires a single API request, so it's much faster than fetch_all_service_requests().
    """
    now = datetime.now()
    current_year = now.year
    current_month = now.month

    # Create a semaphore with limit of 1 since we're only fetching one month
    sem = asyncio.Semaphore(1)

    print(f"Fetching current month: {current_year}-{current_month:02d}")
    await fetch_and_save_month(current_year, current_month, sem, save)
    print("Current month data updated successfully!")


def fetch_acs_census_population_data(
    start_year: int = 2013, end_year: int = 2023, save: bool = True
):
    """Fetch ACS 5-year population data for NYC block groups.

    This function downloads American Community Survey (ACS) 5-year estimates for
    total population (variable B01003_001E) for all block groups in NYC's five counties.
    Block group data is only available from 2012+ in ACS 5-year estimates.

    Args:
        start_year (int): First year to fetch data for. Defaults to 2013. Code won't work prior to 2013.
        end_year (int): Last year to fetch data for (inclusive). Defaults to 2023. Data only available up to 2023.
        save (bool): Whether to save the data to disk. Defaults to True.

    Returns:
        pd.DataFrame: DataFrame with columns ['GEOID', 'population', 'year']

    Notes:
        - NYC counties: 005 (Bronx), 047 (Kings/Brooklyn), 061 (Manhattan),
          081 (Queens), 085 (Richmond/Staten Island)
        - Data is saved to LOCAL_OUTPUT_DIR/acs-population/combined_population_data.csv

    Example:
        >>> df_pop = fetch_acs_census_population_data(start_year=2013, end_year=2023)
    """
    # NYC county FIPS codes
    nyc_counties_list = [
        "005",
        "047",
        "061",
        "081",
        "085",
    ]  # Bronx, Kings, Manhattan, Queens, Richmond

    # ACS variable for total population
    acs_var = ["B01003_001E"]

    # Years to download
    years = list(range(start_year, end_year + 1))

    all_data = []

    # Download data for each year and county
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

    # Combine all years
    df_pop = pd.concat(all_data, ignore_index=True)

    # Parse GEOID from censusgeo index
    def geo_to_geoid(geo):
        params = geo.params()  # params is a method that returns the geographic components
        return params[0][1] + params[1][1] + params[2][1] + params[3][1]

    df_pop["GEOID"] = df_pop["index"].apply(geo_to_geoid)
    df_pop.rename(columns={"B01003_001E": "population"}, inplace=True)
    df_pop = df_pop[["GEOID", "population", "year"]]

    if save:
        output_path = os.path.join(
            LOCAL_OUTPUT_DIR, "acs-population", "combined_population_data.csv"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_pop.to_csv(output_path, index=False)
        print(f"Saved ACS census data to {output_path}")

    return df_pop


def fetch_noaa_weather_data(start_year: int = 2010, end_year: int = 2025, save: bool = True):
    """Fetch NOAA NCLIMGRID daily weather data for NYC counties.

    This function downloads daily weather data (temperature and precipitation) for
    NYC's five counties from the NOAA NCLIMGRID dataset hosted on AWS S3.

    Args:
        start_year (int): First year to fetch data for. Defaults to 2010.
        end_year (int): Last year to fetch data for (inclusive). Defaults to 2025.
        save (bool): Whether to save the data to disk. Defaults to True.

    Returns:
        pd.DataFrame: DataFrame with columns ['date', 'fips', 'tmax', 'tmin', 'tavg',
                      'prcp', 'year', 'month', 'day']

    Notes:
        - NYC county FIPS: 36005 (Bronx), 36047 (Kings), 36061 (Manhattan),
          36081 (Queens), 36085 (Richmond)
        - Data source: https://noaa-nclimgrid-daily-pds.s3.amazonaws.com/
        - Data is saved to LOCAL_OUTPUT_DIR/noaa-nclimgrid-daily/nyc_fips_weather_data.csv
        - Temperature units: Celsius
        - Precipitation units: millimeters

    Example:
        >>> df_weather = fetch_noaa_weather_data(start_year=2010, end_year=2023)
    """
    # NYC county FIPS codes
    nyc_fips = ["36005", "36047", "36061", "36081", "36085"]

    # Years and months to loop over
    years = range(start_year, end_year + 1)
    months = range(1, 13)

    # Base S3 URL
    base_url = "https://noaa-nclimgrid-daily-pds.s3.amazonaws.com/EpiNOAA/v1-0-0/parquet/cty/YEAR={year}/STATUS=scaled/{yyyymm}.parquet"

    # List to store all data
    all_dfs = []

    for year in years:
        for month in months:
            yyyymm = f"{year}{month:02d}"
            url = base_url.format(year=year, yyyymm=yyyymm)

            # Read Parquet file from S3
            try:
                df = pd.read_parquet(url)
                df_nyc = df[df["fips"].isin(nyc_fips)]
                df_nyc = df_nyc[["date", "fips", "tmax", "tmin", "tavg", "prcp"]]
                all_dfs.append(df_nyc)
                print(f"Loaded {yyyymm}, rows: {len(df_nyc)}")
            except Exception as e:
                print(f"Error loading {url}: {e}")
                continue

    # Combine all months/years
    df_weather = pd.concat(all_dfs, ignore_index=True)

    # Convert types and add derived columns
    for col in ["tmax", "tmin", "tavg", "prcp"]:
        df_weather[col] = df_weather[col].astype(float)

    df_weather["date"] = pd.to_datetime(df_weather["date"])
    df_weather["year"] = df_weather["date"].dt.year
    df_weather["month"] = df_weather["date"].dt.month
    df_weather["day"] = df_weather["date"].dt.day

    if save:
        output_path = os.path.join(
            LOCAL_OUTPUT_DIR, "noaa-nclimgrid-daily", "nyc_fips_weather_data.csv"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_weather.to_csv(output_path, index=False)
        print(f"Saved weather data to {output_path}")

    return df_weather
