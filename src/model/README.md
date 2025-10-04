# NYC 311 DOHMH Data Preprocessing Module

This module provides comprehensive data cleaning and preprocessing functions for NYC 311 service request data from the Department of Health and Mental Hygiene (DOHMH).

## Features

The preprocessing module includes the following capabilities:

### Data Cleaning Functions

1. **`remove_duplicates(df)`**
   - Removes exact duplicate records
   - Removes duplicates based on key fields (created_date_hour, complaint_type, incident_address, borough, descriptor, resolution_description)

2. **`create_date_features(df)`**
   - Creates derived date and time-based features:
     - `date`: Date portion of created_date
     - `time_to_resolution`: Duration from created to closed
     - `time_closed_to_resolution_update`: Duration from closed to resolution update
     - `closed_before_created`: Flag for invalid date ordering
     - `identical_created_closed_dates`: Flag for same timestamps
     - `created_at_midnight`: Flag for midnight created dates
     - `closed_at_midnight`: Flag for midnight closed dates
     - `time_to_resolution_days`: Resolution time in days

3. **`filter_invalid_dates(df)`**
   - Removes records with closed dates before created dates
   - Removes records with identical created and closed timestamps

4. **`map_freetext_columns(df, mappings_path=None)`**
   - Maps freetext columns to standardized categories using Excel mappings:
     - `complaint_type` → `complaint_family` and related fields
     - `descriptor` → standardized descriptor categories
     - `resolution_description` → `resolution_outcome`

5. **`filter_top_complaint_families(df, top_n=4)`**
   - Filters dataset to only include top N complaint families by volume

6. **`remove_duplicate_resolutions(df)`**
   - Removes records marked as duplicates of previous resolutions

### External Data Merging

7. **`merge_census_data(df, census_data_path, shapefile_path)`**
   - Performs spatial join with census block groups
   - Merges population data by GEOID and year

8. **`merge_weather_data(df, weather_data_path)`**
   - Merges weather data (temperature, precipitation) by FIPS code and date

### Pipeline Functions

9. **`preprocess_dohmh_data(df, ...)`**
   - Main preprocessing pipeline that applies all cleaning steps in the correct order
   - Highly configurable with boolean flags for each step

10. **`preprocess_and_merge_external_data(df, ...)`**
    - Full pipeline combining preprocessing and external data merging

## Usage

### Basic Preprocessing

```python
import pandas as pd
from model import preprocessing

# Load your data
df = pd.read_parquet("your_data.parquet")

# Apply full preprocessing pipeline
df_clean = preprocessing.preprocess_dohmh_data(df, top_n_families=4)
```

### Full Pipeline with External Data

```python
from model import preprocessing

# Define paths
census_path = "data/landing/acs-population/combined_population_data.csv"
shapefile_path = "src/resources/tl_2022_36_bg"
weather_path = "data/landing/noaa-nclimgrid-daily/nyc_fips_weather_data.csv"

# Apply full pipeline
df_complete = preprocessing.preprocess_and_merge_external_data(
    df,
    census_data_path=census_path,
    shapefile_path=shapefile_path,
    weather_data_path=weather_path,
    top_n_families=4
)
```

### Step-by-Step Processing

For more control, you can apply each step individually:

```python
from model import preprocessing

# Step 1: Remove duplicates
df = preprocessing.remove_duplicates(df)

# Step 2: Create date features
df = preprocessing.create_date_features(df)

# Step 3: Filter invalid dates
df = preprocessing.filter_invalid_dates(df)

# Step 4: Map freetext columns
df = preprocessing.map_freetext_columns(df)

# Step 5: Filter to top complaint families
df = preprocessing.filter_top_complaint_families(df, top_n=4)

# Step 6: Remove duplicate resolutions
df = preprocessing.remove_duplicate_resolutions(df)
```

## Dependencies

- pandas
- numpy
- geopandas
- shapely
- openpyxl (for reading Excel mappings)

## Data Requirements

### Input Data

The input dataframe should contain the following columns:

**Required columns:**
- `created_date`: Timestamp of when the service request was created
- `closed_date`: Timestamp of when the service request was closed
- `complaint_type`: Type of complaint (freetext)
- `descriptor`: Additional descriptor (freetext)
- `resolution_description`: Resolution description (freetext)
- `incident_address`: Address of the incident
- `borough`: NYC borough
- `latitude`: Latitude coordinate
- `longitude`: Longitude coordinate
- `year`: Year of the request
- `month`: Month of the request

**Optional columns:**
- `resolution_action_updated_date`: Timestamp of resolution action update
- `due_date`: Due date for the request
- Other domain-specific columns

### External Data Files

1. **Census Population Data** (`combined_population_data.csv`):
   - Must contain: `GEOID`, `year`, and population-related columns

2. **Census Block Group Shapefile** (directory with `.shp`, `.dbf`, etc.):
   - Must contain: `GEOID` field and geometry

3. **Weather Data** (`nyc_fips_weather_data.csv`):
   - Must contain: `fips`, `date`, `tmax`, `tmin`, `tavg`, `prcp`

4. **Freetext Mappings** (`freetext_column_mappings.xlsx`):
   - Must have three sheets: `complaint_type`, `descriptor`, `resolution_description`
   - Default location: `src/resources/mappings/freetext_column_mappings.xlsx`

## Output

The preprocessed dataframe will include:

1. **Original columns** (cleaned)
2. **Derived date features** (8 new columns)
3. **Mapped categorical columns** (complaint_family, resolution_outcome, etc.)
4. **Census data** (GEOID, population data)
5. **Weather data** (tmax, tmin, tavg, prcp)
6. **Helper columns** (created_date_hour, fips)

## Data Quality Summary

The preprocessing pipeline provides detailed logging at each step:

```
================================================================================
DOHMH Data Preprocessing Pipeline
================================================================================
Initial dataset: 1,030,037 rows

Step 1: Removing duplicates...
Duplicate Removal Summary:
  Initial rows: 1,030,037
  After exact duplicate removal: 952,212
  After field-based deduplication: 928,356
  Total removed: 101,681 (9.87%)

Step 2: Creating date features...
  Created 8 new date-related features

Step 3: Filtering invalid dates...
Date Filter Summary:
  Initial rows: 928,356
  Removed closed_before_created: 4,520
  Removed identical_created_closed_dates: 179,929
  Final rows: 743,907

...
```

## Examples

See `example_usage.py` for complete working examples of:
- Basic preprocessing
- Full pipeline with external data
- Step-by-step processing

## Notes

- The preprocessing pipeline is designed to be idempotent (safe to run multiple times)
- All date filtering removes records that are unreliable for duration modeling
- The default behavior filters to the top 4 complaint families, which typically captures 90%+ of records
- Spatial joins may take several minutes for large datasets

## Related Notebooks

The functions in this module were extracted from:
- `notebooks/Step 3 - Data Cleanup.ipynb`
- `notebooks/Step 4 - Merge New Datasets.ipynb`

