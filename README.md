# nyc-311-service-requests

This repository serves as an analysis of NYC 311 service requests data.

Specifically, we focus on service requests tied to the **DOHMH (Department of Health and Mental Hygiene)**. This agency was selected for several reasons:

- **Interesting issues:** The DOHMH dataset contains specific issues that are intriguing to explore.
- **Data size:** The full service requests dataset contains 41M rows. Some agencies have millions of requests, which would make data wrangling and exploration cumbersome. Additionally, working with a larger dataset would make it harder to ingest new data since the existing dataset already consumes significant memory. DOHMH has over 1M rows, which is large but manageable without needing big data tools.
- **Personal interest:** Many DOHMH requests are health-related. Out of curiosity, I wanted to explore whether the data aligns with locations I’ve been to.

The code in this repository is written to answer the question:

> "How should DOHMH resources be allocated in any given week to maximize impact?"

To address this, the repository creates a dashboard to visualize and analyze service request patterns.

The main driver behind this product is **three types of predictive models**:

1. **Request Forecasting:** Predicts the number of new requests expected in a given week.
2. **Request Severity:** Predicts, for each ticket, the likelihood that it will lead to:
   - **Inspection:** Whether the resolution requires an inspection.
   - **SLA breach:** Whether the ticket will exceed its due date.
3. **Request Duration:** Estimates how long it will take for a ticket to be closed.

Using these models, we can determine:

- The expected number of new requests for any given week and their locations.
- The current backlog of tickets and ongoing demand for DOHMH services.
- The probability of each ticket leading to a significant event and the time it will take to close.

We can combine severity and duration to calculate a **severity-weighted time estimate** for each ticket:
Severity-weighted time = Severity × (Estimated Duration)

Tickets with **high probability and high remaining time until the SLA** should receive priority.

for priotizing, we cna do the following:
Summarize up to the H3 level and calculate the total severity-weighted hours for each H3 cell.

Show the total number of hours and estimate how many people would be needed, using the average number of items they complete.

Take the number of inspectors as an input and show which H3s would be prioritized, based on the sorted H3 cells.

### Prioritization Steps

1. **Summarize at the H3 level**  
   Calculate the total severity-weighted hours for each H3 cell.  

2. **Estimate resources**  
   Show the total number of hours and estimate how many people would be needed, using the average number of items they complete.  

3. **Incorporate inspector input**  
   Take the number of inspectors as an input and show which H3s would be prioritized, based on the sorted H3 cells.  


N of inspectors ≈ total_inspections_completed_in_week / (inspections_per_inspector_per_day * workdays_per_week)
225 / (3 * 5) = 15

Once all open tickets and their estimated workloads are mapped, we can prioritize neighborhoods accordingly.

The dashboard includes **adjustable levers** to:

- Change the day of analysis.
- Adjust duration estimates.
- Modify severity weightings.
- Apply weather overrides.

**Disclaimer:** Cursor code assist was used for many throwaway and visualization-heavy sections to quickly inspect the data and troubleshoot issues like misaligned axis ticks.


https://noaa-nclimgrid-daily-pds.s3.amazonaws.com/index.html#EpiNOAA/v1-0-0/parquet/cty/YEAR=2025/STATUS=scaled/

https://arxiv.org/pdf/2502.08649