"""
Run an ensemble of plans in accordance with Utah's redistricting requirements.

The seed/starting point is the 2021 Utah Congressional District plan.

Political data can not be used to help draw lines, only to evaluate whether a plan is fair after it is drawn.

Neutral redistricting standards, in priority order:
1. Adhering to the Constitution of the United States and federal laws, such as the Voting Rights Act, 52 U.S.C. Secs. 10101 through 10702, including, to the extent required, achieving equal population among districts using the most recent national decennial enumeration made by the authority of the United States; [No more than 0.1% population deviation from the ideal is permitted]
2. Minimizing the division of municipalities and counties across multiple districts, giving first priority to minimizing the division of municipalities and second priority to minimizing the division of counties; [Use the municipal and county region assignments as a surcharge on region splitting; after each iteration, count how many cities and counties are split across districts]
3. creating districts that are geographically compact; [Do not apply to congressional redistricting, but record compactness metrics for each iteration]
4. creating districts that are contiguous and that allow for the ease of transportation throughout the district; [No data; unguided]
5. preserving traditional neighborhoods and local communities of interest; [Use the COI data for higher ed, metro/micro statistical areas, and school districts and surcharges]
6. following natural and geographic features, boundaries, and barriers; and [Aligns well with county lines; no additional work]
7. maximizing boundary agreement among different types of districts. [No additional work]

Things to measure after each iteration:
- Number of city splits
- Number of county splits
- Compactness of districts
- Partisan metrics, based on the average partisan preference for each election year with data
    - Partisan bias
    - Mean median difference
    - Efficiency gap
    - Number of seats each party wins
    - R/D margins in each district
"""