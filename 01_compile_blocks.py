#!/usr/bin/env python3
"""
Comprehensive Utah Census Block Dataset Creator
Following GerryChain format for redistricting analysis

This script combines:
- Election data (2016, 2018, 2020, 2024) at census block level
- Demographic data (race, ethnicity)
- County and municipality assignments
- Communities of Interest (COI) assignments
- All data formatted for GerryChain compatibility

Output columns
--------------
# Basics
- GEOID20
- MUNINAME
- MUNIID
- COUNTYNAME
- COUNTYID
- TOTPOP
- VAP
# Demographics
- NH_WHITE
- NH_BLACK
- NH_AMIN
- NH_ASIAN
- NH_NHPI
- NH_OTHER
- NH_2MORE
- HISP
- H_WHITE
- H_BLACK
- H_AMIN
- H_ASIAN
- H_NHPI
- H_OTHER
- H_2MORE
# Communities of Interest
- AMIND_RES
- HIGHER_ED
- METRO_AREA
- SCHOOL_DIST
# Election results
Format YYOOOP
- Years (2016, 2018, 2020, 2024)
- Offices (PRE, GOV, ATG, AUD, TRE, USS)
- Parties (D, R, O)
"""
