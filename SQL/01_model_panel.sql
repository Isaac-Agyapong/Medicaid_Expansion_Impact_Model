-- =====================================================================
-- Model dataset: the county panel built by the analytics project
-- (github.com/Isaac-Agyapong/Health-Insurance-Coverage-Gap-Analysis, database medicaid_coverage).
--
-- One row per county per year, 2008-2023, for the 3,035 counties in the balanced panel of the
-- 46 states in the study (the 5 states that covered low-income adults before 2014 are excluded).
-- Outcome: uninsured rate of adults 18-64 at or below 138% of the federal poverty level.
-- County traits are 2013 values (before any state expanded), so they cannot be affected by expansion.
-- =====================================================================
SELECT county_fips, county_name, state_abbrev, state_fips, analysis_group, expansion_year, year,
       pct_uninsured, pct_uninsured_moe, population, uninsured,
       pct_uninsured_138_400, pct_uninsured_children,
       rurality, rucc_2013, population_2013, pct_nh_white_2013, pct_nh_black_2013, pct_hispanic_2013,
       pct_age_65plus_2013, poverty_pct_2013, median_income_2013
FROM analytics.mv_county_panel
ORDER BY county_fips, year;
