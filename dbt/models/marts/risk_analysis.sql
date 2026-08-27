-- risk_analysis.sql
-- Risk segmentation with enhanced features

WITH risk_metrics AS (
    SELECT
        grade,
        loan_purpose,
        homeownership,
        state,
        COUNT(*) AS total_loans,
        ROUND(AVG(interest_rate), 2) AS avg_rate,
        ROUND(AVG(debt_to_income), 2) AS avg_dti,
        ROUND(AVG(loan_amount), 2) AS avg_loan,
        ROUND(AVG(annual_income), 2) AS avg_income,
        ROUND(SUM(CASE WHEN loan_status IN ('Charged Off', 'Late (31-120 days)') THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS default_rate,
        ROUND(SUM(paid_total) / NULLIF(SUM(loan_amount), 0) * 100, 2) AS recovery_rate
    FROM {{ source('staging', 'stg_loans') }}
    GROUP BY grade, loan_purpose, homeownership, state
)

SELECT
    grade,
    loan_purpose,
    homeownership,
    state,
    total_loans,
    avg_rate,
    avg_dti,
    avg_loan,
    avg_income,
    default_rate,
    recovery_rate,
    CASE
        WHEN default_rate < 5 THEN 'Low Risk'
        WHEN default_rate < 10 THEN 'Medium Risk'
        WHEN default_rate < 20 THEN 'High Risk'
        ELSE 'Very High Risk'
    END AS risk_category
FROM risk_metrics
ORDER BY default_rate DESC