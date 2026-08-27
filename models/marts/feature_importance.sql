-- feature_importance.sql
-- Identify key factors driving defaults

SELECT
    grade,
    homeownership,
    verification_status AS income_verified,
    ROUND(AVG(debt_to_income), 2) AS avg_dti,
    ROUND(AVG(inquiries_last_12m), 2) AS avg_inquiries,
    ROUND(AVG(delinq_2y), 2) AS avg_delinq,
    ROUND(AVG(account_never_delinq_percent), 2) AS avg_good_accounts,
    ROUND(SUM(CASE WHEN loan_status IN ('Charged Off', 'Late (31-120 days)') THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS default_rate
FROM {{ source('staging', 'stg_loans') }}
GROUP BY grade, homeownership, verification_status
ORDER BY default_rate DESC