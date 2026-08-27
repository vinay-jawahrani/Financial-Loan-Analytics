-- risk_analysis.sql
-- Risk segmentation by grade

SELECT
    grade,
    COUNT(*) AS total_loans,
    ROUND(AVG(loan_amount)::numeric, 2) AS avg_loan,
    ROUND(AVG(interest_rate)::numeric, 2) AS avg_rate,
    ROUND(AVG(debt_to_income)::numeric, 2) AS avg_dti,
    ROUND(AVG(annual_income)::numeric, 2) AS avg_income,
    ROUND((SUM(CASE WHEN loan_status IN ('Charged Off', 'Late (31-120 days)') THEN 1 ELSE 0 END)::numeric / COUNT(*) * 100), 2) AS default_rate,
    ROUND((SUM(CASE WHEN loan_status = 'Fully Paid' THEN 1 ELSE 0 END)::numeric / COUNT(*) * 100), 2) AS paid_rate,
    CASE
        WHEN ROUND((SUM(CASE WHEN loan_status IN ('Charged Off', 'Late (31-120 days)') THEN 1 ELSE 0 END)::numeric / COUNT(*) * 100), 2) < 5 THEN 'Low Risk'
        WHEN ROUND((SUM(CASE WHEN loan_status IN ('Charged Off', 'Late (31-120 days)') THEN 1 ELSE 0 END)::numeric / COUNT(*) * 100), 2) < 10 THEN 'Medium Risk'
        WHEN ROUND((SUM(CASE WHEN loan_status IN ('Charged Off', 'Late (31-120 days)') THEN 1 ELSE 0 END)::numeric / COUNT(*) * 100), 2) < 20 THEN 'High Risk'
        ELSE 'Very High Risk'
    END AS risk_category
FROM {{ ref('stg_loans') }}
GROUP BY grade
ORDER BY grade