-- loan_performance.sql
-- Loan performance summary

SELECT
    grade,
    loan_status,
    COUNT(*) AS total_loans,
    ROUND(AVG(loan_amount)::numeric, 2) AS avg_loan_amount,
    ROUND(AVG(interest_rate)::numeric, 2) AS avg_interest_rate,
    ROUND(AVG(debt_to_income)::numeric, 2) AS avg_dti,
    ROUND(AVG(annual_income)::numeric, 2) AS avg_income,
    ROUND(SUM(paid_total)::numeric, 2) AS total_paid,
    ROUND(SUM(loan_amount)::numeric, 2) AS total_loans_funded,
    ROUND((SUM(paid_total) / NULLIF(SUM(loan_amount), 0) * 100)::numeric, 2) AS recovery_rate
FROM {{ source('staging', 'stg_loans') }}
GROUP BY grade, loan_status
ORDER BY grade, loan_status