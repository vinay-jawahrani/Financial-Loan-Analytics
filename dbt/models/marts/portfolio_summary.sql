-- portfolio_summary.sql
SELECT
    COUNT(*) AS total_loans,
    ROUND(SUM(loan_amount)::numeric, 2) AS total_funded,
    ROUND(SUM(paid_total)::numeric, 2) AS total_received,
    ROUND(SUM(paid_interest)::numeric, 2) AS total_interest_received,
    ROUND(AVG(interest_rate)::numeric, 2) AS avg_interest_rate,
    ROUND(SUM(CASE WHEN loan_status IN ('Charged Off', 'Late (31-120 days)') THEN loan_amount ELSE 0 END)::numeric, 2) AS total_losses,
    ROUND((SUM(paid_total) / NULLIF(SUM(loan_amount), 0) * 100)::numeric, 2) AS portfolio_recovery_rate,
    ROUND((SUM(CASE WHEN loan_status IN ('Charged Off', 'Late (31-120 days)') THEN 1 ELSE 0 END)::numeric / COUNT(*) * 100), 2) AS overall_default_rate,
    ROUND((SUM(CASE WHEN loan_status = 'Fully Paid' THEN 1 ELSE 0 END)::numeric / COUNT(*) * 100), 2) AS overall_paid_rate
FROM public.stg_loans
