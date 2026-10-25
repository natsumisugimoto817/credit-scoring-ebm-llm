"""Business names for features, used in all plots and memos.

One dict per dataset. Plot functions take `labels=...`; a feature that is not in the dict keeps
its raw name, so the code works for any dataset.
"""

LENDING_CLUB = {
    "loan_amnt": "Loan amount ($)",
    "annual_inc": "Annual income ($)",
    "loan_to_income": "Loan / annual income",
    "dti": "Debt-to-income ratio (%)",
    "fico": "FICO score",
    "emp_length_yrs": "Years in current job",
    "credit_hist_months": "Credit history (months)",
    "delinq_2yrs": "Late payments (last 2 years)",
    "inq_last_6mths": "Credit inquiries (last 6 months)",
    "open_acc": "Open credit lines",
    "pub_rec": "Negative public records",
    "revol_bal": "Credit card balance ($)",
    "revol_util": "Credit card utilisation (%)",
    "total_acc": "Total credit lines",
    "mort_acc": "Mortgage accounts",
    "pub_rec_bankruptcies": "Past bankruptcies",
    "home_ownership": "Home ownership",
    "verification_status": "Income verification",
    "purpose": "Loan purpose",
    "application_type": "Application type",
}
