import pandas as pd
import random

# Helper functions
def generate_credit_score():
    return random.randint(300, 850)

def generate_income():
    return round(random.uniform(2000, 10000), 2)

def generate_family_members():
    return random.randint(0, 6)

def generate_outstanding_debt():
    return round(random.uniform(0, 5000), 2)

def generate_loan_request():
    return round(random.uniform(5000, 50000), 2)

def generate_employment_status():
    return random.choice(["Employed", "Self-Employed", "Unemployed"])

# Calculate Debt-to-Income ratio and risk scores based on the business logic above
def calculate_dti(income, debt, loan_payment):
    return round((debt + loan_payment) / income * 100, 2)

def calculate_risk_score(credit_score, dti, family_members):
    # Credit Score
    if credit_score >= 720:
        credit_score_points = 5
    elif 680 <= credit_score < 720:
        credit_score_points = 4
    elif 640 <= credit_score < 680:
        credit_score_points = 3
    elif 600 <= credit_score < 640:
        credit_score_points = 2
    else:
        credit_score_points = 1
    
    # DTI Score
    if dti < 35:
        dti_points = 5
    elif 35 <= dti < 50:
        dti_points = 3
    else:
        dti_points = 1

    # Family Responsibility Score
    if family_members <= 1:
        family_points = 5
    elif 2 <= family_members <= 3:
        family_points = 3
    else:
        family_points = 1

    return credit_score_points + dti_points + family_points

def generate_loan_decision(risk_score):
    return "Approved" if risk_score >= 10 else "Not Approved"

def create_pickle_csv(cnt,typ):
    # Generate synthetic data
    data = []
    for sl in range(cnt):
        income = generate_income()
        credit_score = generate_credit_score()
        family_members = generate_family_members()
        outstanding_debt = generate_outstanding_debt()
        loan_request = generate_loan_request()
        loan_payment = round(loan_request / 60, 2)  # Assuming a 5-year loan period
        
        dti = calculate_dti(income, outstanding_debt, loan_payment)
        risk_score = calculate_risk_score(credit_score, dti, family_members)
        decision = generate_loan_decision(risk_score)
        
        data.append({
            "Application No." : sl +1,
            "Income": income,
            "Credit Score": credit_score,
            "Family Members": family_members,
            "Outstanding Debt": outstanding_debt,
            "Loan Request": loan_request,
            "Employment Status": generate_employment_status(),
            # "Debt-to-Income Ratio (%)": dti,
            # "Risk Score": risk_score,
            "Loan Decision": decision
        })

    # Convert the data to a pandas DataFrame
    df = pd.DataFrame(data)
    df.head()
    # print(df.to_string())

    # Exporting the data to a CSV file
    file_path = "Assignment1/synthetic_data/loan_application_data_" + typ +".csv"
    df.to_csv(file_path, index=False)

    df.to_pickle("Assignment1/synthetic_data/loan_application_data_" + typ +".pkl")

create_pickle_csv(20,"examples")
create_pickle_csv(30,"gold_examples")
