import json
from pathlib import Path

OUT = Path("data")
OUT.mkdir(exist_ok=True)

# ---------- RAG corpus (finance policy / current guidance) ----------
# Think: content you want to stay "fresh" and not bake into weights.
rag_docs = [
    {
        "id": "credit_policy_summary",
        "title": "Retail Banking Credit Policy - Summary",
        "text": (
            "Credit policy (summary):\n"
            "- Minimum FICO for unsecured personal loans: 660\n"
            "- Maximum debt-to-income (DTI): 43%\n"
            "- Require 24 months of employment history for prime tier\n"
            "- For auto loans, maximum LTV: 110% for new vehicles\n"
        ),
    },
    {
        "id": "personal_loan_plus_policy",
        "title": "Personal Loan Plus - Example Underwriting (2026-01-15)",
        "text": (
            "Personal Loan Plus policy:\n"
            "- Minimum FICO: 705\n"
            "- Maximum DTI: 41.5%\n"
            "- Employment history: 18 months minimum\n"
            "- Maximum amount: $35,000\n"
            "- Manual review required if stated_income == true\n"
        ),
    },
    {
        "id": "mortgage_guidelines",
        "title": "Mortgage Underwriting Guidelines",
        "text": (
            "Mortgage guidelines (summary):\n"
            "- Conforming loan max LTV: 80% without PMI\n"
            "- Jumbo loans require 6 months reserves\n"
            "- Minimum FICO for FHA: 580\n"
        ),
    },
    {
        "id": "kyc_requirements",
        "title": "KYC / AML Requirements",
        "text": (
            "KYC requirements:\n"
            "- Verify government ID\n"
            "- Collect SSN or tax ID\n"
            "- Screen against OFAC list\n"
            "- Flag high-risk geographies for manual review\n"
        ),
    },
]

with open(OUT / "rag_corpus.jsonl", "w") as f:
    for d in rag_docs:
        f.write(json.dumps(d) + "\n")

# ---------- LoRA training set (DMN JSON) ----------
# Stable "how we express decisions" patterns for a banking domain.
# We train the model to output *only* DMN JSON with strict formatting.
SCHEMA_HINT = (
    "Use schema: decisionModel {id, name, inputs[{id,type}], output{id,type}, rules[{id,if,then}]}. "
    "Output JSON only."
)
lora_examples = [
    {
        "prompt": (
            "Generate DMN JSON for a personal loan approval decision. "
            "Inputs: fico, dti. Rules: approve when fico >= 680 AND dti <= 40. Otherwise refer. "
            f"{SCHEMA_HINT}"
        ),
        "response": (
            "{\n"
            "  \"decisionModel\": {\n"
            "    \"id\": \"personal_loan_approval\",\n"
            "    \"name\": \"Personal Loan Approval\",\n"
            "    \"inputs\": [\n"
            "      {\"id\": \"fico\", \"type\": \"number\"},\n"
            "      {\"id\": \"dti\", \"type\": \"number\"}\n"
            "    ],\n"
            "    \"output\": {\"id\": \"decision\", \"type\": \"string\"},\n"
            "    \"rules\": [\n"
            "      {\n"
            "        \"id\": \"approve_prime\",\n"
            "        \"if\": \"fico >= 680 and dti <= 40\",\n"
            "        \"then\": \"approve\"\n"
            "      },\n"
            "      {\n"
            "        \"id\": \"refer\",\n"
            "        \"if\": \"otherwise\",\n"
            "        \"then\": \"refer\"\n"
            "      }\n"
            "    ]\n"
            "  }\n"
            "}\n"
        ),
    },
    {
        "prompt": (
            "Generate DMN JSON for a credit card limit decision. "
            "Inputs: fico, income, existing_limit. "
            "Rules: if fico >= 720 and income >= 60000 then limit = existing_limit * 1.5; "
            "if fico >= 660 and income >= 40000 then limit = existing_limit * 1.2; "
            f"else limit = existing_limit. {SCHEMA_HINT}"
        ),
        "response": (
            "{\n"
            "  \"decisionModel\": {\n"
            "    \"id\": \"credit_limit_increase\",\n"
            "    \"name\": \"Credit Limit Increase\",\n"
            "    \"inputs\": [\n"
            "      {\"id\": \"fico\", \"type\": \"number\"},\n"
            "      {\"id\": \"income\", \"type\": \"number\"},\n"
            "      {\"id\": \"existing_limit\", \"type\": \"number\"}\n"
            "    ],\n"
            "    \"output\": {\"id\": \"new_limit\", \"type\": \"number\"},\n"
            "    \"rules\": [\n"
            "      {\n"
            "        \"id\": \"tier_1\",\n"
            "        \"if\": \"fico >= 720 and income >= 60000\",\n"
            "        \"then\": \"existing_limit * 1.5\"\n"
            "      },\n"
            "      {\n"
            "        \"id\": \"tier_2\",\n"
            "        \"if\": \"fico >= 660 and income >= 40000\",\n"
            "        \"then\": \"existing_limit * 1.2\"\n"
            "      },\n"
            "      {\n"
            "        \"id\": \"default\",\n"
            "        \"if\": \"otherwise\",\n"
            "        \"then\": \"existing_limit\"\n"
            "      }\n"
            "    ]\n"
            "  }\n"
            "}\n"
        ),
    },
    {
        "prompt": (
            "Generate DMN JSON for mortgage eligibility. Inputs: fico, ltv, reserves_months. "
            "Rules: approve if fico >= 700 and ltv <= 80 and reserves_months >= 6; else decline. "
            f"{SCHEMA_HINT}"
        ),
        "response": (
            "{\n"
            "  \"decisionModel\": {\n"
            "    \"id\": \"mortgage_eligibility\",\n"
            "    \"name\": \"Mortgage Eligibility\",\n"
            "    \"inputs\": [\n"
            "      {\"id\": \"fico\", \"type\": \"number\"},\n"
            "      {\"id\": \"ltv\", \"type\": \"number\"},\n"
            "      {\"id\": \"reserves_months\", \"type\": \"number\"}\n"
            "    ],\n"
            "    \"output\": {\"id\": \"decision\", \"type\": \"string\"},\n"
            "    \"rules\": [\n"
            "      {\n"
            "        \"id\": \"approve\",\n"
            "        \"if\": \"fico >= 700 and ltv <= 80 and reserves_months >= 6\",\n"
            "        \"then\": \"approve\"\n"
            "      },\n"
            "      {\n"
            "        \"id\": \"decline\",\n"
            "        \"if\": \"otherwise\",\n"
            "        \"then\": \"decline\"\n"
            "      }\n"
            "    ]\n"
            "  }\n"
            "}\n"
        ),
    },
    {
        "prompt": (
            "Generate DMN JSON that returns an error when a required input is missing. "
            "Input: fico. Rule: if fico is missing then error. "
            f"{SCHEMA_HINT}"
        ),
        "response": (
            "{\n"
            "  \"decisionModel\": {\n"
            "    \"id\": \"input_validation\",\n"
            "    \"name\": \"Input Validation\",\n"
            "    \"inputs\": [\n"
            "      {\"id\": \"fico\", \"type\": \"number\"}\n"
            "    ],\n"
            "    \"output\": {\"id\": \"error\", \"type\": \"string\"},\n"
            "    \"rules\": [\n"
            "      {\n"
            "        \"id\": \"missing_fico\",\n"
            "        \"if\": \"fico is null\",\n"
            "        \"then\": \"missing_required_input\"\n"
            "      }\n"
            "    ]\n"
            "  }\n"
            "}\n"
        ),
    },
    {
        "prompt": (
            "Generate DMN JSON for an auto loan rate tier decision. Inputs: fico, ltv, term_months. "
            "Rules: tier_1 if fico >= 720 and ltv <= 100 and term_months <= 60; "
            "tier_2 if fico >= 680 and ltv <= 110; otherwise refer. "
            f"{SCHEMA_HINT}"
        ),
        "response": (
            "{\n"
            "  \"decisionModel\": {\n"
            "    \"id\": \"auto_loan_rate_tier\",\n"
            "    \"name\": \"Auto Loan Rate Tier\",\n"
            "    \"inputs\": [\n"
            "      {\"id\": \"fico\", \"type\": \"number\"},\n"
            "      {\"id\": \"ltv\", \"type\": \"number\"},\n"
            "      {\"id\": \"term_months\", \"type\": \"number\"}\n"
            "    ],\n"
            "    \"output\": {\"id\": \"rate_tier\", \"type\": \"string\"},\n"
            "    \"rules\": [\n"
            "      {\n"
            "        \"id\": \"tier_1\",\n"
            "        \"if\": \"fico >= 720 and ltv <= 100 and term_months <= 60\",\n"
            "        \"then\": \"tier_1\"\n"
            "      },\n"
            "      {\n"
            "        \"id\": \"tier_2\",\n"
            "        \"if\": \"fico >= 680 and ltv <= 110\",\n"
            "        \"then\": \"tier_2\"\n"
            "      },\n"
            "      {\n"
            "        \"id\": \"refer\",\n"
            "        \"if\": \"otherwise\",\n"
            "        \"then\": \"refer\"\n"
            "      }\n"
            "    ]\n"
            "  }\n"
            "}\n"
        ),
    },
    {
        "prompt": (
            "Generate DMN JSON for an overdraft fee waiver decision. "
            "Inputs: account_age_months, overdraft_count_90d, balance. "
            "Rules: waive if account_age_months >= 12 and overdraft_count_90d <= 1 and balance >= 0; "
            f"otherwise deny. {SCHEMA_HINT}"
        ),
        "response": (
            "{\n"
            "  \"decisionModel\": {\n"
            "    \"id\": \"overdraft_fee_waiver\",\n"
            "    \"name\": \"Overdraft Fee Waiver\",\n"
            "    \"inputs\": [\n"
            "      {\"id\": \"account_age_months\", \"type\": \"number\"},\n"
            "      {\"id\": \"overdraft_count_90d\", \"type\": \"number\"},\n"
            "      {\"id\": \"balance\", \"type\": \"number\"}\n"
            "    ],\n"
            "    \"output\": {\"id\": \"decision\", \"type\": \"string\"},\n"
            "    \"rules\": [\n"
            "      {\n"
            "        \"id\": \"waive\",\n"
            "        \"if\": \"account_age_months >= 12 and overdraft_count_90d <= 1 and balance >= 0\",\n"
            "        \"then\": \"waive\"\n"
            "      },\n"
            "      {\n"
            "        \"id\": \"deny\",\n"
            "        \"if\": \"otherwise\",\n"
            "        \"then\": \"deny\"\n"
            "      }\n"
            "    ]\n"
            "  }\n"
            "}\n"
        ),
    },
    {
        "prompt": (
            "Generate DMN JSON for a fraud review decision. Inputs: txn_amount, country_risk, mcc_risk. "
            "Rules: review if txn_amount >= 5000 or country_risk == \"high\" or mcc_risk == \"high\"; "
            f"otherwise approve. {SCHEMA_HINT}"
        ),
        "response": (
            "{\n"
            "  \"decisionModel\": {\n"
            "    \"id\": \"fraud_review\",\n"
            "    \"name\": \"Fraud Review\",\n"
            "    \"inputs\": [\n"
            "      {\"id\": \"txn_amount\", \"type\": \"number\"},\n"
            "      {\"id\": \"country_risk\", \"type\": \"string\"},\n"
            "      {\"id\": \"mcc_risk\", \"type\": \"string\"}\n"
            "    ],\n"
            "    \"output\": {\"id\": \"decision\", \"type\": \"string\"},\n"
            "    \"rules\": [\n"
            "      {\n"
            "        \"id\": \"review\",\n"
            "        \"if\": \"txn_amount >= 5000 or country_risk == \\\"high\\\" or mcc_risk == \\\"high\\\"\",\n"
            "        \"then\": \"review\"\n"
            "      },\n"
            "      {\n"
            "        \"id\": \"approve\",\n"
            "        \"if\": \"otherwise\",\n"
            "        \"then\": \"approve\"\n"
            "      }\n"
            "    ]\n"
            "  }\n"
            "}\n"
        ),
    },
    {
        "prompt": (
            "Generate DMN JSON for a KYC risk tier decision. Inputs: pep, ofac_hit, country_risk. "
            "Rules: block if ofac_hit == true; enhanced if pep == true or country_risk == \"high\"; "
            f"otherwise standard. {SCHEMA_HINT}"
        ),
        "response": (
            "{\n"
            "  \"decisionModel\": {\n"
            "    \"id\": \"kyc_risk_tier\",\n"
            "    \"name\": \"KYC Risk Tier\",\n"
            "    \"inputs\": [\n"
            "      {\"id\": \"pep\", \"type\": \"boolean\"},\n"
            "      {\"id\": \"ofac_hit\", \"type\": \"boolean\"},\n"
            "      {\"id\": \"country_risk\", \"type\": \"string\"}\n"
            "    ],\n"
            "    \"output\": {\"id\": \"risk_tier\", \"type\": \"string\"},\n"
            "    \"rules\": [\n"
            "      {\n"
            "        \"id\": \"block\",\n"
            "        \"if\": \"ofac_hit == true\",\n"
            "        \"then\": \"block\"\n"
            "      },\n"
            "      {\n"
            "        \"id\": \"enhanced\",\n"
            "        \"if\": \"pep == true or country_risk == \\\"high\\\"\",\n"
            "        \"then\": \"enhanced\"\n"
            "      },\n"
            "      {\n"
            "        \"id\": \"standard\",\n"
            "        \"if\": \"otherwise\",\n"
            "        \"then\": \"standard\"\n"
            "      }\n"
            "    ]\n"
            "  }\n"
            "}\n"
        ),
    },
    {
        "prompt": (
            "Generate DMN JSON for a small business loan decision. "
            "Inputs: annual_revenue, years_in_business, fico. "
            "Rules: approve if annual_revenue >= 250000 and years_in_business >= 2 and fico >= 680; "
            f"otherwise refer. {SCHEMA_HINT}"
        ),
        "response": (
            "{\n"
            "  \"decisionModel\": {\n"
            "    \"id\": \"small_business_loan\",\n"
            "    \"name\": \"Small Business Loan\",\n"
            "    \"inputs\": [\n"
            "      {\"id\": \"annual_revenue\", \"type\": \"number\"},\n"
            "      {\"id\": \"years_in_business\", \"type\": \"number\"},\n"
            "      {\"id\": \"fico\", \"type\": \"number\"}\n"
            "    ],\n"
            "    \"output\": {\"id\": \"decision\", \"type\": \"string\"},\n"
            "    \"rules\": [\n"
            "      {\n"
            "        \"id\": \"approve\",\n"
            "        \"if\": \"annual_revenue >= 250000 and years_in_business >= 2 and fico >= 680\",\n"
            "        \"then\": \"approve\"\n"
            "      },\n"
            "      {\n"
            "        \"id\": \"refer\",\n"
            "        \"if\": \"otherwise\",\n"
            "        \"then\": \"refer\"\n"
            "      }\n"
            "    ]\n"
            "  }\n"
            "}\n"
        ),
    },
    {
        "prompt": (
            "Generate DMN JSON for a credit card APR decision. Inputs: fico, utilization. "
            "Rules: apr 14.99 if fico >= 720 and utilization <= 30; "
            "apr 19.99 if fico >= 660 and utilization <= 50; otherwise apr 24.99. "
            f"{SCHEMA_HINT}"
        ),
        "response": (
            "{\n"
            "  \"decisionModel\": {\n"
            "    \"id\": \"credit_card_apr\",\n"
            "    \"name\": \"Credit Card APR\",\n"
            "    \"inputs\": [\n"
            "      {\"id\": \"fico\", \"type\": \"number\"},\n"
            "      {\"id\": \"utilization\", \"type\": \"number\"}\n"
            "    ],\n"
            "    \"output\": {\"id\": \"apr\", \"type\": \"number\"},\n"
            "    \"rules\": [\n"
            "      {\n"
            "        \"id\": \"apr_low\",\n"
            "        \"if\": \"fico >= 720 and utilization <= 30\",\n"
            "        \"then\": 14.99\n"
            "      },\n"
            "      {\n"
            "        \"id\": \"apr_mid\",\n"
            "        \"if\": \"fico >= 660 and utilization <= 50\",\n"
            "        \"then\": 19.99\n"
            "      },\n"
            "      {\n"
            "        \"id\": \"apr_high\",\n"
            "        \"if\": \"otherwise\",\n"
            "        \"then\": 24.99\n"
            "      }\n"
            "    ]\n"
            "  }\n"
            "}\n"
        ),
    },
    {
        "prompt": (
            "Generate DMN JSON for a PMI requirement decision. Input: ltv. "
            "Rules: pmi_required = true if ltv > 80; else false. "
            f"{SCHEMA_HINT}"
        ),
        "response": (
            "{\n"
            "  \"decisionModel\": {\n"
            "    \"id\": \"pmi_required\",\n"
            "    \"name\": \"PMI Required\",\n"
            "    \"inputs\": [\n"
            "      {\"id\": \"ltv\", \"type\": \"number\"}\n"
            "    ],\n"
            "    \"output\": {\"id\": \"pmi_required\", \"type\": \"boolean\"},\n"
            "    \"rules\": [\n"
            "      {\n"
            "        \"id\": \"pmi_yes\",\n"
            "        \"if\": \"ltv > 80\",\n"
            "        \"then\": true\n"
            "      },\n"
            "      {\n"
            "        \"id\": \"pmi_no\",\n"
            "        \"if\": \"otherwise\",\n"
            "        \"then\": false\n"
            "      }\n"
            "    ]\n"
            "  }\n"
            "}\n"
        ),
    },
    {
        "prompt": (
            "Generate DMN JSON for an employment history decision. Input: months_employed. "
            "Rules: approve if months_employed >= 24; otherwise refer. "
            f"{SCHEMA_HINT}"
        ),
        "response": (
            "{\n"
            "  \"decisionModel\": {\n"
            "    \"id\": \"employment_history\",\n"
            "    \"name\": \"Employment History\",\n"
            "    \"inputs\": [\n"
            "      {\"id\": \"months_employed\", \"type\": \"number\"}\n"
            "    ],\n"
            "    \"output\": {\"id\": \"decision\", \"type\": \"string\"},\n"
            "    \"rules\": [\n"
            "      {\n"
            "        \"id\": \"approve\",\n"
            "        \"if\": \"months_employed >= 24\",\n"
            "        \"then\": \"approve\"\n"
            "      },\n"
            "      {\n"
            "        \"id\": \"refer\",\n"
            "        \"if\": \"otherwise\",\n"
            "        \"then\": \"refer\"\n"
            "      }\n"
            "    ]\n"
            "  }\n"
            "}\n"
        ),
    },
]

with open(OUT / "lora_train.jsonl", "w") as f:
    for ex in lora_examples:
        f.write(json.dumps(ex) + "\n")


print("Wrote:")
print(" - data/rag_corpus.jsonl")
print(" - data/lora_train.jsonl")
