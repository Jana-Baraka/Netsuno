from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

class ProcurementAnalyzer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("nlp-thedeep/contract-bert")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "nlp-thedeep/contract-bert"
        )
    
    def analyze_procurement_docs(self, text):
        """Detect favorable/unfavorable contract terms"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        return self.model(**inputs).logits
    
    def score_vendor_risk(self, vendor_history):
        """Use Ibrahim Governance Index"""
        return pd.merge(
            vendor_history,
            pd.read_csv("https://iiag.online/downloads/iiag.csv"),
            on="country"
        )['score'].mean()