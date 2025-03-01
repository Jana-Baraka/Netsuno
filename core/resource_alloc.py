import requests
import pandas as pd
from huggingface_hub import hf_hub_download

class ResourceOptimizer:
    def __init__(self):
        self.ioda_api = "https://api.ioda.inetintel.cc.gatech.edu/v2/events"
    
    def get_outage_alerts(self, region_code="AFR"):
        """Fetch real-time outage data from IODA"""
        return requests.get(
            f"{self.ioda_api}?region={region_code}"
        ).json()
    
    def optimize_5g_params(self, speedtest_data):
        """Use Infinite Dataset Hub model"""
        model = hf_hub_download(
            repo_id="infinite-dataset-hub/5GNetworkOptimization",
            filename="optimizer.pkl"
        )
        return model.predict(speedtest_data)
    
    def calculate_priority_score(self, school, outage_data):
        """Prioritize schools in outage-prone areas"""
        return (
            school['students'] * 0.6 + 
            outage_data['outage_frequency'] * 0.4
        )