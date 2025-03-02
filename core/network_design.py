import geopandas as gpd
import rasterio
from huggingface_hub import hf_hub_download
from transformers import BeitForSemanticSegmentation  # Changed import

class NetworkDesigner:
    def __init__(self):
        # Load correct BEiT segmentation model
        self.land_cover_model = BeitForSemanticSegmentation.from_pretrained(
            "microsoft/beit-base-finetuned-ade-640-640"
        )
    
    def load_schools_data(self, country_code="ETH"):
        """Fetch Giga schools data from API"""
        return gpd.read_file(
            f"https://giga-api.unicef.org/schools?country={country_code}"
        )
    
    def analyze_coverage_gaps(self, schools_gdf, cell_towers_gdf):
        """Identify areas with schools but no existing towers"""
        return gpd.sjoin_nearest(
            schools_gdf, 
            cell_towers_gdf,
            how="left",
            max_distance=5000
        ).query("distance > 5000")
    
    def classify_terrain(self, sentinel2_tif_path):
        """Land cover classification using Sentinel-2"""
        with rasterio.open(sentinel2_tif_path) as src:
            img = src.read([4,3,2])  # RGB bands
        return self.land_cover_model(img.unsqueeze(0))