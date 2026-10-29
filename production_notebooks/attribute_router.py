"""Subtype router with a safe fallback for incomplete future metadata."""
import numpy as np
import pandas as pd
from sales_forecast_config import LOCATION_ATTRIBUTE_COLUMNS

CORE_ATTRIBUTES=['is_relocated','prototype','building_square_footage','total_site_square_footage',
                 'has_outside_seating','total_drivethru_lanes','total_order_points']

class AttributeSubtypeRouter:
    def __init__(self,attribute_model,fallback_model,min_core_fraction=.70):
        self.attribute_model=attribute_model;self.fallback_model=fallback_model
        self.min_core_fraction=min_core_fraction
    def predict(self,data):
        x=data.copy()
        for c in LOCATION_ATTRIBUTE_COLUMNS:
            if c not in x:x[c]=np.nan
        fallback=self.fallback_model.predict(x)
        use=x.subtype.eq('FSU').to_numpy() & x[CORE_ATTRIBUTES].notna().mean(axis=1).ge(self.min_core_fraction).to_numpy()
        if use.any(): fallback[use]=self.attribute_model.predict(x.loc[use])
        return np.asarray(fallback,float)
