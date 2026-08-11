"""Random Forest ensemble classifier."""
from sklearn.ensemble import RandomForestClassifier

def build_model():
    return RandomForestClassifier(
        n_estimators=300, max_depth=10, random_state=42, n_jobs=-1
    )
