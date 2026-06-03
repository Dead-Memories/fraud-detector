import pandas as pd
import numpy as np
import json
import os
import logging
from catboost import CatBoostClassifier

logger = logging.getLogger(__name__)

logger.info('Loading pretrained model...')
model = CatBoostClassifier()
model.load_model('./models/fraud_model.cbm')

model_th = 0.5
logger.info('Pretrained model loaded.')


def make_pred(dt, path_to_file, output_dir):
    proba = model.predict_proba(dt)[:, 1]

    submission = pd.DataFrame({
        'index': pd.read_csv(path_to_file).index,
        'prediction': (proba > model_th).astype(int)
    })
    logger.info('Prediction complete for file: %s', path_to_file)

    fi = dict(zip(dt.columns.tolist(), model.get_feature_importance().tolist()))
    top5 = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True)[:5])
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, 'feature_importance.json')
    with open(json_path, 'w') as f:
        json.dump(top5, f, indent=2)
    logger.info('Feature importance saved to: %s', json_path)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.hist(proba, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
        plt.xlabel('Predicted Score')
        plt.ylabel('Frequency')
        plt.title('Score Distribution')
        plt.tight_layout()
        chart_path = os.path.join(output_dir, 'score_distribution.png')
        plt.savefig(chart_path, dpi=100)
        plt.close()
        logger.info('Score distribution chart saved to: %s', chart_path)
    except Exception as e:
        logger.warning('Could not generate chart: %s', e)

    return submission
