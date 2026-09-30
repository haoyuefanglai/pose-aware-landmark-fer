
# --- 路径引导：src/ 下 core / pipeline / experiments / apps 之间可互相 import ---
# 本段由结构重构引入。算法逻辑不依赖它，仅用于把同级子目录加入模块搜索路径，
# 使 `from feature_extractor import ...` 这类平铺导入在跨目录后依然有效。
import os as _os
import sys as _sys

_SRC_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _sub in ("core", "pipeline", "experiments", "apps"):
    _sub_path = _os.path.join(_SRC_DIR, _sub)
    if _os.path.isdir(_sub_path) and _sub_path not in _sys.path:
        _sys.path.insert(0, _sub_path)
# --- 路径引导结束 ---

import unittest
import tempfile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from classifier import ExpressionClassifier, EXPRESSION_CLASSES, train_and_evaluate
from feature_extractor import FaceFeatureExtractor

class PipelineTests(unittest.TestCase):
    def test_rules_keep_neutral_with_ck_model(self):
        clf = ExpressionClassifier('model_svm.pkl')
        label, score, probs = clf.predict(np.zeros(136), {}, {})
        self.assertEqual(label, 'neutral')
        self.assertEqual(set(probs), set(EXPRESSION_CLASSES))
        self.assertAlmostEqual(sum(probs.values()), 1)
        self.assertTrue(np.isfinite(score))

    def test_smile_actions(self):
        label, _, _ = ExpressionClassifier().predict(np.zeros(136), {},
            {'mouthSmileLeft': .8, 'mouthSmileRight': .8})
        self.assertEqual(label, 'smile')

    def test_ml_uses_estimator_class_order(self):
        with tempfile.TemporaryDirectory() as folder:
            model = DummyClassifier(strategy='prior').fit(np.zeros((3,136)), ['sad','smile','smile'])
            path = str(Path(folder) / 'model.pkl')
            joblib.dump({'model': model, 'classes': ['smile','sad']}, path)
            clf = ExpressionClassifier(path)
            label, _, probs = clf.predict(np.zeros(136), use_ml_model=True)
            self.assertEqual(label, 'smile')
            self.assertAlmostEqual(probs['smile'], 2/3)

    def test_missing_ml_model_is_error(self):
        with self.assertRaises(ValueError):
            ExpressionClassifier().predict(np.zeros(136), use_ml_model=True)

    def test_normalization_translation_scale(self):
        extractor = FaceFeatureExtractor.__new__(FaceFeatureExtractor)
        pts = np.random.default_rng(42).normal(size=(68,2)).astype(np.float32)
        np.testing.assert_allclose(extractor._normalize_landmarks(pts),
            extractor._normalize_landmarks(pts * 3 + 12), atol=1e-5)

    def test_blank_image_has_no_face(self):
        extractor = FaceFeatureExtractor(static_mode=True)
        try:
            self.assertFalse(extractor.process_frame(np.zeros((240,320,3),np.uint8))['detected'])
        finally:
            extractor.close()

    def test_training_refits_and_validates_subjects(self):
        with tempfile.TemporaryDirectory() as folder:
            df = pd.DataFrame(np.random.default_rng(4).normal(size=(12,136)), columns=[f'feat_{i}' for i in range(136)])
            df['subject_id'] = np.repeat(['a','b','c'],4)
            df['label'] = ['smile','sad'] * 6
            csv = str(Path(folder) / 'data.csv')
            saved = str(Path(folder) / 'model.pkl')
            df.to_csv(csv,index=False)
            train_and_evaluate(csv, n_splits=3, save_path=saved)
            data = joblib.load(saved)
            self.assertEqual(data['training_samples'],12)
            self.assertEqual(data['model'].shape_fit_[0],12)
            df['subject_id'] = 'one'
            df.to_csv(csv,index=False)
            with self.assertRaises(ValueError):
                train_and_evaluate(csv)

if __name__ == '__main__':
    unittest.main(verbosity=2)
