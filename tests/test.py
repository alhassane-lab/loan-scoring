import pytest
import sys
import os

from ..app.app import training_features


def test_training_features_():
    #loan_scoring_classifier, training_features, raw_data = load()
    assert len(training_features)==799
