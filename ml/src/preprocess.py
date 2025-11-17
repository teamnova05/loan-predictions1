# ml/src/preprocess.py
"""
Robust preprocessor builder and helper utilities.
"""

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import inspect
import numpy as np

def _onehot_encoder_compat(**kwargs):
    """
    Return a OneHotEncoder compatible with installed sklearn version.
    Newer sklearn uses 'sparse_output'. Older uses 'sparse'.
    """
    onehot_init_sig = inspect.signature(OneHotEncoder)
    params = onehot_init_sig.parameters

    # translate kwarg names according to installed sklearn
    if 'sparse' in params:
        # older sklearn: support sparse_output by mapping to sparse
        if 'sparse_output' in kwargs:
            kwargs['sparse'] = kwargs.pop('sparse_output')
    else:
        # newer sklearn: map 'sparse' -> 'sparse_output' if provided
        if 'sparse' in kwargs:
            kwargs['sparse_output'] = kwargs.pop('sparse')

    return OneHotEncoder(**kwargs)


def build_preprocessor(numeric_cols, categorical_cols):
    """
    Build a ColumnTransformer with numeric and categorical pipelines.
    numeric_cols: list of numeric column names
    categorical_cols: list of categorical column names
    Returns: ColumnTransformer
    """
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        # create OneHotEncoder compatible with multiple sklearn versions
        ('onehot', _onehot_encoder_compat(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'  # explicit and safe
    )
    return preprocessor


def get_feature_names_from_preprocessor(preprocessor, numeric_cols=None, categorical_cols=None):
    """
    Return a list of output feature names for the ColumnTransformer `preprocessor`.
    If raw column lists are provided (numeric_cols, categorical_cols) they will be used as fallbacks.
    Works across sklearn versions and handles OneHotEncoder variants.
    """
    feature_names = []

    # ColumnTransformer -> .transformers is list of (name, transformer, cols)
    try:
        transformers = preprocessor.transformers
    except Exception:
        # Not a ColumnTransformer: fallback
        return None

    for name, transformer, cols in transformers:
        # skip dropped columns or remainder info
        if name == 'remainder' and transformer == 'drop':
            continue
        if transformer == 'drop':
            continue

        # If transformer is a Pipeline, pick the last step (e.g., OneHotEncoder)
        transformer_obj = transformer
        if hasattr(transformer, 'named_steps'):
            # take final step
            last_step = list(transformer.named_steps.values())[-1]
            transformer_obj = last_step

        # Attempt to call get_feature_names_out with input feature names when possible
        try:
            if hasattr(transformer_obj, 'get_feature_names_out'):
                # Some versions accept the original column names as an argument.
                try:
                    out = transformer_obj.get_feature_names_out(cols)
                except Exception:
                    # fallback: call without args
                    out = transformer_obj.get_feature_names_out()
                # convert to plain strings
                feature_names.extend([str(x) for x in out])
                continue
        except Exception:
            # fall through to fallback handling
            pass

        # Fallback: if cols is a list/tuple, just append them
        if isinstance(cols, (list, tuple, np.ndarray)):
            feature_names.extend([str(c) for c in cols])
        else:
            # Could be slice or integer indices - attempt to use provided numeric/categorical lists
            if numeric_cols and isinstance(cols, slice):
                feature_names.extend([str(c) for c in numeric_cols])
            elif categorical_cols and isinstance(cols, slice):
                feature_names.extend([str(c) for c in categorical_cols])
            else:
                feature_names.append(str(cols))

    # Final verification: if names are empty but numeric/categorical lists provided, use them
    if (not feature_names) and (numeric_cols or categorical_cols):
        feature_names = []
        if numeric_cols:
            feature_names.extend([str(c) for c in numeric_cols])
        if categorical_cols:
            feature_names.extend([str(c) for c in categorical_cols])

    # Ensure it's a plain Python list
    return list(feature_names)
