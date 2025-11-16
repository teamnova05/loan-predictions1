# robust build_preprocessor that works with different sklearn versions
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import inspect

def _onehot_encoder_compat(**kwargs):
    """
    Return a OneHotEncoder compatible with your installed sklearn version.
    Newer sklearn uses 'sparse_output'. Older uses 'sparse'.
    """
    onehot_init_sig = inspect.signature(OneHotEncoder)
    params = onehot_init_sig.parameters

    if 'sparse' in params:
        # older sklearn
        if 'sparse_output' in kwargs:
            kwargs['sparse'] = kwargs.pop('sparse_output')
    else:
        # newer sklearn
        if 'sparse' in kwargs:
            kwargs['sparse_output'] = kwargs.pop('sparse')

    return OneHotEncoder(**kwargs)

def build_preprocessor(numeric_cols, categorical_cols):
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', _onehot_encoder_compat(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    return preprocessor
