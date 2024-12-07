import joblib

MODELS_CACHE_DIR='./cache'
DEFAULT_NAME = ""

def hash_model(model):
    return hash(str(model)) % 10000

def common_name(hash_model, name):
    return f'{name}-{hash_model}.pkl'

def load(model, name=DEFAULT_NAME):
    hash_id = hash_model(model)
    return joblib.load(f'{MODELS_CACHE_DIR}/{common_name(hash_id, name)}')

def save(model, name=DEFAULT_NAME):
    hash_id = hash_model(model)
    joblib.dump(model, f'{MODELS_CACHE_DIR}/{common_name(hash_id, name)}')

def load_or_fit(model, x, y, name = DEFAULT_NAME):
    fitted = None
    try:
        fitted = load(model, name)
    except FileNotFoundError:
        fitted = model.fit(x,y)
        save(fitted, name)
    return fitted
