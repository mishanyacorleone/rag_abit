from threading import local
_thread_locals = local()

def set_retrieved_context(ctx):
    _thread_locals.retrieved_context = ctx

def get_retrieved_context():
    return getattr(_thread_locals, 'retrieved_context', None)

def clear_retrieved_context():
    _thread_locals.retrieved_context = None

    