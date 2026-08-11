"""Registration engine: elastix parameters, initialization, execution, application.

Import from the modules directly (``from regix.registration.engine import
ElastixEngine``). Torch is imported only inside ConvexAdam/anatomix execution
functions, so importing the ordinary CPU pipeline remains lightweight.
"""
