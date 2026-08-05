"""Registration engine: elastix parameters, initialization, execution, application.

Import from the modules directly (``from regix.registration.engine import
ElastixEngine``): this package deliberately re-exports nothing, so that pulling in
one module does not import the six others -- ``convexadam`` in particular reaches
for torch.
"""
