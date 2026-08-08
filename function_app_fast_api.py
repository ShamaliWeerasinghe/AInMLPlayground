import azure.functions as func

# Import the FastAPI application object from your API module.
# This is the app that contains routes like /health and /predict.
from api.fastapi.ml_api import app as fastapi_app

# Create an Azure Functions ASGI app wrapper.
# This automatically creates the HTTP trigger plumbing needed
# to route incoming Function App HTTP requests to FastAPI.
app = func.AsgiFunctionApp(
    app=fastapi_app,
    # Change to FUNCTION or ADMIN if you want key-based auth.
    http_auth_level=func.AuthLevel.ANONYMOUS,
)
