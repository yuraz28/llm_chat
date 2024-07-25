import uvicorn
from fastapi import FastAPI
import routes
import os

app = FastAPI()

app.include_router(routes.router)

if __name__ == "__main__":
    uvicorn.run("server:app", host='0.0.0.0', port=os.getenv("PORT", default=8000), log_level="info")