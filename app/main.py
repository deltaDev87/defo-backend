from fastapi import FastAPI
from app.routes import router
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://deforestation-detector-app.vercel.app"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)

@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Backend Status</title>
        <style>
            body {
                margin: 0;
                height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                background-color: #000000;
                font-family: Arial, sans-serif;
                color: white;
            }
            h1 {
                font-size: 2rem;
            }
        </style>
    </head>
    <body>
        <h1>🚀 Backend Running at http://127.0.0.1:8000/</h1>
    </body>
    </html>

    """
