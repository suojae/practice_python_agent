from fastapi import FastAPI

app = FastAPI(title="Practice Python Agent", version="1.0.0")

@app.get("/")
def read_root():
    return {"message": "Hello from Practice Python Agent!"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}
