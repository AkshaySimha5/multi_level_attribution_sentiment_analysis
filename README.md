#  A Multi-Level Attribution Approach for Explainable Sentiment Analysis

### Build the Docker image
```docker build -t xai-container .```

### Run the analysis
```docker run --rm -p 8000:8000 -v ${PWD}\output:/app/output xai-container```

## Frontend
```streamlit run frontend.py```
