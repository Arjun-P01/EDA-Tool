# AI-Powered EDA Tool

An interactive Exploratory Data Analysis tool for the UCI Adult Income dataset, powered by GPT-4o-mini. Built with Streamlit, containerized with Docker, and deployed to Azure Container Apps via GitHub Actions CI.

**Live demo:** https://eda-tool.proudbush-b75f2f95.eastus2.azurecontainerapps.io

---

## Features

- Automated dataset loading from UCI ML Repository (32,561 records)
- 6 interactive visualizations: age distribution, income class split, education vs income, hours per week, income by sex, top occupations
- Correlation heatmap across all numerical features
- GPT-4o-mini AI summary — generates a 3-paragraph data-driven analysis on demand

## Tech Stack

| Layer | Technology |
|-------|-----------|
| App | Python, Streamlit |
| AI | OpenAI GPT-4o-mini |
| Data | Pandas, Matplotlib, Seaborn |
| Container | Docker |
| Registry | Azure Container Registry |
| Hosting | Azure Container Apps |
| CI | GitHub Actions |

## Architecture

```
GitHub push
    │
    ▼
GitHub Actions CI
    │  docker build (verifies image builds)
    │
    ▼
Azure Container Registry
    │  edatoolregistry.azurecr.io/eda-tool:latest
    │
    ▼
Azure Container Apps
    │  public HTTPS endpoint
    │
    ▼
User browser → Streamlit app → OpenAI API
```

## Run Locally

**Without Docker:**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
echo "OPENAI_API_KEY=your_key_here" > .env
streamlit run app.py
```

**With Docker:**
```bash
docker build -t eda-tool .
docker run -p 8501:8501 -e OPENAI_API_KEY=your_key_here eda-tool
```

Open http://localhost:8501

## CI/CD

Every push to `main` triggers a GitHub Actions workflow (`.github/workflows/ci.yml`) that builds the Docker image to verify no build errors were introduced.

To deploy a new version:
```bash
docker build --platform linux/amd64 -t edatoolregistry.azurecr.io/eda-tool:latest .
docker push edatoolregistry.azurecr.io/eda-tool:latest
az containerapp update --name eda-tool --resource-group eda-tool-rg --image edatoolregistry.azurecr.io/eda-tool:latest
```

## Dataset

UCI Adult Income Dataset — predicts whether income exceeds $50K/year based on census data.
- 32,561 training records, 15 features
- Source: https://archive.ics.uci.edu/ml/datasets/adult
