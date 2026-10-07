# Event-Aware Predictive Auto-Scaling for E-Commerce Platforms

## Run

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python data\generate_data.py
python app.py
```
Open http://127.0.0.1:5000

## Concept
The system automatically checks a calendar of upcoming e-commerce sales periods such as Weekend Sales, Summer Sale, Diwali Sale, Mega Sale, Black Friday, Christmas and New Year. It combines that calendar context with historical workload data and a Random Forest model to forecast CPU demand. A predictive scaling policy then estimates how many cloud instances would be required.

The project is a cloud auto-scaling prototype/simulator; it does not claim to provision real cloud infrastructure.
