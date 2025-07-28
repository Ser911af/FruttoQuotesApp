# Price Analysis Project

This project is designed to analyze supplier pricing and sales data using Python. It leverages various libraries to clean, visualize, and model the data for better decision-making regarding supplier selection and pricing strategies.

## Project Structure

```
price-analysis
├── .venv                     # Virtual environment for dependency isolation
├── data                      # Directory containing CSV data files
│   ├── Market - Consolidado(Consolidado).csv  # Historical supplier quotes
│   └── reportSalesFrutto_2025-07-24.csv        # Actual sales data
├── recomendador_proveedores.py  # Script for loading and analyzing supplier quotes
├── app_dashboard.py             # Streamlit app for user interface and data visualization
├── modelo_precio.py             # Optional script for training a price prediction model
├── requirements.txt             # List of project dependencies
└── README.md                    # Project documentation
```

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd price-analysis
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**:
   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```

4. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Scripts

- To run the supplier recommendation script:
  ```bash
  python recomendador_proveedores.py
  ```

- To launch the Streamlit dashboard:
  ```bash
  streamlit run app_dashboard.py
  ```

- To train the price prediction model (optional):
  ```bash
  python modelo_precio.py
  ```

## Usage

- Use `recomendador_proveedores.py` to analyze supplier quotes and find the best vendors based on product and city.
- The `app_dashboard.py` provides an interactive interface to visualize data and make informed decisions.
- The optional `modelo_precio.py` allows for predictive modeling of product prices based on historical data.

## License

This project is licensed under the MIT License - see the LICENSE file for details.