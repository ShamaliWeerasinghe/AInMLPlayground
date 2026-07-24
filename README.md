# Industrial Robot Predictive Maintenance

## 1. Developing the ML Model
The training pipeline is implemented in `ml/random_forest/train_and_deploy.py` and uses the sensor dataset at `Data/industrial_robot_sensor_data.txt`. The script performs preprocessing, train/test split, RandomForest training, and evaluation in one run. Model artifacts are exported to `ml_models/` as `random_forest_failure_predictor.joblib` and `random_forest_metrics.json`.

## 2. Exposing the Created Model as a FastAPI
The API is implemented in `api/fastapi/ml_api.py` and loads the trained model from `ml_models/random_forest_failure_predictor.joblib`. It exposes `/predict` for failure prediction and `/health` for service checks. Input payload fields match the same sensor columns used during model training.

## 3. Exposing the Same Model as a Streamlit Web App
The Streamlit app is implemented in `webapp/streamlit_app.py` and loads the same trained model from `ml_models/random_forest_failure_predictor.joblib`. It provides an interactive UI where you enter sensor values and get failure probability with a maintenance recommendation. This offers a quick demo interface for non-API users while keeping model logic consistent with FastAPI.

## How To Use the FastAPI
1. Install API dependencies from `requirements/api.txt`:

```bash
py -m pip install -r requirements/api.txt
```

2. Run the API from project root:

```bash
py -m uvicorn api.fastapi.ml_api:app --host 0.0.0.0 --port 8000 --reload
```

3. Test with Swagger UI:
- Open `http://127.0.0.1:8000/docs`
- Use `POST /predict` with a JSON payload such as:

```json
{
	"timestamp": "2026-01-20 10:30:00",
	"robot_id": "RB-101",
	"operating_hours": 320.5,
	"temp_c": 81.4,
	"vibration_mm_s": 5.2,
	"motor_current_a": 20.3,
	"hydraulic_pressure_bar": 101.1,
	"ambient_humidity_pct": 48.0,
	"spindle_rpm": 1498.0,
	"input_voltage_v": 400.9,
	"error_code": "E17",
	"maintenance_due_days": 6
}
```
## How To Use the Streamlit Web App
1. Install app dependencies from `requirements/app.txt` (run from project root):

```bash
py -m pip install -r requirements/app.txt
```

2. Start the web app:

```bash
py -m streamlit run webapp/streamlit_app.py
```

3. Open the app in your browser:
- `http://localhost:8501`

4. In the UI:
- Enter the robot sensor inputs
- Click `Predict Failure Risk`
- Review the failure probability and recommendation shown on screen

## Docker Commands (FastAPI and Streamlit)
1. Log in to Docker Hub:

```bash
docker login
```

2. Build the FastAPI image:

```bash
docker build -f docker/DockerFile.api -t ainml-fastapi:latest .
```

3. Build the Streamlit image:

```bash
docker build -f docker/DockerFile.webapp -t ainml-streamlit:latest .
```

4. Push the FastAPI image to Docker Hub:

```bash
docker tag ainml-fastapi:latest <your-dockerhub-username>/ainml-fastapi:latest
docker push <your-dockerhub-username>/ainml-fastapi:latest
```

5. Push the Streamlit image to Docker Hub:

```bash
docker tag ainml-streamlit:latest <your-dockerhub-username>/ainml-streamlit:latest
docker push <your-dockerhub-username>/ainml-streamlit:latest
```

6. Run the FastAPI container:

```bash
docker run --rm -p 8000:8000 ainml-fastapi:latest
```

7. Run the Streamlit container:

```bash
docker run --rm -p 8501:8501 ainml-streamlit:latest
```

8. Open the apps:
- FastAPI docs: `http://localhost:8000/docs`
- Streamlit UI: `http://localhost:8501`