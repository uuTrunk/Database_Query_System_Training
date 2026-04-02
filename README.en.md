# Database Query System - Training (Concurrency Predicting & ML Workflow Node)

This segregated repository houses the **Machine Learning Concurrency Scheduling Microservice**, fully extracted from the former Controller hub. Currently maintained by an ultra-light **Django API**, it facilitates the frontend by scoring a textual prompt's generation survival chance along with providing untainted **offline sample resampling and retraining frameworks**.

## Core Roles & Architecture

This specialized module features two definitive components:

### 1. The Concurrency Expectation Service (Django Backend port `8001`)
- **`training_backend/`**: Django project settings routing the microservice.
- **`api/`**: Serves a single crucial `http://localhost:8001/api/predict` JSON inference URL.
- **Function**: Immediately intercepts the frontend `Controller`'s user string and feeds it to its localized regression model loaded from `saves/model_best.pth`. The pipeline subsequently projects an algorithm mapping the text features dynamically to return an advised concurrency threads value, typically constrained effectively between dimensions of 1 and 5. 

### 2. Output & Feature Production Scripts (Training Pipelines)
- **`training/`**: A suite of ML engineering offline pipeline flows. Features parameter collection, `gen_training_questions.py` tagging mechanisms, and the eventual model assembly function within (`model.py`).
- **Function**: Executes entirely independently without background ports. Instead calibrating locally from the latest interaction statistics produced globally, running the parameters recursively through an integrated BERT model architecture. The iteration terminates, producing updated checkpoints across `./saves()`.

## Running the Application

Everything herein dictates an invocation of the standardized python dependency suite located in the Conda `dqs` virtual configuration encompassing requirements for `PyTorch`, `Transformers` and `Django`.

### Initializing the Inference Microservice

Activate the corresponding workspace and utilize the re-badged server script (`main.py` overriding former configurations):
```bash
conda activate dqs
python main.py runserver 8001
```

### Full-Scale Regeneration or Offline Retraining Executions

As libraries and data endpoints like databases are exclusively hosted in `Database_Query_System_Agent` now, retrieving active queries algorithmically mandates access via their namespace imports dynamically loaded at script start. Ensure port `8000` responds accurately prior to batch invocations. 
```bash
conda activate dqs
# Step 1: Synthesize a newly populated set of ML testing constraints 
python -m training.gen_training_questions 
# Step 2: Establish base performance graphs logging requests physically querying the Agent stack
python -m training.test_ask_graph 
# Step 3: Train and cache the newest generation model weight `.pth` files autonomously
python -m training.model
```
Once run optimally, the generated `.pth` checkpoints replace historical files across `saves/`. Subsequent server restart triggers initialization sequences onto newly computed weights yielding precise front-end suggestions.
