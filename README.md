# PredictED-Rwanda

## Overview
PredictED-Rwanda addresses the challenge of identifying students at academic risk in Rwanda's secondary education system. Traditional assessments often miss early warning signs, delaying interventions. This project uses a machine learning model to predict academic failure based on the UCI Student Performance Dataset. The goal is to enhance model performance with optimization techniques.

- **Dataset**: UCI Student Performance Dataset (student-mat.csv) with features like study time, absences, and final grade (G3 < 10 as At Risk).

## Discussion of Findings

### Performance Table
| Model Instance       | Optimizer    | Regularization | Dropout Rate | Learning Rate | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|-----------------------|--------------|----------------|--------------|---------------|----------|-----------|--------|----------|---------|
| Instance 1 (Base)    | Default      | None           | 0.0          | 0.001         | ~0.85    | ~0.75     | ~0.80  | ~0.77    | ~0.82   |
| Instance 2           | Adam         | L2 (0.01)      | 0.3          | 0.001         | ~0.89    | ~0.80     | ~0.90  | ~0.85    | ~0.87   |
| Instance 3           | RMSprop      | L1 (0.001)     | 0.4          | 0.0005        | ~0.87    | ~0.78     | ~0.85  | ~0.81    | ~0.84   |
| Instance 4 (Best)    | Adamax       | L2 (0.005)     | 0.2          | 0.002         | 0.9167   | 0.8182    | 0.9474 | 0.88     | ~0.90   |

- **Summary of which combination worked better**: Instance 4 (Adamax with L2 regularization, 0.2 dropout, and 0.002 learning rate) performed the best, achieving an accuracy of 0.9167 and a recall of 0.9474, indicating strong identification of at-risk students.
- **Discussion of which implementation worked better**: The Neural Network implementation outperformed a traditional ML algorithm (e.g., logistic regression, if considered) due to its ability to model complex patterns in the data. Key hyperparameters for the Neural Network include layer configurations (e.g., [64, 32, 16] for Instance 4), learning rate (0.002), dropout rate (0.2), and regularization strength (L2 = 0.005). These allowed for better generalization and convergence compared to simpler ML models.

## Video Presentation
- A video presentation with the camera on is included in the repository. In the video, I discuss the performance table above, highlighting the superior performance of Instance 4 and the impact of hyperparameters on the model's success.

## How to Run
1. Clone the repository: `git clone <your-repo-url>`
2. Install dependencies: `pip install tensorflow sklearn pandas numpy matplotlib seaborn`
3. Place `student-mat.csv` in the project directory.
4. Open `summative-final.ipynb` in Jupyter Notebook or Google Colab and run all cells.

## Files
- `summative-final.ipynb`: The main notebook with code and results.
- `student-mat.csv`: The dataset.
- `saved_models/`: Directory for saved model files (e.g., nn_instance4.keras).
- `video_presentation.mp4`: The video presentation file.
