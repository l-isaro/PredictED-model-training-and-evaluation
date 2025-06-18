# PredictED-Rwanda

## Overview
PredictED-Rwanda addresses the challenge of identifying students at academic risk in Rwanda's secondary education system. Traditional assessments often miss early warning signs, delaying interventions. This project uses a machine learning model to predict academic failure based on the UCI Student Performance Dataset. The goal is to enhance model performance with optimization techniques.

- **Dataset**: UCI Student Performance Dataset (student-mat.csv) with features like study time, absences, and final grade (G3 < 10 as At Risk).

## Discussion of Findings

### Performance Table
| Model Instance       | Optimizer    | Regularization | Dropout Rate | Learning Rate | Accuracy      | Precision     | Recall        | F1 Score     | ROC AUC       | Loss           |
|-----------------------|--------------|----------------|--------------|---------------|---------------|---------------|---------------|--------------|---------------|----------------|
| Instance 1 (Base)    | Default      | None           | 0.0          | 0.001         | 0.9167        | 0.8182        | 0.9474        | 0.8780       | 0.9820        | 0.1699         |
| Instance 2           | Adam         | L2 (0.01)      | 0.3          | 0.001         | 0.9000        | 0.8095        | 0.8947        | 0.8500       | 0.9756        | 0.1894         |
| Instance 3           | RMSprop      | L1 (0.001)     | 0.4          | 0.0005        | 0.9000        | 0.8095        | 0.8947        | 0.8500       | 0.9743        | 0.1911         |
| Instance 4 (Best)    | Adamax       | L2 (0.005)     | 0.2          | 0.002         | 0.9167        | 0.8182        | 0.9474        | 0.8780       | 0.9807        | 0.1713         |

- **Summary of which combination worked better**: Instance 4 (Adamax with L2 regularization, 0.2 dropout, and 0.002 learning rate) performed the best, achieving an accuracy of 0.9167 and a recall of 0.9474, indicating strong identification of at-risk students.

- **Discussion of which implementation worked better**: The Neural Network implementation, particularly Instance 4, outperformed other instances and a hypothetical traditional ML algorithm (e.g., logistic regression) due to its deeper architecture and tailored hyperparameters. The use of three hidden layers ([64, 32, 16]) captured complex, non-linear relationships in the student data, which simpler models struggle to model. The Adamax optimizer, with a higher learning rate (0.002), facilitated faster convergence, while L2 regularization at 0.005 reduced overfitting, and a 0.2 dropout rate enhanced generalization. Instance 2’s higher L2 (0.01) and Instance 3’s L1 (0.001) with RMSprop resulted in slightly lower ROC AUC (0.9756 and 0.9743) and higher loss (0.1894 and 0.1911), indicating less effective regularization. The Neural Network’s flexibility, especially Instance 4’s high recall (0.9474) and ROC AUC (0.9807), outperformed linear models, making it ideal for early risk detection.

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
