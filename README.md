# PredictED-Rwanda

## Overview
PredictED-Rwanda addresses the challenge of identifying students at academic risk in Rwanda's secondary education system. Traditional assessments often miss early warning signs, delaying interventions. This project uses a machine learning model to predict academic failure based on the UCI Student Performance Dataset. The goal is to enhance model performance with optimization techniques.

- **Dataset**: UCI Student Performance Dataset (student-mat.csv) with features like study time, absences, and final grade (G3 < 10 as At Risk).

## Discussion of Findings

### Performance Table
| Model Instance       | Optimizer    | Regularization | Dropout Rate | Learning Rate | Accuracy      | Precision     | Recall        | F1 Score     | ROC AUC       | Loss           |
|-----------------------|--------------|----------------|--------------|---------------|---------------|---------------|---------------|--------------|---------------|----------------|
| Instance 1 (Base)    | Default      | None           | 0.0          | 0.001         | 0.6333        | 0.2857        | 0.1053        | 0.1538       | 0.6483        | 0.6188         |
| Instance 2           | Adam         | L2 (0.01)      | 0.3          | 0.001         | 0.6833        | 0.5000        | 0.1053        | 0.1739       | 0.6714        | 0.5810         |
| Instance 3           | RMSprop      | L1 (0.001)     | 0.4          | 0.0005        | 0.6667        | 0.4000        | 0.1053        | 0.1667       | 0.6650        | 0.5870         |
| Instance 4 (Best)    | Adamax       | L2 (0.005)     | 0.2          | 0.002         | 0.7167        | 0.6250        | 0.2632        | 0.3704       | 0.6739        | 0.5921         |

- **Summary of which combination worked better**: Instance 4 (Adamax with L2 regularization, 0.2 dropout, and 0.002 learning rate) performed the best, achieving the highest accuracy of 0.7167, a recall of 0.2632, and an F1 score of 0.3704. This combination outperformed others by balancing regularization and optimization, improving the model's ability to identify at-risk students compared to the base model and other optimized instances.

- **Discussion of which implementation worked better**: The Neural Network implementation, particularly Instance 4, emerged as the best performer after correcting the feature set to exclude G1 and G2, addressing initial label leakage. With features like age, studytime, absences, freetime, goout, and the encoded sex, Instance 4’s deeper architecture ([64, 32, 16]) and Adamax optimizer with a 0.002 learning rate effectively modeled the data’s non-linear patterns. The L2 regularization (0.005) and 0.2 dropout rate reduced overfitting, as evidenced by a lower loss (0.5921) compared to Instance 1 (0.6188), while the recall (0.2632) and F1 score (0.3704) improved over other instances. Instance 2 (Adam with L2 0.01) and Instance 3 (RMSprop with L1 0.001) showed moderate gains (accuracy 0.6833 and 0.6667), but their higher dropout (0.3 and 0.4) and lower learning rates limited recall. Compared to a hypothetical linear model, the Neural Network’s flexibility with these hyperparameters provided a better trade-off, though the low recall across all instances suggests a class imbalance or limited predictive power in the current feature set, warranting further feature engineering.

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
- `video_presentation.mp4`: https://youtu.be/5j7lrC6OJis
