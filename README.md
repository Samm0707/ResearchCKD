# ResearchCKD

Related to my research on risk prediction on kidney disease using Deep learning.

Here’s the detailed structure of my project with all the steps and techniques used:

1. Data Loading and Preprocessing Libraries:
pandas, numpy, scipy, sklearn, tensorflow, seaborn, matplotlib
•	Loading Data:
Load the dataset from CSV using pandas.
•	Basic Exploration:
Check the first 5 and last 5 entries.
Display column names, data types, and statistics.
•	Handling Missing Values:
Check for missing values using isnull().sum().
•	Handling Duplicates:
Detect duplicates using duplicated().
•	Label Encoding:
Use binary labels for categorical columns such as Sex and disease history features.
•	Statistical Analysis:
Use chi2_contingency for association between categorical features.
chi2_contingency - This test is commonly used in research to explore relationships between 
categorical data, such as in medical studies, surveys, or social sciences, to see if there is a connection between two characteristics or behaviors.

2. Feature Selection
•	Dependent Variable:
Target column: EventCKD35 (Binary outcome: 0 for no CKD event, 1 for CKD event).
•	Independent Variables:
Variables related to patient demographics, medical history, medication use, and baseline measurements.
•	Feature Importance Calculation:
Neural Network Model (Sequential):
Use dense layers to estimate feature importance.
Extract importance values by analyzing the first layer’s weights.
•	Removing Low Importance Features:
Remove features with importance values below a specified threshold.

3. Data Balancing
Balancing Techniques: SMOTE (Synthetic Minority Oversampling Technique) to balance the dataset's class distribution for EventCKD35.

4. Train-Test Split
•	Method: Use train_test_split from sklearn.
•	Split Ratio: 80% for training, 20% for testing.
•	Stratified Sampling: Ensure the split maintains class distribution using stratify.

5. Normalization
•	Scaling: Use StandardScaler to normalize the dataset features.
•	LSTM Input Preparation: Reshape the training and testing sets for LSTM, converting them into a 3D array (samples, time steps, features).

6. Model Creation
•	Architecture:
LSTM with Attention Mechanism:
LSTM layers for sequence modeling.
Apply an Attention layer to focus on relevant time steps.
•	Layers:
1.	Input Layer: Reshape input for LSTM.
2.	LSTM Layer: 64 units with return_sequences=True.
3.	Attention Layer: Self-attention to weigh important time steps.
4.	Dense Layer: 32 units, followed by dropout for regularization.
5.	Output Layer: Sigmoid activation for binary classification.

•	Feedforward Neural Network (FNN): This model processes information in layers to make predictions based on input data. It’s straightforward and useful for many tasks.

•	Convolutional Neural Network (CNN): Primarily used for recognizing patterns, this model looks for important features in the data, like signs of kidney disease, by applying filters.

•	Long Short-Term Memory (LSTM): LSTMs are great for analyzing sequences, like patient health over time. They remember past information to understand how conditions develop.

•	CNN-LSTM Hybrid: This combines CNN and LSTM strengths. First, it finds important features, then it looks at how those features change over time, making it effective for complex data.

•	Attention-LSTM: Similar to LSTM, but it focuses on the most relevant parts of the data. This helps it make more accurate predictions by emphasizing important information.

7. Model Training
•	Optimizer: Use Adam optimizer with a learning rate of 0.001.
•	Loss Function: Binary cross-entropy for classification.
•	Batch Size & Epochs: Train the model for 500 epochs with a batch size of 32.
•	Validation Split: Use 20% of training data for validation.

1.	Optimizer (Adam): This is a tool that helps the model learn by adjusting the parameters (weights) to minimize errors. The Adam optimizer is efficient and adapts the learning speed based on the data, making it effective for many types of problems.
2.	Loss Function (Binary Cross-Entropy): This function measures how well the model’s predictions match the actual outcomes. In your case, it helps assess how accurately the model predicts kidney disease (yes or no). Lower values indicate better performance.
3.	Batch Size & Epochs: You trained the model with 32 examples at a time (batch size) and repeated this process 500 times (epochs) to improve its learning. This way, the model gradually gets better by seeing the data multiple times.
4.	Validation Split: By reserving 20% of the training data for validation, you can check how well the model performs on unseen data. This helps ensure that the model generalizes well and doesn’t just memorize the training data.

8. Model Evaluation
•	Prediction: Use the trained model to predict CKD risk.
•	Get probability values and convert them into binary classes (0 or 1).
•	Metrics: Precision, recall, F1-score from classification_report.
•	AUC-ROC Curve: Plot the ROC curve to evaluate the model's performance.
Compute the AUC score using roc_auc_score.

a)	Precision tells us how many of the predicted CKD cases were actually CKD. It’s important because it helps us understand how many of our positive predictions are correct.
b)	Recall measures how many actual CKD cases were correctly identified by the model. This is crucial for ensuring we don’t miss patients who really have CKD.
c)	F1-Score is the balance between precision and recall. It’s useful when we want to have a single measure that considers both false positives and false negatives.

AUC-ROC Curve: The ROC curve visualizes the model’s performance across different thresholds, showing the trade-off between sensitivity (true positive rate) and specificity (1 - false positive rate).
AUC (Area Under the Curve) quantifies the overall ability of the model to distinguish between the positive and negative classes. A higher AUC indicates better model performance. This is important for understanding how well the model can predict CKD compared to just random guessing.


9. Visualization
•	Histograms & KDE Plots: Visualize the distribution of numerical features using sns.histplot.
•	Correlation Matrix: Visualize feature correlation using sns.heatmap.
•	Feature Importance Visualization: Bar plot of feature importance scores.
•	ROC Curve: Plot and annotate the ROC curve to interpret the model’s performance.
