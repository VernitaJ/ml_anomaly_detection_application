# Prototype App for Prediction-Based Anomaly Detection

This repository contains the prototype app developed as part of a thesis project on prediction-based anomaly detection. The app aims to demonstrate the suitability of an anomaly detection system through a functional implementation.

## Treatment Implementation

The author chose a prototyping pattern as the evaluation method to enhance the scientific rigor of the study. This method, described by Sonnenberg et al., involves implementing an artifact design as a generic solution to showcase the artifact's suitability [23]. The implementation follows the workflow artifact described in Section V, resulting in a functional smart anomaly detection system.

### 1) Requirements

Five participants contributed to the system requirements by selecting the most significant qualities. These qualities were further refined into specific non-functional requirements (NFRs) by the author and the stakeholder closely involved with the smart anomaly detection system. Since the implementation was experimental, the focus was on NFRs. The table below outlines the qualities and resulting NFRs:

| Quality    | Non-functional Requirement                             |
|------------|------------------------------------------------------|
| Correctness | 1. The system should maintain a false positive rate of less than 5%. (False Positive Rate) |
| Performance | 2. The system should detect anomalies within a maximum response time of 500 milliseconds. (Latency) |
| Availability | 3. The system should have a maximum allowable downtime of 1 hour per month. (Downtime) |
| Modifiability | 4. The system should allow for the retraining of an ML model within a day. (ML Model update) |

### 2) System Design

The system design took into account the functional requirements, although the implementation being experimental, the author relied on data obtained during the study. The design incorporated modules identified from the literature review, such as Data, Machine Learning, Anomaly Detection, and Notification modules. Additionally, the system design considered the environment at ESS, including the Target wheel, control system, and PVs. For a detailed view of the system design, refer to Appendix E.

### 3) Data Analysis

The development phase began with data analysis. Data from the Target wheel assembly underwent various methods of analysis, including data visualizations, Fast Fourier Transformations (FFT), and metric comparisons between original and mutated data (standard deviation, mean, and mean absolute deviation - MAD). The results of the data analysis, including visualizations, can be found in Appendix E.

### 4) Data Processing

During this phase, the data for training the machine learning model was prepared. This involved data cleaning and creating training and testing datasets with input samples and timesteps. The Python code used to create these datasets can be found in Appendix E. Additionally, a data stream component was created using Kafka to simulate the Control system environment at ESS, generating and consuming data for the Data Module's prediction processing.

### 5) Machine Learning Model

The ML model design, selection, training, and evaluation were performed in this phase. The author chose the LSTM model based on its proven accuracy in time-series based predictions [14], [34]. The model was implemented using Tensorflow, and its summary can be found in Appendix E. The data from the Data Processing phase was used for training, validation, and testing, and the process was repeated until satisfactory metrics were obtained. The resulting model was saved for predicting future data points, and the metrics were saved for anomaly detection calculations during production (see Appendix E - 4).

### 6) Anomaly Classification

The selection of sliding window sizes was guided by the visualizations from the Data Analysis phase. The velocity graph showed a repetitive pattern every four minutes, leading to the selection of sliding windows of 4 and 8 minutes. Additionally, a sliding window of 1 minute was chosen to ensure stability in velocity. For a visualization of the sliding window of 4 minutes indicating anomalous behavior, refer to Fig 12 in the prototype user interface.

### 7) Testing

For this phase, the author focused on the results from the Requirements phase to limitthe experiment to the expected qualities of the system. Since this implementation is a prototype, extensive testing was not performed. To simulate the implementation of an anomaly detection system, the non-functional requirements were translated into Python tests, which can be found in Appendix E. Please note that these tests were considered valid for the author to accept this phase as completed.

### 8) Deployment

As the system is a prototype, the deployment was simulated using Docker containers. Users can run the prototype system by utilizing the Docker containers included in the implementation, such as the data stream, hosted ML models, and data processing, machine learning, anomaly detection, and notification modules. Essentially, the prototype implementation simulates a distributed anomaly detection system.

### 9) Maintenance

To finalize the system implementation and simulate a maintenance task, the author utilized the functionality on the UI to retrain the ML model for the velocity dataset. A new dataset was uploaded, and the model underwent retraining. The output on the UI displayed the current model's accuracy metrics compared to the retrained ML model. The retrained model showed lower performance, leading to the decision not to replace the current model. For the output of the UI, refer to Fig 13.

Since Maintenance was the last phase in the system, and with no dependencies, the author assumes that this phase will be continuously considered. Therefore, the simulation implementation phase was completed.


