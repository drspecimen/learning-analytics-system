# Project Backlog — AI-Driven Learning Analytics System

## Sprint 1: Requirements Analysis
- [x] Finalize functional requirements (Table 3.1)
- [x] Finalize non-functional requirements (Table 3.2)
- [x] Define Administrator role permissions
- [x] Define Lecturer role permissions

## Sprint 2: Database Design
- [ ] Design ERD schema (Figure 3.3)
- [ ] Create Users table
- [ ] Create Students table
- [ ] Create Datasets table
- [ ] Create Predictions table
- [ ] Create Model_Results table
- [ ] Create System_Logs table
- [ ] Write data-access layer functions

## Sprint 3: System Architecture / Backend
- [ ] Build Presentation Layer (Streamlit UI shell)
- [ ] Build Business Layer (auth, validation, routing)
- [ ] Build Machine Learning Layer (model service wrapper)
- [ ] Connect Database Layer to Business Layer
- [ ] Implement login/authentication

## Sprint 4: AI Model Development
- [ ] Acquire OULAD dataset
- [ ] Clean and preprocess data
- [ ] Encode categorical variables
- [ ] Normalize numerical features
- [ ] Engineer additional features (cumulative scores, VLE clicks)
- [ ] Split train/test data
- [ ] Train Logistic Regression
- [ ] Train Decision Tree
- [ ] Train Random Forest
- [ ] Perform hyperparameter tuning
- [ ] Evaluate models (Accuracy, Precision, Recall, F1, AUC-ROC)
- [ ] Select and serialize best model

## Sprint 5: Dashboard Development
- [ ] Build login page
- [ ] Build Administrator dashboard view
- [ ] Build Lecturer dashboard view
- [ ] Add Plotly charts and visualizations
- [ ] Add filters (module, period, performance category)
- [ ] Add report generation/export

## Sprint 6: Testing
- [ ] Unit test ML layer
- [ ] Integration test all four architecture layers
- [ ] Usability test dashboard
- [ ] Validate full flow against Figure 3.11 flowchart

## Sprint 7: Deployment
- [ ] Final code cleanup and commit
- [ ] Deploy to Streamlit Community Cloud (or chosen host)
- [ ] Tag release on GitHub
- [ ] Final documentation update