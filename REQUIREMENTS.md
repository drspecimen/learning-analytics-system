# Requirements — AI-Driven Learning Analytics System

## Functional Requirements (User Stories)

### Authentication
- [ ] As an Administrator, I can log in with a username and password so that I can access system management features.
- [ ] As a Lecturer, I can log in with a username and password so that I can access my student data and predictions.
- [ ] As any user, I am logged out automatically after a period of inactivity.

### Student Data Management
- [ ] As a Lecturer, I can upload a student dataset (CSV) so that it can be processed by the system.
- [ ] As a Lecturer, I can edit individual student records so that data stays accurate.
- [ ] As an Administrator, I can view and manage all datasets uploaded across the system.

### Performance Prediction
- [ ] As a Lecturer, I can run a prediction on a student or dataset so that I can see their likely academic outcome.
- [ ] As a Lecturer, I can see a confidence score alongside each prediction so that I can judge reliability.
- [ ] As an Administrator, I can trigger model retraining when new data is available.

### Dashboard Visualization
- [ ] As a Lecturer, I can view a dashboard of interactive charts summarizing student performance trends.
- [ ] As an Administrator, I can view system-wide KPIs (total students, prediction accuracy, active users).
- [ ] As any user, I can filter dashboard data by module, presentation/session, or performance category.

### Report Generation
- [ ] As a Lecturer, I can generate a downloadable report (PDF/CSV) of predictions for my students.
- [ ] As an Administrator, I can export system-wide reports for record-keeping.

### Model Comparison
- [ ] As an Administrator, I can view a comparison of Logistic Regression, Decision Tree, and Random Forest performance (Accuracy, Precision, Recall, F1) so that I can confirm the best model is in use.

## Non-Functional Requirements (Acceptance Criteria)

- [ ] **Usability**: A new user can complete login → view dashboard within 2 minutes without training.
- [ ] **Performance**: Predictions return within 3 seconds for a single student; dashboard loads within 5 seconds.
- [ ] **Reliability**: System handles invalid file uploads gracefully (clear error message, no crash).
- [ ] **Security**: Passwords are hashed (never stored in plain text); role-based access is enforced on every page.
- [ ] **Scalability**: System remains responsive with at least 1,000+ student records loaded.
- [ ] **Maintainability**: Code is organized into separate modules per layer (Presentation, Business, ML, Database) so components can be updated independently.

## Role Definitions (per Figure 3.7)

### Administrator — full system access
- [ ] Manage user accounts (create, edit, deactivate Lecturer/Admin accounts)
- [ ] Manage all datasets across the system
- [ ] Trigger and monitor model retraining
- [ ] View system-wide dashboard and logs
- [ ] Manage system settings
- [ ] Generate system-wide reports

### Lecturer — scoped to their own students
- [ ] Upload and manage their own student records
- [ ] Run predictions on their students
- [ ] View their own dashboard (their students/modules only)
- [ ] Generate reports for their own students
- [ ] Cannot manage other users or system-wide settings