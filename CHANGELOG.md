# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2024-12-27

### Added
- Modern glassmorphism UI with Tailwind CSS
- Animated loading screen with progress bar
- Interactive loan term slider with preset buttons
- Self-employed toggle button
- History sidebar with session persistence
- Toast notifications for user feedback
- Risk score circular progress animation
- Responsive design for all screen sizes

### Changed
- Updated currency to INR (₹) for Indian market
- Redesigned result panel with modern minimal look
- Improved form layout with compact styling
- Enhanced input validation

### Fixed
- Form submission handling
- DOM initialization order
- API health check with fallback

## [1.0.0] - 2024-12-23

### Added
- Initial project setup
- FastAPI backend with prediction endpoint
- Logistic Regression model training pipeline
- SMOTE for handling imbalanced data
- Feature engineering with income ratios
- Basic HTML/CSS frontend
- Model persistence with joblib
- API documentation with Swagger UI
- Health check endpoint
- CORS configuration
- Environment variable support

### Technical Details
- Python 3.8+ compatibility
- scikit-learn 1.3+ for ML
- FastAPI 0.104+ for API
- Pydantic validation
- Static file serving
