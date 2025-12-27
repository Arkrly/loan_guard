# Contributing to LoanGuard

Thank you for your interest in contributing! This project is primarily a portfolio/learning project, but contributions are welcome.

## 🚀 Getting Started

1. **Fork the repository**
2. **Clone your fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/loan_risk_sys.git
   cd loan_risk_sys
   ```

3. **Set up the development environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # Linux/Mac
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   uvicorn src.app:app --reload
   ```

## 📝 Code Style

- **Python**: Follow PEP 8 guidelines
- **JavaScript**: Use modern ES6+ syntax
- **HTML/CSS**: Use Tailwind CSS utility classes

## 🔧 Development Guidelines

### Backend (Python/FastAPI)
- Use type hints for function parameters
- Add docstrings to functions
- Handle exceptions gracefully
- Use Pydantic for data validation

### Frontend (HTML/JS/Tailwind)
- Use semantic HTML elements
- Keep JavaScript modular
- Follow the existing glassmorphism design
- Test on multiple screen sizes

## 📦 Pull Request Process

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes
3. Test thoroughly
4. Commit with clear messages: `git commit -m "feat: add new feature"`
5. Push to your fork: `git push origin feature/your-feature`
6. Open a Pull Request

### Commit Message Format
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation
- `style:` - Formatting, no code change
- `refactor:` - Code restructuring
- `test:` - Adding tests

## 🐛 Reporting Issues

Please include:
- Description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Screenshots if applicable
- Environment details

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.
