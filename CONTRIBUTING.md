# Contributing to LoanGuard

Thank you for your interest in contributing to LoanGuard! We welcome all contributions.

---

## 🌿 Branch Structure

This repository uses a **two-branch workflow**:

| Branch | Purpose |
|--------|---------|
| `master` | **Production** — Clean, deployment-ready code only |
| `dev` | **Development** — Full workspace with models, datasets, notebooks |

> ⚠️ **Important:** All development work should be done on the `dev` branch. Never push development changes directly to `master`.

---

## 🚀 Getting Started

### 1. Fork & Clone

```bash
git clone https://github.com/YOUR_USERNAME/loan_guard.git
cd loan_guard
```

### 2. Switch to the Dev Branch

```bash
git checkout dev
```

The `dev` branch contains:
- 📓 Jupyter notebooks for experimentation
- 📊 Datasets and processed data
- 🤖 All ML model files
- 📖 Development documentation (`DEVELOPMENT.md`)

### 3. Set Up Environment

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

### 4. Run Locally

```bash
uvicorn src.app:app --reload --host 127.0.0.1 --port 8080
```

---

## 📝 Development Guidelines

### Code Style

- **Python**: Follow PEP 8 guidelines
- **JavaScript**: Use modern ES6+ syntax
- **HTML/CSS**: Use Tailwind CSS utility classes

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

---

## 📦 Pull Request Process

### 1. Create a Feature Branch (from dev)

```bash
git checkout dev
git pull origin dev
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

- Write clean, documented code
- Test your changes locally
- Update documentation if needed

### 3. Commit with Clear Messages

```bash
git commit -m "feat: add new feature"
```

**Commit Prefixes:**
| Prefix | Usage |
|--------|-------|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `docs:` | Documentation |
| `style:` | Formatting (no code change) |
| `refactor:` | Code restructuring |
| `test:` | Adding tests |

### 4. Push & Open PR

```bash
git push origin feature/your-feature-name
```

Then open a **Pull Request to the `dev` branch** (not master).

---

## 🐛 Reporting Issues

When reporting issues, please include:

- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Screenshots (if applicable)
- Your environment details (OS, Python version)

---

## 📚 Resources

- [DEVELOPMENT.md](https://github.com/Arkrly/loan_guard/blob/dev/DEVELOPMENT.md) — Full development guide (on dev branch)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [scikit-learn Docs](https://scikit-learn.org/)

---

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

<p align="center">
  Thank you for helping make LoanGuard better! 🙌
</p>
