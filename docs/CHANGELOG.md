# 📝 Changelog

All notable changes to the LoanIQ project are documented in this file.

---

## [1.0.0] - 2024-12-27

### 🎉 Initial Release

Complete end-to-end loan risk prediction system with modern UI.

---

## [1.1.0] - 2024-12-27

### 🔒 Security & Validation

#### Added
- **Pydantic Validation** - Comprehensive input validation using Literal types and Field constraints
- **Total Income Validator** - Custom model validator ensures at least one income source
- **Environment-based CORS** - CORS origins configurable via `ALLOWED_ORIGINS` environment variable
- **Generic Error Messages** - API errors no longer leak internal implementation details

#### Fixed
- **CRITICAL**: Zero income (0+0) no longer causes infinity errors
- **CRITICAL**: Negative income values no longer produce NaN crashes  
- **CRITICAL**: Invalid categorical values (e.g., Gender='Other') are now properly rejected

---

### 🎨 Frontend Improvements

#### Added
- **Toast Notifications** - User-friendly error/success messages replace browser alerts
- **SEO Meta Tags** - Description, keywords, Open Graph, Twitter cards
- **Emoji Favicon** - 🧠 brain icon as favicon
- **Footer** - Version info, API docs link, health check link
- **ARIA Labels** - Improved accessibility for screen readers
- **Form Validation CSS** - Visual feedback for invalid inputs

#### Fixed
- **Color Contrast** - Muted text color improved for WCAG compliance (#606070 → #8a8a9a)
- **Dynamic Model Version** - No longer hardcoded, fetched from API response

---

### 🔧 Backend Improvements

#### Added
- **Health Check Endpoint** (`/health`) - Returns API and model status
- **Response Models** - Typed Pydantic response schemas
- **Comprehensive Logging** - Info, warning, error logs throughout

#### Changed
- **Lifespan Context Manager** - Replaced deprecated `@app.on_event("startup")`
- **API Title** - Updated to "LoanIQ - Smart Risk Intelligence API"

---

### 📚 Documentation

#### Added
- **CHANGELOG.md** - This file
- **Updated README.md** - Professional portfolio-focused documentation
- **ASCII Preview** - Visual representation of the UI

#### Removed
- Outdated project structure references

---

## Validation Rules Summary

| Field | Type | Constraints |
|-------|------|-------------|
| Gender | Literal | "Male" \| "Female" |
| Married | Literal | "Yes" \| "No" |
| Dependents | Literal | "0" \| "1" \| "2" \| "3+" |
| Education | Literal | "Graduate" \| "Not Graduate" |
| Self_Employed | Literal | "Yes" \| "No" |
| ApplicantIncome | float | ≥ 0 |
| CoapplicantIncome | float | ≥ 0 |
| LoanAmount | float | > 0, ≤ 10000 |
| Loan_Amount_Term | float | 12 - 480 |
| Credit_History | Literal | 0.0 \| 1.0 |
| Property_Area | Literal | "Urban" \| "Rural" \| "Semiurban" |
| **Total Income** | custom | ApplicantIncome + CoapplicantIncome > 0 |

---

## Files Changed

### v1.1.0

| File | Changes |
|------|---------|
| `src/app.py` | Validation, lifespan, health endpoint, logging |
| `frontend/index.html` | SEO, ARIA, favicon, footer |
| `frontend/style.css` | Toast styles, footer, form validation, color contrast |
| `frontend/app.js` | Toast system, health check, error mapping |
| `README.md` | Complete rewrite |
| `docs/CHANGELOG.md` | Created |

---

## Testing Verification

All validation tests pass:

```
✅ Valid request → Returns prediction
✅ Gender='Other' → 422 "Input should be 'Male' or 'Female'"
✅ Income=0+0 → 422 "Total income must be greater than 0"
✅ Income=-100 → 422 "greater than or equal to 0"
✅ LoanAmount=0 → 422 "greater than 0"
✅ Dependents='5' → 422 Literal error
✅ Credit_History=0.5 → 422 Literal error
```
