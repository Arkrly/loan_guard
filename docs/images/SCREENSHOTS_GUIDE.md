# Adding Screenshots to README

## Instructions

The README references two screenshots that need to be added:

1. **`docs/images/dashboard.png`** - Main dashboard/form view
2. **`docs/images/result.png`** - Result panel after prediction

## How to Capture Screenshots

1. Start the application:
   ```bash
   uvicorn src.app:app --host 127.0.0.1 --port 8080
   ```

2. Open http://127.0.0.1:8080 in your browser

3. **Dashboard Screenshot:**
   - Take a screenshot of the main form view
   - Crop to show the form nicely
   - Save as `docs/images/dashboard.png`

4. **Result Screenshot:**
   - Fill in sample data and click "Analyze Risk"
   - Take a screenshot showing the result panel
   - Save as `docs/images/result.png`

## Recommended Dimensions

- **Dashboard:** ~1200x800px
- **Result:** ~600x600px or full screen showing result

## Tips

- Use dark mode in browser for consistency
- Capture at high resolution for clarity
- Crop any browser chrome if needed
