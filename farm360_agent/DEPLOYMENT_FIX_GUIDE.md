# 🚀 Farm360 Deployment Guide - Real AI Configuration

## ⚠️ CRITICAL: Enable Real AI Responses

### Current Issue
The agent returns fallback responses like:
> "⚠️ **AI Mode Not Enabled** - Real AI responses require a valid GOOGLE_API_KEY"

This means the GOOGLE_API_KEY is not configured.

---

## 🔧 FIX STEPS

### 1️⃣ Get Your Google Gemini API Key

1. Visit: https://aistudio.google.com/apikey
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Copy the generated key (looks like: `AIzaSy...`)

---

### 2️⃣ Local Development Setup

#### Option A: Using .env file (Recommended)

Edit your `.env` file in the project root:

```bash
GOOGLE_API_KEY=AIzaSyYourActualKeyHere
FARM360_API_KEY=your_custom_secret_key
```

#### Option B: Environment variable

**Windows PowerShell:**
```powershell
$env:GOOGLE_API_KEY="AIzaSyYourActualKeyHere"
python farm360_agent/app.py
```

**Linux/Mac:**
```bash
export GOOGLE_API_KEY="AIzaSyYourActualKeyHere"
python farm360_agent/app.py
```

---

### 3️⃣ Render Deployment Setup

#### Add Environment Variable in Render Dashboard:

1. Go to your Render dashboard
2. Select your Farm360 service
3. Click **"Environment"** tab
4. Click **"Add Environment Variable"**
5. Add:
   - **Key:** `GOOGLE_API_KEY`
   - **Value:** `AIzaSyYourActualKeyHere`
6. Click **"Save Changes"**
7. Redeploy the service

**Screenshot Guide:**
```
Render Dashboard → Your Service → Environment → Add Variable
┌─────────────────────────────────────┐
│ Key:    GOOGLE_API_KEY              │
│ Value:  AIzaSy... (your actual key) │
│         [Save Changes]              │
└─────────────────────────────────────┘
```

---

## ✅ VERIFICATION

### Run Diagnostic Tool

```bash
cd farm360_agent
python diagnose_api_key.py
```

**Expected Output (Success):**
```
🔍 FARM360 API KEY DIAGNOSTIC TOOL
================================================================================

1️⃣ Checking GOOGLE_API_KEY environment variable...
   ✅ Found in environment: AIzaSy...XYZ12
   Length: 39 characters

2️⃣ Checking .env file...
   ✅ .env file found at: C:\...\ml models\.env
   ✅ GOOGLE_API_KEY entry exists in .env
   ✅ Value is set (not placeholder): AIzaSy...XYZ12

3️⃣ Checking config.py loading...
   ✅ Loaded in settings: AIzaSy...XYZ12
   Length: 39 characters

4️⃣ Testing Farm360Agent initialization...
   Importing agent... ✅
   Initializing with use_mock_llm=False...
   [DEBUG] ✅ Real AI responses ENABLED - Using Gemini model
   ✅ SUCCESS! Real AI responses ENABLED
   Status: has_llm = True

5️⃣ Testing actual query...
   Query: 'What is crop yield forecasting?'
   
   Response preview (first 200 chars):
   --------------------------------------------------------------------------------
   ### Summary
   🌾 **Crop Yield Analysis**: I'll help you optimize agricultural production...
   --------------------------------------------------------------------------------
   
   ✅ Response appears to be from real AI

================================================================================
✅ ALL CHECKS PASSED!
🎉 Real AI responses are properly configured!
================================================================================
```

---

## 🚨 TROUBLESHOOTING

### Issue: "No valid GOOGLE_API_KEY found"

**Cause:** API key not loaded or still using placeholder

**Fix:**
1. Ensure `.env` file has actual key (not placeholder text)
2. Restart your Python application
3. Check if pydantic-settings is installed: `pip install pydantic-settings`

### Issue: "GenAI configuration failed"

**Cause:** Invalid API key or network issue

**Fix:**
1. Verify API key is correct (should start with `AIzaSy`)
2. Check internet connection
3. Ensure Google Gemini API is accessible in your region

### Issue: Works locally but not on Render

**Cause:** Environment variable not set in Render

**Fix:**
1. Follow Section 3️⃣ above to add env var in Render dashboard
2. Redeploy after adding variable
3. Check Render logs for confirmation

---

## 📊 RESPONSE BEHAVIOR

### With Valid API Key (Real AI):
```markdown
### Summary
🌾 **Crop Yield Analysis**: Based on agricultural data and predictive modeling...

### Analysis
📊 **Yield Forecast**: The model predicts approximately 4.2 tons/hectare...

### Recommendations
🎯 **Optimization Strategies**:
- Consider soil testing for precision agriculture
- Monitor weather patterns for irrigation planning
...
```

### Without API Key (Fallback Mode):
```markdown
⚠️ **AI Mode Not Enabled**

Real AI responses require a valid GOOGLE_API_KEY. Currently using local ML models only.

### Summary
🌾 **Crop Yield Analysis**: I'll help you optimize Rice production using predictive modeling.

### Analysis
📊 **Yield Forecast**: [Local model prediction]...
```

---

## 🔒 SECURITY BEST PRACTICES

### DO:
✅ Store API keys in environment variables  
✅ Use `.env.example` as template (without actual keys)  
✅ Add `.env` to `.gitignore`  
✅ Rotate keys periodically  
✅ Use different keys for dev/production  

### DON'T:
❌ Commit actual API keys to Git  
❌ Hardcode keys in source files  
❌ Share keys publicly  
❌ Use same key across multiple projects  

---

## 📝 QUICK REFERENCE

### File Locations:
- **Config:** `farm360_agent/config.py`
- **Main Logic:** `farm360_agent/main.py`
- **Environment:** `.env` (project root)
- **Diagnostic:** `farm360_agent/diagnose_api_key.py`

### Key Validation:
```python
from farm360_agent.config import settings

print(f"API Key Status: {'SET' if settings.google_api_key else 'NOT SET'}")
print(f"API Key Length: {len(settings.google_api_key) if settings.google_api_key else 0}")
```

### Agent Initialization:
```python
from farm360_agent.main import Farm360Agent

# For real AI responses (requires API key)
agent = Farm360Agent(use_mock_llm=False)

# For testing without API key
agent = Farm360Agent(use_mock_llm=True)
```

---

## 🎯 EXPECTED RESULTS

After proper configuration:

| Feature | Before | After |
|---------|--------|-------|
| **Response Source** | Local ML models | Google Gemini AI + Tools |
| **Response Quality** | Good | Excellent & Dynamic |
| **Tool Usage** | Available | ✅ Active (yield, dairy, disease, weather) |
| **Context Awareness** | Basic | Advanced with memory |
| **Error Messages** | Generic | Specific & Helpful |
| **User Experience** | Static | Conversational & Engaging |

---

## 💡 PRO TIPS

1. **Test Locally First:** Verify AI works before deploying
2. **Monitor Usage:** Check Google Cloud Console for API usage
3. **Set Budget Alerts:** Avoid unexpected charges
4. **Log Verification:** Check debug logs for confirmation
5. **Fallback Ready:** System still works with local models if AI fails

---

## 🆘 NEED HELP?

### Diagnostic Commands:

```bash
# Check environment variable
echo $env:GOOGLE_API_KEY  # Windows PowerShell
echo $GOOGLE_API_KEY      # Linux/Mac

# Run diagnostic tool
python farm360_agent/diagnose_api_key.py

# Check logs
tail -f farm360_agent/logs/*.log
```

### Common Error Patterns:

| Error | Meaning | Solution |
|-------|---------|----------|
| `settings.google_api_key is None` | Not loaded from env | Check .env file format |
| `GenAI configuration failed` | Invalid key or network | Verify key & connectivity |
| `has_llm = False` | LLM not initialized | Check steps 1-3 above |

---

## ✅ SUCCESS CHECKLIST

- [ ] API key obtained from Google AI Studio
- [ ] Key added to `.env` file (not placeholder)
- [ ] Application restarted after setting key
- [ ] Diagnostic tool shows all green checkmarks
- [ ] Test query returns AI-generated response
- [ ] Render environment variable set (if deploying)
- [ ] Logs show "GenAI LLM Orchestrator Configured Successfully"

**Once all boxes checked, you're ready to go! 🎉**

---

## 🔄 DEPLOYMENT WORKFLOW

### Local Development → Render Production

1. **Develop Locally:**
   ```bash
   # Set key locally
   $env:GOOGLE_API_KEY="AIzaSy..."
   
   # Test
   python farm360_agent/diagnose_api_key.py
   
   # Run app
   python farm360_agent/app.py
   ```

2. **Deploy to Render:**
   - Push code to GitHub
   - Add `GOOGLE_API_KEY` to Render environment
   - Trigger redeploy
   - Monitor logs for success

3. **Verify Production:**
   - Access your Render URL
   - Test a query
   - Confirm AI responses (not fallback)

---

**🚀 Your Farm360 agent is now powered by real AI!**
