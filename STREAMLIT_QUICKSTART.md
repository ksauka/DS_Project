# 🚀 Quick Start: Run the Streamlit App

Get the DS_Project Streamlit interface running in 2 minutes!

## Step 1: Install Streamlit Dependencies (1 minute)

```bash
cd /home/kudzai/projects/DS_Project

# Install Streamlit and visualization libraries
pip install streamlit>=1.28.0 plotly>=5.17.0 altair>=5.0.0 st-aggrid>=1.0.0

# Verify installation
streamlit --version
```

## Step 2: Set Up Environment (30 seconds)

```bash
# Create .env if it doesn't exist
touch .env

# Add your OpenAI API key (optional for now)
# nano .env
# Add: OPENAI_API_KEY=sk-your-key-here
```

## Step 3: Run the App (30 seconds)

```bash
# From DS_Project directory
streamlit run app_main.py
```

**Expected Output:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

## Step 4: Open in Browser

Click the provided URL or visit: **http://localhost:8501**

---

## 🎨 What You'll See

### Home Page
- Project overview
- Session status
- Quick start guide
- Example workflows

### Navigation (Sidebar)
- 🏠 Home
- 🎯 Train
- 📊 Evaluate
- 💬 User Study
- 📈 Analysis
- ⚙️ Settings

---

## 📝 Explore the Interface (2-3 minutes)

1. **Click through each page** to see the UI structure
2. **Try the Settings page** to configure datasets
3. **Check the Help expander** in the sidebar for tips
4. **Try file uploads** (uploads are prepared but backend not connected yet)

---

## 💡 What's Working Now

✅ Full UI navigation  
✅ Configuration interface  
✅ Session management (UI side)  
✅ File utilities (UI side)  
✅ Input validation functions  
✅ Output formatting functions  

❌ Actual model training (Phase 2)  
❌ Actual evaluation (Phase 2)  
❌ Real inference (Phase 2)  
❌ Real visualizations (Phase 2)  

---

## 🔧 Troubleshooting

### Port Already in Use
```bash
# Use different port
streamlit run app_main.py --server.port 8502
```

### Module Not Found Errors
```bash
# Check your working directory
pwd  # Should be /home/kudzai/projects/DS_Project

# Reinstall dependencies
pip install -r requirements.txt
pip install -r requirements-streamlit.txt
```

### Streamlit Cache Issues
```bash
# Clear cache
rm -rf ~/.streamlit/
streamlit run app_main.py
```

### Can't Access Remotely
Edit `.streamlit/config.toml`:
```toml
[server]
headless = false
address = "0.0.0.0"
```

---

## 📚 Documentation to Read

After running the app, check these files:

1. **[STREAMLIT_README.md](STREAMLIT_README.md)** - Complete user guide
2. **[STREAMLIT_IMPLEMENTATION_PLAN.md](STREAMLIT_IMPLEMENTATION_PLAN.md)** - Architecture & roadmap
3. **[STREAMLIT_INTEGRATION_GUIDE.md](STREAMLIT_INTEGRATION_GUIDE.md)** - Technical integration details
4. **[STREAMLIT_PHASE1_SUMMARY.md](STREAMLIT_PHASE1_SUMMARY.md)** - What was built

---

## ⏭️ Next Steps

### For Testing UI
1. Run `streamlit run app_main.py`
2. Navigate through all pages
3. Read documentation
4. Provide feedback on design

### For Full Implementation (Phase 2)
- See [STREAMLIT_INTEGRATION_GUIDE.md](STREAMLIT_INTEGRATION_GUIDE.md) for adapter implementation
- Follow checklist in [STREAMLIT_PHASE1_SUMMARY.md](STREAMLIT_PHASE1_SUMMARY.md)
- Connect to existing DS_Project components

---

## 🎯 Quick Test Checklist

Run through these to verify everything works:

- [ ] App starts without errors
- [ ] Home page loads and displays
- [ ] Sidebar navigation works (click all 6 pages)
- [ ] Settings page configurable
- [ ] Train page shows options
- [ ] Evaluate page has controls
- [ ] User Study has mode selection
- [ ] Analysis has tabs
- [ ] No console errors
- [ ] Session ID displayed in sidebar

---

## 💬 Feedback

The UI is Phase 1 and fully functional! 

**Current state:**
- All navigation works ✅
- All configuration options present ✅
- Input validation ready ✅
- File utilities prepared ✅
- Documentation complete ✅

**Next phase** will connect to actual:
- Model training
- Evaluation logic
- Inference engine
- Real visualizations

---

## 🔗 Quick Links

- **App Entry:** `app_main.py`
- **Pages:** `src/streamlit_app/pages/`
- **Components:** `src/streamlit_app/components/`
- **Config:** `.streamlit/config.toml`

---

## ✨ Tips

- **Dark Mode:** Settings → Theme (your browser)
- **Full Screen:** Press F for fullscreen charts
- **Responsive:** Works on desktop and tablet
- **Fast:** Uses browser caching for performance

---

**That's it!** 🎉

Your Streamlit interface is now running. Enjoy exploring!

For questions, see the documentation files listed above.
