# Hugging Face Spaces Deployment - Quick Reference

## 🚀 One-Command Deployment

```bash
# Login once
huggingface-cli login

# Deploy
bash scripts/deployment/deploy_to_huggingface.sh YOUR_USERNAME ds-banking-assistant
```

## 📋 Pre-Deployment Checklist

- [ ] Hugging Face account created
- [ ] HF CLI installed: `pip install huggingface_hub`
- [ ] Logged in: `huggingface-cli login`
- [ ] Git LFS installed: `git lfs version`
- [ ] Space created at [huggingface.co/new-space](https://huggingface.co/new-space)
- [ ] Models uploaded to Dropbox `/ds_project_models/`

## 🔑 Required Secrets

Add these to Space Settings → Variables and secrets:

```toml
OPENAI_API_KEY = "sk-..."
OPENAI_MODEL = "gpt-4o-mini"
GITHUB_TOKEN = "github_pat_..."
GITHUB_REPO = "ksauka/hicxai-data-private"
DROPBOX_APP_KEY = "..."
DROPBOX_APP_SECRET = "..."
DROPBOX_REFRESH_TOKEN = "..."
```

## 📁 Files Created

- `scripts/deployment/deploy_to_huggingface.sh` - Deployment script
- `runtime.txt` - Python version (3.10)
- `packages.txt` - System dependencies (none needed)
- `.gitattributes` - Git LFS configuration for models
- `README_HUGGINGFACE.md` - Space README template
- `docs/HUGGINGFACE_DEPLOYMENT.md` - Full deployment guide

## 🔄 Update Deployment

```bash
# After code changes, redeploy:
bash scripts/deployment/deploy_to_huggingface.sh YOUR_USERNAME ds-banking-assistant
```

## 🌐 Space URLs

After deployment:
- **Space page**: `https://huggingface.co/spaces/YOUR_USERNAME/ds-banking-assistant`
- **Direct app**: `https://YOUR_USERNAME-ds-banking-assistant.hf.space`
- **Embed**: Add `/embed` to direct URL for iframe

## 🎯 For Research Studies

### Anonymous Deployment:
1. Create organization at [huggingface.co/organizations/new](https://huggingface.co/organizations/new)
2. Deploy: `bash scripts/deployment/deploy_to_huggingface.sh your-org ds-assistant`
3. Make space private (Settings → Visibility)
4. Share embed URL with participants

### Professional Setup:
- Disable discussions (Settings → Discussions)
- Use neutral space name
- Private visibility
- No personal branding

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Build fails | Check Logs tab, verify requirements.txt |
| Models not loading | Check Dropbox secrets, verify files in `/ds_project_models/` |
| Session logging fails | Verify GitHub token has `repo` permission |
| Space timing out | Large models - use Dropbox instead of Git LFS |

## 📊 Monitoring

1. **Logs tab** - Real-time build and runtime logs
2. **GitHub repo** - Session files appear after participant completes
3. **Dropbox** - Check model download count (if API provides)

## ⚡ Quick Test

```bash
# Test locally first
streamlit run app_main_hicxai.py

# Then deploy
bash scripts/deployment/deploy_to_huggingface.sh YOUR_USERNAME test-space
```

## 📚 Full Documentation

See [docs/HUGGINGFACE_DEPLOYMENT.md](docs/HUGGINGFACE_DEPLOYMENT.md) for:
- Detailed setup instructions
- Multiple conditions deployment
- Advanced configuration
- Production checklist
- Alternative platforms

## 🆘 Support

- **Hugging Face Docs**: [huggingface.co/docs/hub/spaces](https://huggingface.co/docs/hub/spaces)
- **Discord**: [hf.co/join/discord](https://hf.co/join/discord)
- **GitHub Issues**: For app-specific problems

---

**Pattern from**: [anthrokit/XAIagent/scripts/deployment](https://github.com/ksauka/anthrokit/tree/main/XAIagent/scripts/deployment)
