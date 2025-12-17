# Git Setup and Push Guide

## Step 1: Initialize Git Repository
```bash
cd /Users/jamessukanto/Desktop/codes/projs/rag-multimodal
git init
```

## Step 2: Add All Files to Git
```bash
git add .
```

## Step 3: Create Initial Commit
```bash
git commit -m "Initial commit"
```

## Step 4: Create a Remote Repository
You need to create a repository on GitHub, GitLab, or another git hosting service:

**For GitHub:**
1. Go to https://github.com/new
2. Create a new repository (don't initialize with README, .gitignore, or license)
3. Copy the repository URL (e.g., `https://github.com/username/repo-name.git`)

**For GitLab:**
1. Go to https://gitlab.com/projects/new
2. Create a new project
3. Copy the repository URL

## Step 5: Add Remote Repository
```bash
# Replace with your actual repository URL
git remote add origin https://github.com/username/rag-multimodal.git
```

## Step 6: Push to Remote
```bash
# Push to main branch (or master if your default branch is master)
git branch -M main
git push -u origin main
```

If your default branch is `master` instead:
```bash
git push -u origin master
```

---

## Quick Reference - All Commands in Order

```bash
# Navigate to project
cd /Users/jamessukanto/Desktop/codes/projs/rag-multimodal

# Initialize git
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit"

# Add remote (replace with your actual URL)
git remote add origin https://github.com/username/rag-multimodal.git

# Push to remote
git branch -M main
git push -u origin main
```

---

## Troubleshooting

**If you get authentication errors:**
- For HTTPS: You may need to use a Personal Access Token instead of password
- For SSH: Set up SSH keys and use SSH URL instead: `git@github.com:username/repo.git`

**If you want to check status:**
```bash
git status
```

**If you want to see what will be committed:**
```bash
git status
```

**If you need to remove the remote and re-add:**
```bash
git remote remove origin
git remote add origin <your-url>
```

