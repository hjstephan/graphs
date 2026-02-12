# Tag Creation and Branch Cleanup Instructions

## Purpose
This document provides instructions for creating the v2.0.0 tag and deleting merged branches.

## Automated Workflow

A GitHub Actions workflow has been created to automate this process. To use it:

1. Go to the GitHub repository page
2. Click on the "Actions" tab
3. Select "Create Tag v2.0.0 and Delete Merged Branches" from the workflows list
4. Click "Run workflow"
5. Choose the options:
   - **Create tag v2.0.0**: Select 'true' to create the tag
   - **Delete merged branches**: Select 'true' to delete merged branches
6. Click "Run workflow" to execute

## Manual Instructions

If you prefer to do this manually using the command line:

### Creating the v2.0.0 Tag

```bash
# Ensure you're on the main branch
git checkout main
git pull origin main

# Create the annotated tag
git tag -a v2.0.0 -m "Release v2.0.0"

# Push the tag to the remote repository
git push origin v2.0.0
```

### Deleting Merged Branches

The following branches have been merged into main and can be safely deleted:

- `copilot/add-information-processing-direction` (merged via PR #1)
- `copilot/update-brain-information-processing` (merged via PR #2)
- `copilot/update-science-documents` (merged via PR #3)
- `copilot/update-graphs-document` (merged via PR #4)
- `copilot/update-chapter-11-and-release` (merged via PR #6)

To delete these branches:

```bash
# Delete branches one by one
git push origin --delete copilot/add-information-processing-direction
git push origin --delete copilot/update-brain-information-processing
git push origin --delete copilot/update-science-documents
git push origin --delete copilot/update-graphs-document
git push origin --delete copilot/update-chapter-11-and-release
```

Or delete all at once:

```bash
git push origin --delete \
  copilot/add-information-processing-direction \
  copilot/update-brain-information-processing \
  copilot/update-science-documents \
  copilot/update-graphs-document \
  copilot/update-chapter-11-and-release
```

## Verification

After completion, verify:

1. Tag v2.0.0 exists:
   ```bash
   git fetch --tags
   git tag -l
   ```

2. Branches have been deleted:
   ```bash
   git fetch --prune
   git branch -r
   ```
