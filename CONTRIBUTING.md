# Contributing Guide
This repo is a shared space.  
These guidelines is to help everyone collaborate smoothly while keeping authorship (and anything else) clear.  

## Where to Put Your Work
Suggested repo structure:
```
.
├── Tips/          # No subfolder asked in this file
│   ├── start-with-blackhole.md
│   └── github_and_VScode.md
├── exp/
│   ├── RF/        # Name of subfolder can be modi on ur own
│   ├── z0/
│   ├── (someone else)
│   └── aqing/                      
├── interface/
│   ├── RF/
│   ├── z0/
│   ├── (someone else)
│   └── aqing/                      
├── tutorial/
│   ├── RF/        
│   ├── z0/
│   ├── (someone else)
│   └── aqing/
├── whiteBoard.md 
├── CONTRIBUTING.md      
└── README.md
```
Each main folder may contain author-specific subfolders.  
If you are not sure where something should go, **starting in your own subfolder ;)**

## Modifying Existing Content
- Small changes: open branch and PR
- Larger changes: open an issue first

## Commit Messages
Please avoid commit messages like `update`, `test`, or `haha py`.  
A short, descriptive sentence helps others (and future you) understand what changed.

In general, try to use **clear and commonly understood language** when writing commit messages.

For example:
- `Fix typo in interface README`
- `Update installation steps for Llama2-7b`
- `Add notes for local LLM setup`

## Else
### Avoid Path Flatting
GitHub automatically flattens paths when a directory **only contains a single** sub-directory and **no other files**.  
To avoid that, we should always ensure a parent directory is displayed as an independent path, which means there shouldn't be only one folder in any parent folder (e.g. `tutorial/`, `interface/`)  
Adding an extra file to that directory can help :)  
The "eatra file" can be `thisFolder.md` or `.gitkeep`
