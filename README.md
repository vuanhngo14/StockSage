# FYP-23-S4-25
Stock market prediction 

## Using github 

### Moving between branches 
1. List out all branches
    git branch -a 

2. Move to a specific branch 
    git checkout <branch name>

### Get latest change 
1. For main branch 
    - Git fetch origin 
    - Git pull origin 

2. For sub-branches created, get from main branch 
    - Git fetch origin main (Main here means that we fetch from main branch)
    - Git pull origin main 

### Create a new branch 
- git checkout -b <branch name>

### Delete a branch
1. Remove locally 
    1.1 Basic delte
        - git branch -d <branch name>

    1.2 Branch got unmerged cahnges
        - git branch -D branch-name

2. Remote branch deletion 
    - git push origin --delete branch-name




