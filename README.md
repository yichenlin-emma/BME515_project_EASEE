# BME515 project EASEE
### Intro to Git
##### To clone a repository to local computer
+ open command window
+ get into your desired folder
+ in Github, press green "Code" button, copy SSH link
+ in command window, enter ```git clone <SSH link>```

##### To pull code from gitHub (everytime before starting your new code, update the repository from GitHub)
+ ```git pull```: pull info from GitHub to local


##### Branch (new modification of code must be made in the branch rather than 'main')
+ ```git branch <new branch name>```: create a new branch in local
+ ```git branch```: check which branch you are in right now
+ ```git checkout <branch name>```: enter the branch  

##### To push code onto gitHub
+ ```git add <file name>```: add the file to the staging area (start using git to track file)
+ ```git commit -m "<commit message>"```: put file that git is tracking into repository
+ ```git push```: pushing info from local to GitHub

##### Merge repository in GitHub (after pushing your branch onto GitHub)
+ click "Pull requests"
+ click "Compare & pull requests"
+ click "create pull request"

##### Other useful command for Git
+ ```git status```: check the staging area, will appear which file is modified
+ ```git log```: history of commit made in repository
+ ```git log -- oneline```: brief histroy of commit made in repository
