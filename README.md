# Natural Language Processing Assigments:
## [Project Github Repository](https://github.com/kativenOG/NLP_assignments)
## Project Completion State:
- [x] First Assignment (in directory **1**)
- [ ] Second Assignment (in directory **2**)
- [ ] Third Assignment (in directory **3**)
## Installation:
Assignments are in the form of jupyter notebooks, they contain both the code and all the discussion on the project assigment and result.</br>
There are 3 ways to install the dependencies and run the notebooks on your local machine:
- Create a Conda enviroment in which all dependencies installed are defined inside a environment.yml file
```
git clone git@github.com:kativenOG/NLP_assignments.git
cd NLP_assignments
conda env create -f environment.yml
conda activate nlp
jupyter-notebook
```
- Install all the dependencies with no virtualization by running 
```
git clone git@github.com:kativenOG/NLP_assignments.git
cd NLP_assignments
python3 -m pip install -r requirements.txt
jupyter-notebook
```
- Inside a container by running the provided Dockerfile
