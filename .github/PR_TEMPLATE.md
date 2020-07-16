---
name: 'Contribution Guideline '
about: Refer to Contirbution Guideline
title: ''
labels: ''
assignees: ''

---
### Contribution Guideline

Please send your PRs to `dev` branch if it is not directly related to a specific branch.
Before making a Pull Request, check your changes for basic mistakes and style problems by using a linter.
We have cardboardlinter setup in this repository, so for example, if you've made some changes and would like to run the linter on just the changed code, you can use the follow command:

```bash
pip install pylint cardboardlint
cardboardlinter --refspec master
```