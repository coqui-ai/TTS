# Contribution guidelines

This repository is governed by Mozilla's code of conduct and etiquette guidelines. For more details, please read the [Mozilla Community Participation Guidelines](https://www.mozilla.org/about/governance/policies/participation/).

Before making a Pull Request, check your changes for basic mistakes and style problems by using a linter. We have cardboardlinter setup in this repository, so for example, if you've made some changes and would like to run the linter on just the differences between your work and master, you can use the follow command:

```bash
pip install pylint cardboardlint
cardboardlinter --refspec master
```

This will compare the code against master and run the linter on all the changes. To run it automatically as a git pre-commit hook, you can do do the following:

```bash
cat <<\EOF > .git/hooks/pre-commit
#!/bin/bash
if [ ! -x "$(command -v cardboardlinter)" ]; then
    exit 0
fi

# First, stash index and work dir, keeping only the
# to-be-committed changes in the working directory.
echo "Stashing working tree changes..." 1>&2
old_stash=$(git rev-parse -q --verify refs/stash)
git stash save -q --keep-index
new_stash=$(git rev-parse -q --verify refs/stash)

# If there were no changes (e.g., `--amend` or `--allow-empty`)
# then nothing was stashed, and we should skip everything,
# including the tests themselves.  (Presumably the tests passed
# on the previous commit, so there is no need to re-run them.)
if [ "$old_stash" = "$new_stash" ]; then
    echo "No changes, skipping lint." 1>&2
    exit 0
fi

# Run tests
cardboardlinter --refspec HEAD -n auto
status=$?

# Restore changes
echo "Restoring working tree changes..." 1>&2
git reset --hard -q && git stash apply --index -q && git stash drop -q

# Exit with status from test-run: nonzero prevents commit
exit $status
EOF
chmod +x .git/hooks/pre-commit
```

This will run the linters on just the changes made in your commit.