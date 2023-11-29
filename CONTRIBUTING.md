# Contribution guidelines

Welcome to the 🐸TTS!

This repository is governed by [the Contributor Covenant Code of Conduct](https://github.com/coqui-ai/TTS/blob/main/CODE_OF_CONDUCT.md).

## Where to start.
We welcome everyone who likes to contribute to 🐸TTS.

You can contribute not only with code but with bug reports, comments, questions, answers, or just a simple tweet to spread the word.

If you like to contribute code, squash a bug but if you don't know where to start, here are some pointers.

- [Development Road Map](https://github.com/coqui-ai/TTS/issues/378)

    You can pick something out of our road map. We keep the progess of the project in this simple issue thread. It has new model proposals or developmental updates etc.

- [Github Issues Tracker](https://github.com/coqui-ai/TTS/issues)

    This is a place to find feature requests, bugs.

    Issues with the ```good first issue``` tag are good place for beginners to take on.

- ✨**PR**✨ [pages](https://github.com/coqui-ai/TTS/pulls) with the ```🚀new version``` tag.

    We list all the target improvements for the next version. You can pick one of them and start contributing.

- Also feel free to suggest new features, ideas and models. We're always open for new things.

## Call for sharing language models
If possible, please consider sharing your pre-trained models in any language (if the licences allow for you to do so). We will include them in our model catalogue for public use and give the proper attribution, whether it be your name, company, website or any other source specified.

This model can be shared in two ways:
1. Share the model files with us and we serve them with the next 🐸 TTS release.
2. Upload your models on GDrive and share the link.

Models are served under `.models.json` file and any model is available under TTS CLI or Server end points.

Either way you choose, please make sure you send the models [here](https://github.com/coqui-ai/TTS/discussions/930).

## Sending a ✨**PR**✨

If you have a new feature, a model to implement, or a bug to squash, go ahead and send a ✨**PR**✨.
Please use the following steps to send a ✨**PR**✨.
Let us know if you encounter a problem along the way.

The following steps are tested on an Ubuntu system.

1. Fork 🐸TTS[https://github.com/coqui-ai/TTS] by clicking the fork button at the top right corner of the project page.

2. Clone 🐸TTS and add the main repo as a new remote named ```upsteam```.

    ```bash
    $ git clone git@github.com:<your Github name>/TTS.git
    $ cd TTS
    $ git remote add upstream https://github.com/coqui-ai/TTS.git
    ```

3. Install 🐸TTS for development.

    ```bash
    $ make system-deps  # intended to be used on Ubuntu (Debian). Let us know if you have a different OS.
    $ make install
    ```

4. Create a new branch with an informative name for your goal.

    ```bash
    $ git checkout -b an_informative_name_for_my_branch
    ```

5. Implement your changes on your new branch.

6. Explain your code using [Google Style](https://google.github.io/styleguide/pyguide.html#381-docstrings) docstrings.

7. Add your tests to our test suite under ```tests```  folder. It is important to show that your code works, edge cases are considered, and inform others about the intended use.

8. Run the tests to see how your updates work with the rest of the project. You can repeat this step multiple times as you implement your changes to make sure you are on the right direction.

    ```bash
    $ make test  # stop at the first error
    $ make test_all  # run all the tests, report all the errors
    ```

9. Format your code. We use ```black``` for code and ```isort``` for ```import``` formatting.

    ```bash
    $ make style
    ```

10. Run the linter and correct the issues raised. We use ```pylint``` for linting.  It helps to enforce a coding standard, offers simple refactoring suggestions.

    ```bash
    $ make lint
    ```

11. When things are good, add new files and commit your changes.

    ```bash
    $ git add my_file1.py my_file2.py ...
    $ git commit
    ```

    It's a good practice to regularly sync your local copy of the project with the upstream code to keep up with the recent updates.

    ```bash
    $ git fetch upstream
    $ git rebase upstream/master
    # or for the development version
    $ git rebase upstream/dev
    ```

12. Send a PR to ```dev``` branch.

    Push your branch to your fork.

    ```bash
    $ git push -u origin an_informative_name_for_my_branch
    ```

    Then go to your fork's Github page and click on 'Pull request' to send your ✨**PR**✨.

    Please set ✨**PR**✨'s target branch to ```dev``` as we use ```dev``` to work on the next version.

13. Let's discuss until it is perfect. 💪

    We might ask you for certain changes that would appear in the ✨**PR**✨'s page under 🐸TTS[https://github.com/coqui-ai/TTS/pulls].

14. Once things look perfect, We merge it to the ```dev``` branch and make it ready for the next version.

## Development in Docker container

If you prefer working within a Docker container as your development environment, you can do the following:

1. Fork 🐸TTS[https://github.com/coqui-ai/TTS] by clicking the fork button at the top right corner of the project page.

2. Clone 🐸TTS and add the main repo as a new remote named ```upsteam```.

    ```bash
    $ git clone git@github.com:<your Github name>/TTS.git
    $ cd TTS
    $ git remote add upstream https://github.com/coqui-ai/TTS.git
    ```

3. Build the Docker Image as your development environment (it installs all of the dependencies for you):

    ```
    docker build --tag=tts-dev:latest -f .\dockerfiles\Dockerfile.dev .
    ```

4. Run the container with GPU support:

    ```
    docker run -it --gpus all tts-dev:latest /bin/bash
    ```

Feel free to ping us at any step you need help using our communication channels.

If you are new to Github or open-source contribution, These are good resources.

- [Github Docs](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/proposing-changes-to-your-work-with-pull-requests)
- [First-Contribution](https://github.com/firstcontributions/first-contributions)
