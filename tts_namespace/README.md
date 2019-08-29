This folder contains a symlink called TTS to the parent folder:

    lrwxr-xr-x  TTS -> ..

This is used to appease the distribute/setuptools gods. When the project was
initially set up, the repository folder itself was considered a namespace, and
development was done with `sys.path` hacks. This means if you tried to install
TTS, `setup.py` would see the packages `models`, `utils`, `layers`... instead of
 `TTS.models`, `TTS.utils`...

Installing TTS would then pollute the package namespace with generic names like
those above. In order to make things installable in both install and development
modes (`pip install /path/to/TTS` and `pip install -e /path/to/TTS`), we needed
to add an additional 'TTS' namespace to avoid this pollution. A virtual redirect
using `packages_dir` in `setup.py` is not enough because it breaks the editable
installation, which can only handle the simplest of `package_dir` redirects.

Our solution is to use a symlink in order to add the extra `TTS` namespace. In
`setup.py`, we only look for packages inside `tts_namespace` (this folder),
which contains a symlink called TTS pointing to the repository root. The final
result is that `setuptools.find_packages` will find `TTS.models`, `TTS.utils`...

With this hack, `pip install -e` will then add a symlink to the `tts_namespace`
in your `site-packages` folder, which works properly. It's important not to add
anything else in this folder because it will pollute the package namespace when
installing the project.

This does not work if you check out your project on a filesystem that does not
support symlinks.