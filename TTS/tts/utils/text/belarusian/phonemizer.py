import os

finder = None


def init():
    try:
        import jpype
        import jpype.imports
    except ModuleNotFoundError:
        raise ModuleNotFoundError("Belarusian phonemizer requires to install module 'jpype1' manually. Try `pip install jpype1`.")

    try:
        jar_path = os.environ["BEL_FANETYKA_JAR"]
    except KeyError:
        raise KeyError("You need to define 'BEL_FANETYKA_JAR' environment variable as path to the fanetyka.jar file")

    jpype.startJVM(classpath=[jar_path])

    # import the Java modules
    from org.alex73.korpus.base import GrammarDB2, GrammarFinder

    grammar_db = GrammarDB2.initializeFromJar()
    global finder
    finder = GrammarFinder(grammar_db)


def belarusian_text_to_phonemes(text: str) -> str:
    # Initialize only on first run
    if finder is None:
        init()

    from org.alex73.fanetyka.impl import FanetykaText
    return str(FanetykaText(finder, text).ipa)
