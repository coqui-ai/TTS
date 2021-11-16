import re

# List of (regular expression, replacement) pairs for abbreviations in french:
abbreviations_fr = [
    (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
    for x in [
        ("M", "monsieur"),
        ("Mlle", "mademoiselle"),
        ("Mlles", "mesdemoiselles"),
        ("Mme", "Madame"),
        ("Mmes", "Mesdames"),
        ("N.B", "nota bene"),
        ("M", "monsieur"),
        ("p.c.q", "parce que"),
        ("Pr", "professeur"),
        ("qqch", "quelque chose"),
        ("rdv", "rendez-vous"),
        ("max", "maximum"),
        ("min", "minimum"),
        ("no", "numéro"),
        ("adr", "adresse"),
        ("dr", "docteur"),
        ("st", "saint"),
        ("co", "companie"),
        ("jr", "junior"),
        ("sgt", "sergent"),
        ("capt", "capitain"),
        ("col", "colonel"),
        ("av", "avenue"),
        ("av. J.-C", "avant Jésus-Christ"),
        ("apr. J.-C", "après Jésus-Christ"),
        ("art", "article"),
        ("boul", "boulevard"),
        ("c.-à-d", "c’est-à-dire"),
        ("etc", "et cetera"),
        ("ex", "exemple"),
        ("excl", "exclusivement"),
        ("boul", "boulevard"),
    ]
] + [
    (re.compile("\\b%s" % x[0]), x[1])
    for x in [
        ("Mlle", "mademoiselle"),
        ("Mlles", "mesdemoiselles"),
        ("Mme", "Madame"),
        ("Mmes", "Mesdames"),
    ]
]
