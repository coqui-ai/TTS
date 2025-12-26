import re

_END = r"(?=\s|$|[,\.;:\)\]\}\!?])"


def _c(pattern: str) -> re.Pattern:
    """Compile a case-insensitive regex."""
    return re.compile(pattern, re.IGNORECASE)


abbreviations_it = [
    # --- Courtesy / correspondence
    (_c(r"\bspett\.?\s*(?:le\.?)?\b" + _END), "spettabile"),
    (_c(r"\begr\.?\b" + _END), "egregio"),
    (_c(r"\bgent\.?\s*mo\.?\b" + _END), "gentilissimo"),
    (_c(r"\bgent\.?\s*ma\.?\b" + _END), "gentilissima"),
    (_c(r"\bill\.?\s*mo\.?\b" + _END), "illustrissimo"),
    (_c(r"\bill\.?\s*ma\.?\b" + _END), "illustrissima"),
    (_c(r"\batt\.?\b" + _END), "attenzione"),
    (_c(r"\bc\.?\s*a\.?\b" + _END), "cortese attenzione"),  # often "alla c.a."
    # --- People / titles / professions
    (_c(r"\bsig\.?\b" + _END), "signor"),
    (_c(r"\bsig\.?\s*ra\.?\b" + _END), "signora"),
    (_c(r"\bsig\.?\s*na\.?\b" + _END), "signorina"),
    (_c(r"\bsigg\.?\b" + _END), "signori"),
    (_c(r"\bdott\.?\b" + _END), "dottore"),
    (_c(r"\bdott\.?\s*ssa\.?\b" + _END), "dottoressa"),
    (_c(r"\bprof\.?\b" + _END), "professore"),
    (_c(r"\bprof\.?\s*ssa\.?\b" + _END), "professoressa"),
    (_c(r"\bing\.?\b" + _END), "ingegnere"),
    (_c(r"\barch\.?\b" + _END), "architetto"),
    (_c(r"\bavv\.?\b" + _END), "avvocato"),
    (_c(r"\bgeom\.?\b" + _END), "geometra"),
    (_c(r"\brag\.?\b" + _END), "ragioniere"),
    (_c(r"\bcomm\.?\b" + _END), "commercialista"),
    (_c(r"\bdott\.?\s*comm\.?\b" + _END), "dottore commercialista"),
    # --- Generic shorthand / Latin-ish
    (_c(r"\becc\.?\b" + _END), "eccetera"),
    (_c(r"\betc\.?\b" + _END), "eccetera"),
    (_c(r"\bes\.?\b" + _END), "per esempio"),
    (_c(r"\bp\.?\s*es\.?\b" + _END), "per esempio"),
    (_c(r"\bad\.?\s*es\.?\b" + _END), "ad esempio"),
    (_c(r"\bcioe\.?\b" + _END), "cioè"),
    (_c(r"\boss\.?\b" + _END), "ossia"),
    (_c(r"\bcfr\.?\b" + _END), "confronta"),
    (_c(r"\bvd\.?\b" + _END), "vedi"),
    (_c(r"\bvv\.?\b" + _END), "vedi"),
    # --- Document structure / references (useful for academic text)
    (_c(r"\bfig\.?\b" + _END), "figura"),
    (_c(r"\btab\.?\b" + _END), "tabella"),
    (_c(r"\beq\.?\b" + _END), "equazione"),
    (_c(r"\bsez\.?\b" + _END), "sezione"),
    (_c(r"\bsec\.?\b" + _END), "sezione"),
    (_c(r"\bcap\.?\b" + _END), "capitolo"),
    (_c(r"\bpar\.?\b" + _END), "paragrafo"),
    (_c(r"\bapp\.?\b" + _END), "appendice"),
    (_c(r"\bpagg\.?\b" + _END), "pagine"),
    (_c(r"\bpag\.?\b" + _END), "pagina"),
    # --- Identifiers / contact / numbering
    (_c(r"\btel\.?\b" + _END), "telefono"),
    (_c(r"\bcell\.?\b" + _END), "cellulare"),
    (_c(r"\bint\.?\b" + _END), "interno"),
    (_c(r"\bn\.?\b" + _END), "numero"),
    (_c(r"\bnn\.?\b" + _END), "numeri"),
    (_c(r"\bnr\.?\b" + _END), "numero"),
    (_c(r"\bcod\.?\s*fisc\.?\b" + _END), "codice fiscale"),
    (_c(r"\bc\.?\s*f\.?\b" + _END), "codice fiscale"),
    (_c(r"\bp\.?\s*iva\.?\b" + _END), "partita iva"),
    (_c(r"\bpiva\b" + _END), "partita iva"),
    # --- Dates / formal ref (common in letters)
    (_c(r"\bu\.?\s*s\.?\b" + _END), "ultimo scorso"),
    (_c(r"\bp\.?\s*v\.?\b" + _END), "prossimo venturo"),
    # --- Company legal forms (common in names)
    (_c(r"\bs\.?\s*p\.?\s*a\.?\b" + _END), "società per azioni"),
    (_c(r"\bs\.?\s*r\.?\s*l\.?\b" + _END), "società a responsabilità limitata"),
    (_c(r"\bs\.?\s*n\.?\s*c\.?\b" + _END), "società in nome collettivo"),
    (_c(r"\bs\.?\s*a\.?\s*s\.?\b" + _END), "società in accomandita semplice"),
    (_c(r"\bcoop\.?\b" + _END), "cooperativa"),
    (_c(r"\bonlus\b" + _END), "organizzazione non lucrativa di utilità sociale"),
    # --- Months (common in dates)
    (_c(r"\bgen\.?\b" + _END), "gennaio"),
    (_c(r"\bfeb\.?\b" + _END), "febbraio"),
    (_c(r"\bmar\.?\b" + _END), "marzo"),
    (_c(r"\bapr\.?\b" + _END), "aprile"),
    (_c(r"\bmag\.?\b" + _END), "maggio"),
    (_c(r"\bgiu\.?\b" + _END), "giugno"),
    (_c(r"\blug\.?\b" + _END), "luglio"),
    (_c(r"\bago\.?\b" + _END), "agosto"),
    (_c(r"\bset\.?\b" + _END), "settembre"),
    (_c(r"\bott\.?\b" + _END), "ottobre"),
    (_c(r"\bnov\.?\b" + _END), "novembre"),
    (_c(r"\bdic\.?\b" + _END), "dicembre"),
    # --- Web/contact tokens (optional but often helpful in prompts)
    (_c(r"\be-?mail\b" + _END), "email"),
    (_c(r"\bwww\.?\b" + _END), "doppia vu doppia vu doppia vu"),
]

# More aggressive abbreviations
abbreviations_it_aggressive = [
    # Address components
    (_c(r"\bv\.?\s*le\.?\b" + _END), "viale"),  # v.le
    (_c(r"\bp\.?\s*za\.?\b" + _END), "piazza"),  # p.za
    (_c(r"\bp\.?\s*zza\.?\b" + _END), "piazza"),  # p.zza
    (_c(r"\bp\.?\s*le\.?\b" + _END), "piazzale"),  # p.le
    (_c(r"\bc\.?\s*so\.?\b" + _END), "corso"),  # c.so
    # Saints: expand "S." to "san" only when followed by a capitalized token
    # (helps avoid matching "s." in arbitrary contexts).
    (_c(r"\bs\.(?=\s+[A-ZÀ-ÖØ-Ý])"), "san"),
    (_c(r"\bs\.\s*maria\b"), "santa maria"),
]

abbreviations_it_all = abbreviations_it + abbreviations_it_aggressive
