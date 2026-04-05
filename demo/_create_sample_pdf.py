"""
demo/_create_sample_pdf.py
Helper: generate a realistic sample student exam PDF for the demo.
Called automatically by run_demo.py if the PDF doesn't exist yet.
"""

from __future__ import annotations

from pathlib import Path

SAMPLE_CONTENT = """\
NOM : Ahmed Benali
PRÉNOM : Ahmed  
ÉTUDIANT ID : ahmed_benali_01
MATIÈRE : Programmation Python — Algorithmique
DATE : 05/04/2026
NOTE MAXIMALE : 20 points

══════════════════════════════════════════════════════════════
EXERCICE 1 — Fonctions récursives (5 points)
══════════════════════════════════════════════════════════════

Question 1.1 : Écrire une fonction récursive qui calcule le factoriel de n.

Réponse de l'étudiant :

def factoriel(n):
    # Cas de base
    if n = 0:
        return 1
    return n * factoriel(n-1)

# Test
print(factoriel(5))

Remarque : j'ai essayé avec n=5, le résultat devrait être 120.

──────────────────────────────────────────────────────────────

Question 1.2 : Calculer la somme des éléments d'une liste récursivement.

Réponse de l'étudiant :

def somme_liste(lst):
    if len(lst) == 0:
        return 0
    return lst[0] + somme_liste(lst[1:])

# Test avec [1, 2, 3, 4, 5]
print(somme_liste([1, 2, 3, 4, 5]))  # Attendu : 15

══════════════════════════════════════════════════════════════
EXERCICE 2 — Structures de données (7 points)
══════════════════════════════════════════════════════════════

Question 2.1 : Différence entre liste et tuple en Python.

Réponse de l'étudiant :
Une liste peut être modifiée (mutable) mais un tuple ne peut pas être modifié
(immutable). Les listes utilisent [] et les tuples utilisent ().
Exemple : ma_liste = [1, 2, 3] et mon_tuple = (1, 2, 3).

──────────────────────────────────────────────────────────────

Question 2.2 : Implémenter une pile (stack) avec une liste Python.

Réponse de l'étudiant :

class Pile:
    def __init__(self):
        self.elements = []
    
    def empiler(self, valeur):
        self.elements.append(valeur)
    
    def depiler(self):
        if self.est_vide():
            return None
        return self.elements.pop()
    
    def est_vide(self):
        return len(self.elements) == 0
    
    def sommet(self):
        return self.elements[-1]

# Test
p = Pile()
p.empiler(10)
p.empiler(20)
p.empiler(30)
print(p.depiler())  # 30

══════════════════════════════════════════════════════════════
EXERCICE 3 — Complexité algorithmique (4 points)
══════════════════════════════════════════════════════════════

Question 3.1 : Quelle est la complexité de la recherche binaire ?

Réponse de l'étudiant :
La recherche binaire est O(log n) car on divise la liste en deux à chaque étape.
C'est plus rapide que la recherche linéaire qui est O(n).

──────────────────────────────────────────────────────────────

Question 3.2 : Implémenter la recherche binaire.

Réponse de l'étudiant :

def recherche_binaire(liste, cible):
    gauche = 0
    droite = len(liste)  # bug : devrait être len(liste) - 1
    
    while gauche <= droite:
        milieu = (gauche + droite) // 2
        if liste[milieu] == cible:
            return milieu
        elif liste[milieu] < cible:
            gauche = milieu + 1
        else:
            droite = milieu - 1
    return -1

══════════════════════════════════════════════════════════════
EXERCICE 4 — Programmation orientée objet (4 points)
══════════════════════════════════════════════════════════════

Question 4.1 : Définir les concepts d'héritage et polymorphisme.

Réponse de l'étudiant :
L'héritage permet à une classe fille d'hériter des attributs et méthodes
d'une classe mère. Le polymorphisme permet à plusieurs classes d'avoir
des méthodes avec le même nom mais des comportements différents.
Je ne suis pas sûr de bien comprendre le polymorphisme avec les interfaces.

──────────────────────────────────────────────────────────────

FIN DE COPIE — Ahmed Benali
"""


def create_sample_pdf(output_path: str | Path) -> Path:
    """
    Write *SAMPLE_CONTENT* into a real PDF at *output_path* using PyMuPDF.

    Returns the Path of the created file.
    """
    import fitz  # PyMuPDF

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = fitz.open()
    lines = SAMPLE_CONTENT.split("\n")

    # Layout constants
    MARGIN_X = 50
    MARGIN_Y = 60
    LINE_HEIGHT = 14
    PAGE_HEIGHT = 842   # A4
    PAGE_WIDTH = 595    # A4
    LINES_PER_PAGE = (PAGE_HEIGHT - 2 * MARGIN_Y) // LINE_HEIGHT

    page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    y = MARGIN_Y
    line_count = 0

    for line in lines:
        if line_count >= LINES_PER_PAGE:
            page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
            y = MARGIN_Y
            line_count = 0

        font_size = 10
        bold = False

        if line.startswith("NOM") or line.startswith("PRÉNOM") or line.startswith("MATIÈRE"):
            font_size = 10
            bold = True
        elif line.startswith("══"):
            font_size = 9
        elif line.startswith("EXERCICE"):
            font_size = 11
            bold = True
        elif line.startswith("Question"):
            font_size = 10
            bold = True

        font = "helv" if not bold else "helvB"
        page.insert_text(
            (MARGIN_X, y),
            line if line else " ",
            fontname=font,
            fontsize=font_size,
            color=(0, 0, 0),
        )
        y += LINE_HEIGHT
        line_count += 1

    doc.save(str(output_path))
    doc.close()
    return output_path
