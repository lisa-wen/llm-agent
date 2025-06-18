from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Liste der korrekten seltenen Krankheiten
rare_diseases = [
    "Hutchinson-Gilford Progeria Syndrome",
    "Fibrodysplasia Ossificans Progressiva",
    "Stiff Person Syndrome",
    "Erdheim-Chester Disease",
    "Paraneoplastic Pemphigus",
    "Hallermann-Streiff Syndrome",
    "Alkaptonuria",
    "Norrie Disease",
    "Hyper IgM Syndrome",
    "Takayasu Arteritis"
]

# Liste der absichtlich falsch geschriebenen Krankheiten
misspelled_diseases = [
    "Hutchison-Guilford Progeria",
    "Fibrodisplasia Osificans Progressiva",
    "Stif Person Sindrome",
    "Erdman-Chester Diesease",
    "Paraneoplastik Pemfigus",
    "Halerman-Strief Syndrome",
    "Alcaptunurea",
    "Norie Desease",
    "Hyper IGM Syndrom",
    "Takayasue Artheritis"
]

# SentenceTransformer-Modell laden
model = SentenceTransformer("SalmanFaroz/DisEmbed-v1")

# Einbettungen (Embeddings) für die korrekten Krankheiten erzeugen
correct_embeddings = model.encode(rare_diseases)

# Einbettungen für die falsch geschriebenen Krankheiten erzeugen
misspelled_embeddings = model.encode(misspelled_diseases)

# Kosinus-Ähnlichkeit zwischen den falsch geschriebenen und den korrekten Krankheiten berechnen
similarity_matrix = cosine_similarity(misspelled_embeddings, correct_embeddings)

# Für jede falsche Schreibweise die ähnlichste korrekte Krankheit finden
for i, misspelled in enumerate(misspelled_diseases):
    most_similar_index = np.argmax(similarity_matrix[i])  # Index der höchsten Ähnlichkeit
    most_similar_disease = rare_diseases[most_similar_index]  # Entsprechender Krankheitsname
    score = similarity_matrix[i][most_similar_index]  # Ähnlichkeitswert
    print(f"Falsch geschrieben: {misspelled}\n Original: {most_similar_disease} (Ähnlichkeit: {score:.4f})\n")
