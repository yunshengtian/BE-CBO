from .gp import fit_gp_classifier
from .de import fit_de_classifier

fit_classifier = {
    'gp': fit_gp_classifier,
    'de': fit_de_classifier,
}
