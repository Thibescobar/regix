# Clôture de l'audit

Cette matrice relie chacun des 134 constats de `AUDIT.md` à sa résolution dans la
version 0.2.0. Elle décrit l'état de l'archive de livraison, pas celui de la branche
historique mentionnée dans `HANDOFF.md`.

Statuts : **C** = correction vérifiée localement ; **D** = décision de conception rendue
explicite et protégée ; **C/E** = correction locale présente, avec validation externe
encore obligatoire. Une exigence externe n'est jamais présentée comme acquise.

## Documentation et promesses

| ID | Statut | Résolution et preuve principale |
|---|---:|---|
| A-01 | C | `regix presets NAME` restitue le YAML source avec ses commentaires ; contrat dans `test_documentation.py`. |
| A-02 | C | Tableau des presets corrigé et test mécanique des étapes/N4. |
| A-03 | C | Entrées elastix persistées, paramètres et transformées renommés sans ambiguïté ; tests de rejeu/existence. |
| A-04 | C | Masque corporel natif calculé une fois, seuil explicite et morphologie physique ; tests pipeline. |
| A-05 | C | Description de `features.enabled=auto` alignée sur la résolution réelle et contrôlée par les tests documentaires. |
| A-06 | C | Tous les champs de profil alimentent contexte, ROI, gates ou manifeste ; test réflexif de consommation. |
| A-07 | C | Badge et ventilation portés à 212, comparés à `pytest --collect-only`. |
| A-08 | C | Docstring DICOM ramené aux comportements implémentés : regroupement UID, choix explicite, diagnostics géométriques. |
| A-09 | C | Empreintes d'entrée réellement calculées et sérialisées sans chemin source ; contrats de manifeste. |
| A-10 | C | Exports paresseux du package registration ; test d'import sans chargement des bibliothèques lourdes. |
| A-11 | C | CI vérifie réellement le code retour et l'identité du moteur rapporté par `doctor`. |
| A-12 | C | Rapport d'environnement inclut Regix, ITK, elastix, SimpleITK, pydicom, NumPy et composants de rendu. |
| A-13 | C | Options `segment --backend`, `--task`, `--fast/--no-fast` et `--timeout` ajoutées ; aide CLI testée. |
| A-14 | C | Chargeur unifié pour elastix `.txt`, ITK `.tfm`, `.itk.txt` et `.h5` ; matrice de tests par format. |
| A-15 | C/E | Normalisation déplacée dans le consommateur anatomix et documentée ; exécution des variantes/poids reste à valider sur le site GPU. |
| A-16 | C/E | Contrat golden/déterminisme et protocole de capture ajoutés ; reproductibilité reste surveillée par build elastix, pas garantie universellement. |
| A-17 | C | README distingue trois exports linéaires systématiques du DICOM REG conditionnel et des sorties déformables. |
| A-18 | C | Contrat de rééchantillonnage corrigé et vérifié en géométrie physique, y compris grilles obliques. |

## Comportements fonctionnels

| ID | Statut | Résolution et preuve principale |
|---|---:|---|
| B-01 | C | Résolution multi-label des alias pulmonaires, volumes sommés et cache stable ; régressions dédiées. |
| B-02 | C | Extraction du premier volume 3D complet d'un 4D et refus explicite du 2D. |
| B-03 | C | `regix apply` charge le `.itk.txt` produit par Regix ; erreur utilisateur stable pour format invalide. |
| B-04 | C | Garde de quantification branché sur le vrai contexte et testé à travers le pipeline. |
| B-05 | C | Porte métrique dégénérée tient compte du sens/type de métrique et ne rejette plus un MSE parfait. |
| B-06 | C | Tâche, mode rapide et délai TotalSegmentator configurables et transmis au sous-processus. |
| B-07 | C | Échec d'une initialisation explicitement demandée fatal ; seul multistart peut éliminer un candidat. |
| B-08 | C | MIND reste le repli CPU si anatomix est absent, même lorsque torch est installé. |
| B-09 | C | Overwrite nettoie uniquement l'inventaire Regix centralisé et conserve les fichiers utilisateur. |
| B-10 | C | Fusion profonde conserve les `None` explicites pour preset, Python et API. |
| B-11 | C | Conversion de transformée initiale respecte/vérifie la convention source au lieu de forcer `ComputeZYX`. |
| B-12 | C | Organe entièrement hors champ devient un échec de chevauchement, avec mesure et raison. |
| B-13 | C | Caractère déformable dérivé de la transformée exécutée, non d'une expression de configuration ambiguë. |
| B-14 | C | Instances d'un même UID fusionnées entre répertoires avant chargement. |
| B-15 | C | UID de série exposé en configuration, CLI et API, avec test de sélection explicite. |
| B-16 | C | Multistart utilise NCC en monomodal et NMI en multimodal. |

## Sécurité, confidentialité et DICOM

| ID | Statut | Résolution et preuve principale |
|---|---:|---|
| C-01 | C | HMAC-SHA256, clé aléatoire par processus sans secret, refus du vide et avertissement de faiblesse. |
| C-02 | C | Redaction/empreintes opaques dans manifeste, configuration et logs ; canari de chemin testé. |
| C-03 | C | Allowlist canonique, refus des symlinks hors racine, token optionnel, erreurs opaques et guide de déploiement. |
| C-04 | C | Noms de cas batch limités à un composant sûr ; traversées et séparateurs refusés. |
| C-05 | C | Racine UID validée/configurable, provenance DICOM renseignée et avertissement sur la racine de test. |
| C-06 | C/E | Toutes les coupes référencées, séquences/type 2 ajoutés et dates cohérentes ; test `dciodvfy` explicite mais skippé si outil absent. |
| C-07 | C | Dépendance resserrée à pydicom 3.x et code 2.x résiduel supprimé. |
| C-08 | C/E | Références privées/obsolètes supprimées et politique de SOP Class contrôlée ; validation des modalités/PACS reste externe. |

## Architecture

| ID | Statut | Résolution et preuve principale |
|---|---:|---|
| D-01 | C | `RunState`, étapes libres pour masques/contexte et entrée QC structurée extraient les dépendances critiques de l'orchestrateur. |
| D-02 | C | `Mask` immuable porte les rôles criterion/initialization/QC/body ; ROI reste un objet géométrique distinct. |
| D-03 | C | Fenêtres de profils devenues recommandations/entrées de consommateurs, jamais prétraitement implicite des intensités elastix. |
| D-04 | C | Génération du champ déplacée dans `preprocess.geometry`, indépendante du module GPU. |
| D-05 | C | `load_transform` centralise les formats et chaînes supportés. |
| D-06 | C | `layout.py` centralise noms, inventaire et nettoyage des artefacts. |
| D-07 | C | État de run local ; volumes/configuration d'appel non mutés ; réutilisation du pipeline testée. |
| D-08 | D | Constructeur HTML conservé sans dépendance de template, mais échappement, tri déterministe, formats compacts et tests hostiles rendent le choix explicite. |

## Simplicité et dette technique

| ID | Statut | Résolution et preuve principale |
|---|---:|---|
| E-01 | C | Comparaison de grille unique dans `preprocess.geometry`. |
| E-02 | C | Valeur de fond centralisée et réutilisée. |
| E-03 | C | Symboles morts supprimés ; vulture à 90 % bloque la CI avec allowlist explicite. |
| E-04 | C | Dépendances utilisées déclarées, extras séparés et déclarations inutiles retirées. |
| E-05 | C | Distances de surface calculées dans un passage partagé. |
| E-06 | C | Métriques utilisent la définition commune de grille identique. |
| E-07 | C | Indirection de lecture de paramètres supprimée/absorbée par le parseur public. |
| E-08 | C | Construction SRO morte remplacée par les vraies séquences DICOM. |
| E-09 | C | Rapport trie ses collections avant sérialisation ; sortie déterministe testée. |
| E-10 | C | Rapport d'environnement assigne chaque composant une seule fois. |
| E-11 | C | `keep_intermediate` supprime traces/logs de travail, sans toucher aux livrables de rejeu. |
| E-12 | C | Déclaration de templates inexistants retirée ; contenu de roue contrôlé. |

## Robustesse et gestion d'erreurs

| ID | Statut | Résolution et preuve principale |
|---|---:|---|
| F-01 | C | Toutes les formes invalides de `--set` deviennent des erreurs CLI courtes sans traceback. |
| F-02 | C | Options d'énumération typées/validées par Typer avant Pydantic. |
| F-03 | C | Copies d'exports secondaires protégées et consignées comme dégradation. |
| F-04 | C | Parseur elastix accepte espaces, guillemets, multilignes et doublons selon une règle testée ; fichier vide refusé. |
| F-05 | C | Sérialiseur cite valeurs non finies/ambiguës et protège les chaînes/underscores. |
| F-06 | C | Conversion ITK vérifiée sur centre et coins intérieurs réels ; zéro sonde évaluée est une erreur. |
| F-07 | C | Fichiers landmark `index` refusés avec instruction de conversion physique. |
| F-08 | C | Axes principaux dégénérés détectés avant construction de la transformée. |
| F-09 | C | Gates de ratio de déplacement physiologique, explicites ou dérivées du profil ; organe rigide protégé. |
| F-10 | C | Repli union de labels rendu visible et limité ; cible absente ne passe plus silencieusement. |
| F-11 | C | Stockage borné/paginé et pool fermé par lifespan FastAPI. |
| F-12 | C | Clé cache inclut volume complet, géométrie, backend/version/options ; cache corrompu rejeté. |
| F-13 | C | Import optionnel isolé de l'exécution ; `ImportError` interne n'est plus converti en dépendance absente. |
| F-14 | C | Bounding box accepte correctement un label map non binaire et ses messages décrivent le contrat réel. |
| F-15 | C | Morphologie par rayon physique autorise zéro voxel sur un axe épais au lieu de sur-dilater. |
| F-16 | C | Bornes Pydantic ajoutées aux espacements, tailles, slices, itérations, threads, rotations et délais. |
| F-17 | C | QC désactivé renvoie `NOT_EVALUATED`, jamais un faux avertissement. |

## Performance et mémoire

| ID | Statut | Résolution et preuve principale |
|---|---:|---|
| G-01 | C | QC d'intensité travaille sur vues/échantillons bornés et évite les copies float64 répétées. |
| G-02 | C | `Volume.describe()` utilise vues et quantiles bornés sans copie intégrale. |
| G-03 | C | Axes principaux vectorisés en coordonnées physiques. |
| G-04 | C | Filtres de distance de surface partagés par organe. |
| G-05 | C | MIND traite les décalages avec réutilisation/chunking et garde mémoire. |
| G-06 | C | Découverte DICOM parcourt une fois l'arbre, groupe ensuite par UID et borne les candidats. |
| G-07 | C | Figures WebP et sidecars optionnels réduisent le rapport ; contenu textuel reste autonome. |
| G-08 | C | Masque corporel natif mémorisé puis rééchantillonné ; statistiques bornées. |
| G-09 | C | Jacobien/déplacement linéaires analytiques ; champ dense float32 seulement si nécessaire. |

## Contrats et typage

| ID | Statut | Résolution et preuve principale |
|---|---:|---|
| H-01 | C | TRE reçoit un transformeur de points explicite et l'utilise selon son contrat. |
| H-02 | C | `AppliedTransform` expose des comportements cohérents, avec indisponibilité explicite plutôt que repli divergent. |
| H-03 | C | Valeurs `StageConfig.extra` préservent le type booléen et la sérialisation elastix attendue. |
| H-04 | C | `Volume.with_image` copie les métadonnées. |
| H-05 | C | `compose` renvoie toujours un transform indépendant, même avec un élément. |
| H-06 | C | `ParamContext` exige les champs de contexte ; constructeur depuis `RunState` testé. |
| H-07 | C | Indisponibilité métrique représentée par `None` et JSON strict refuse NaN/Inf. |
| H-08 | C | `TypedDict` pour rapports QC et validation des clés avant évaluation des gates. |
| H-09 | C | Mypy ajouté à l'extra dev et à la CI en mode progressif `continue-on-error`, conformément à la recommandation d'audit. |

## Couverture de tests

| ID | Statut | Résolution et preuve principale |
|---|---:|---|
| I-01 | C | Quinze tests API couvrent requêtes, sécurité, état et erreurs. |
| I-02 | C/E | Test SRO renommé/renforcé et validation externe conditionnelle ajoutée ; outil absent de l'environnement de livraison. |
| I-03 | C | Quantification traversée par un run pipeline complet. |
| I-04 | C | Fantôme anisotrope à coupes épaisses et direction oblique ajouté, erreur physique bornée. |
| I-05 | C | JSON strict, inventaire relatif et contrats exacts de fichiers testés. |
| I-06 | C | Couverture exécutée sur toute la suite, seuil CI 78 et badge à au plus un point. |
| I-07 | C | Batch couvre noms hostiles, collisions, échecs et résumé. |
| I-08 | C | Absence/import partiel des dépendances optionnelles testé sans chargement accidentel. |
| I-09 | C/E | Fixture zoo et replay stabilisés ; golden reste explicitement spécifique au build. |
| I-10 | C | Invariants de grille, composition, FOV, copie et points physiques ajoutés. |

## Dépendances, packaging et CI

| ID | Statut | Résolution et preuve principale |
|---|---:|---|
| J-01 | C | pydicom borné sur la série 3.x requise. |
| J-02 | C/E | CI installe les extras séparément et vérifie leur résolution ; GPU/poids externes ne sont pas exécutés. |
| J-03 | C | Variables d'environnement documentées, validées et rapportées sans secrets. |
| J-04 | C | Bornes hautes ajoutées et lock numérique Python 3.12 fourni avec job résolu/locké. |
| J-05 | C | Build sdist/wheel, `twine check`, installation hors source et présence des presets en CI. |
| J-06 | C | Quinze contrats documentaires comparent README, CLI, presets, versions, packaging et CI. |
| J-07 | C | Timeouts sur les jobs, format bloquant et artefact QC exigé. |
| J-08 | C | `--version`, sous-commande version et sorties JSON/machine pour commandes concernées. |

## Documentation, ergonomie et maintenance

| ID | Statut | Résolution et preuve principale |
|---|---:|---|
| K-01 | C | `CONTRIBUTING`, `CHANGELOG`, architecture, opérations, clôture et vérification ajoutés. |
| K-02 | C | README muni d'une navigation par public vers les guides spécialisés. |
| K-03 | C | YAML arbitraire imprimé sans interprétation Rich. |
| K-04 | C | `doctor --json` cohérent, imports optionnels bornés et impact de chaque capacité affiché. |
| K-05 | C | Version elastix obtenue de la distribution réelle, distincte de la version ITK. |
| K-06 | C | `load_preset` distingue nom inconnu, chemin absent et YAML invalide. |
| K-07 | C | Construction de configuration copie les dictionnaires entrants. |
| K-08 | C | Message de valeurs non finies décrit le repli réellement appliqué. |
| K-09 | C | Avertissement FOV centralisé pour éviter les doublons. |
| K-10 | C | Recouvrement calculé sur parallélépipèdes orientés en espace physique. |

## Évolution et risques assumés

| ID | Statut | Résolution et preuve principale |
|---|---:|---|
| L-01 | C | Version portée à 0.2.0 et source unique dans `regix.__version__`/métadonnées dynamiques. |
| L-02 | C | Suffixes `.itk.txt`, `.elastix.txt` et `.parameters.txt` non ambigus, chargeur associé. |
| L-03 | C | Sorties e2e exclues de l'archive/paquet et règle PHI documentée. |
| L-04 | C | Replis explicites consignés ; demandes explicites, schémas inconnus et erreurs de backend ne sont plus avalés. |
| L-05 | D | `regix qc OUT_DIR` réévalue la politique depuis métriques archivées sans réintroduire les chemins PHI ; recalcul de métriques exige un nouveau run explicite. |
| L-06 | D | Backend unique conservé comme choix produit ; options TotalSegmentator encapsulées/validées et extension future documentée. |
| L-07 | C/E | Déterminisme testé et dépendances bornées ; propriété toujours qualifiée par plateforme/build. |
| L-08 | C | `keep_intermediate` et `write_elastix_inputs` séparés ; livrables transform/paramètres jamais supprimés par le nettoyage de traces. |
| L-09 | C/E | Copies réduites, MIND borné, analyse linéaire analytique, mémoire disponible dans doctor et guide de capacité ; pas de moteur out-of-core. |

## Limite de cette clôture

La matrice atteste la présence des corrections et contrats dans le dépôt. Les preuves
exécutées sont consignées dans `VERIFICATION.md`. Les points marqués **C/E** exigent encore
les outils ou environnements indiqués : ils ne bloquent pas la cohérence de l'archive,
mais bloquent toute affirmation de conformité DICOM, de performance GPU, de capacité
clinique ou de validation médicale.
