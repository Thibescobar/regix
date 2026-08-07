# Audit exhaustif — Regix

> Audit réalisé à froid, sans hypothèse de bonne foi sur le code, sur le dépôt à l'état
> du commit `15bb249` (branche `main`, arbre de travail propre).
>
> **Aucun fichier du projet n'a été modifié** (hors la création de ce document).
>
> Chaque constat est étiqueté :
> - **[CONFIRMÉ]** — reproduit par exécution réelle pendant l'audit (le script /
>   la commande est indiqué) ;
> - **[LECTURE]** — établi par lecture du code, logique tracée de bout en bout, mais
>   non exécuté ;
> - **[À VÉRIFIER]** — risque plausible identifié, non reproduit, nécessitant un
>   environnement ou des données absents ici.

---

## 0. Périmètre, méthode et environnement d'audit

### 0.1 Ce qui a été lu

L'intégralité des 40 fichiers versionnés (`git ls-files`), soit :

| Zone | Fichiers | Volume |
|---|---|---|
| Package `regix/` | 28 modules Python | ~296 Ko |
| Presets | 8 YAML | ~12 Ko |
| Tests | 6 fichiers | ~130 Ko |
| Doc / build / CI | `README.md`, `pyproject.toml`, `.gitignore`, `.github/workflows/ci.yml` | ~39 Ko |
| Artefacts doc | `docs/images/*` (3 fichiers), `LICENSE` | — |

Le répertoire `e2e_out/` présent dans l'arbre de travail (non versionné, correctement
ignoré) a été exploité comme **preuve d'exécution réelle** : `run_manifest.json`,
`report.html`, `config_effective.yaml` et les fichiers elastix ont servi à confronter
le comportement observé aux affirmations du code et du README.

### 0.2 Environnement d'exécution utilisé pour les vérifications

```
Python 3.13 (anaconda3)   SimpleITK 2.5.6   numpy 2.3.5
pydantic 2.12.4           typer 0.20.0      pydicom 3.0.2
itk / elastix 5.4.6 (via itk-elastix, présent et fonctionnel)
torch / monai / anatomix / TotalSegmentator : ABSENTS
```

### 0.3 Décompte de tests réel

```
pytest --collect-only -q
  tests/test_cli.py                      16
  tests/test_dicom_io.py                  7
  tests/test_pipeline.py                 13
  tests/test_registration_internals.py   19
  tests/test_units.py                    67
  ------------------------------------------
  TOTAL                                 122
```

---

## 1. Liste exhaustive des problèmes

Les identifiants sont stables et réutilisés dans le plan de correction (§2), la
checklist (§3) et le top 10 (§6).

---

### A. Cohérence entre ce que le projet AFFIRME et ce que le code FAIT

C'est la catégorie la plus fournie, et c'est délibéré : le README de ce projet est
inhabituellement affirmatif (« établi par mesure », « vérifié par test », « mathématiquement
lossless »). Chaque affirmation a donc été traitée comme une spécification et confrontée
au code. Les affirmations exactes sont listées en §5 (points positifs) ; ci-dessous, celles
qui ne tiennent pas.

---

#### A-01 — `regix presets NAME` ne conserve **aucun** commentaire, contrairement au README

- **Emplacement** : `README.md:189` ; `regix/config.py:475-478` (`RegistrationConfig.to_yaml`) ;
  `regix/cli.py:172-173`.
- **Gravité** : **Important**
- **Affirmation** : README, §Presets — « `regix presets NAME` prints the full YAML,
  **comments included**. »
- **Ce que fait le code** : `to_yaml()` fait `yaml.safe_dump(self.model_dump(mode="json"))`.
  Il sérialise l'**objet pydantic**, pas le fichier source. Les commentaires YAML sont
  perdus par construction.
- **[CONFIRMÉ]**
  ```
  '#' dans la sortie de `regix presets ct_mr_abdomen` : 0
  '#' dans regix/presets/ct_mr_abdomen.yaml           : 13
  ```
- **Pourquoi c'est un problème** : les commentaires des presets ne sont pas décoratifs —
  ils portent la **justification clinique** (« le CBCT est rigide parce que la question est
  "de combien bouger la table" », « N4 off : les descripteurs sont déjà invariants au
  contraste »). Le README désigne explicitement `regix presets NAME` comme le moyen de les
  lire. Un utilisateur qui suit le README ne verra jamais ces justifications.
- **Conséquences** : perte du seul canal documenté de transmission du raisonnement
  clinique encodé dans les presets ; l'utilisateur croit avoir tout vu.
- **Correction recommandée** : deux options, la première est la bonne.
  1. Ajouter à `cli.presets` une lecture brute du fichier source quand le preset est
     fourni (bundled ou chemin) : `console.print(Syntax(path.read_text(), "yaml"))`, et
     réserver `to_yaml()` (config résolue, sans commentaires) à un drapeau `--resolved`.
     Attention : le rendu `rich` doit passer par `Syntax` ou `console.print(..., markup=False)`,
     sinon les crochets du YAML sont interprétés comme du balisage rich (voir K-03).
  2. À défaut, corriger le README.
- **Tests nécessaires** :
  - `test_presets_detail_shows_the_source_comments` : `runner.invoke(app, ["presets", "ct_mr_abdomen"])`
    → `"N4 off by default" in result.stdout`.
  - Test paramétré sur les 8 presets : la sortie contient au moins un `#`.

---

#### A-02 — Le README annonce « N4 on the MR » pour `ct_mr_abdomen`, le preset désactive explicitement N4

- **Emplacement** : `README.md:181` (tableau des presets) contre
  `regix/presets/ct_mr_abdomen.yaml:26-31`.
- **Gravité** : **Important**
- **Affirmation** : « `ct_mr_abdomen` | MR to CT | rigid + affine + B-spline 20 mm |
  Features, multi-start, **N4 on the MR** ».
- **Ce que fait le code** : le preset écrit `n4_bias_correction: false` avec un commentaire
  de 4 lignes expliquant *pourquoi* : les descripteurs anatomix/MIND sont invariants au
  contraste, donc la correction n'apporte presque rien.
- **[CONFIRMÉ]** : `tests/test_units.py::test_no_bundled_preset_enables_n4_by_default`
  existe et *verrouille* le comportement inverse de ce qu'annonce le README. Le projet
  teste donc activement la contradiction.
- **Statut de la justification** : le commentaire du preset est **une bonne justification**
  (elle est technique, spécifique et vérifiable). Le problème n'est pas le choix, c'est
  que le README n'a pas suivi.
- **Conséquences** : un utilisateur qui choisit ce preset *pour* la correction de biais
  ne l'obtient pas ; il attribuera un éventuel échec à autre chose.
- **Correction recommandée** : remplacer la cellule par « Features, multi-start, N4 disponible
  (désactivé par défaut) » et renvoyer au commentaire du preset.
- **Tests nécessaires** : test de documentation — parser le tableau du README et vérifier
  que chaque affirmation « N4 » correspond à `n4_bias_correction: true` dans le preset cité.
  (Cf. J-06 pour la généralisation.)

---

#### A-03 — Les fichiers de paramètres elastix ne sont **pas** rejouables « as-is »

- **Emplacement** : `README.md:299` et `README.md:539-540` (pied du rapport HTML) ;
  `regix/registration/params.py:494-497` (l'en-tête écrit dans chaque fichier) ;
  `regix/pipeline.py` (aucune écriture des images de travail).
- **Gravité** : **Important**
- **Affirmations** :
  - README : « the complete chain of elastix parameter files, **replayable as-is with the
    elastix binary** » ;
  - `params.write_parameter_file` écrit littéralement dans chaque fichier :
    `// Replay with: elastix -f fixed.nii.gz -m moving.nii.gz -out . -p parameters.txt` ;
  - pied du `report.html` : « The parameter files and run manifest shipped alongside this
    report allow the computation to be **replayed exactly**. »
- **Ce que fait le code** : les images réellement remises à elastix sont
  `fixed_work` / `moving_work` — réorientées LPS, éventuellement clippées, rééchantillonnées
  à `working_spacing_mm`, éventuellement recadrées sur la ROI. **Elles ne sont jamais
  écrites sur disque.** La seule image de sortie est `moving_registered.nii.gz`, qui est le
  *résultat* sur la grille fixe d'origine, pas une entrée.
- **[CONFIRMÉ]**
  ```
  pipeline écrit fixed_work/moving_work sur disque : False
  contenu de e2e_out/ : aucun fichier fixed*.nii.gz / moving_work*.nii.gz
  ```
- **Aggravation sur le chemin features** : quand les canaux anatomix/MIND sont actifs,
  elastix reçoit **N images fixes et N images mobiles** (`-f0 -f1 … -m0 -m1 …`). La ligne
  `elastix -f fixed.nii.gz -m moving.nii.gz` est alors non seulement inexploitable, elle
  est *structurellement fausse*. Les canaux ne sont écrits que si
  `output.write_features: true`, dont le défaut est `false`.
- **Conséquences** : la promesse de reproductibilité — argument central du projet, repris
  jusque dans le pied du rapport clinique — n'est pas tenue. Six mois plus tard, un
  investigateur a les paramètres mais pas les entrées.
- **Correction recommandée** :
  1. Écrire `elastix/inputs/fixed_work.nii.gz` et `moving_work.nii.gz` (compressés) sous un
     drapeau `output.write_elastix_inputs`, activé par défaut si
     `runtime.keep_intermediate` (cf. E-11 qui rend ce drapeau fonctionnel) ;
  2. générer l'en-tête de rejeu **à partir des chemins réels** et du nombre de canaux :
     `elastix -f0 … -f3 … -m0 … -t0 ../initial_transform.txt -out . -p parameters.txt` ;
  3. si (1) est jugé trop coûteux en disque, retirer les trois affirmations.
- **Tests nécessaires** :
  - `test_elastix_inputs_are_written_and_the_replay_line_matches` : après un run,
    tous les fichiers cités dans la ligne `// Replay with:` de chaque `parameters.txt`
    existent, relativement au répertoire du stage.
  - Variante multi-canaux (chemin MIND, sans GPU) : la ligne cite `-f0..-fN`.

---

#### A-04 — Le README affirme que les deux masques corporels utilisent « le même seuil −300 HU » : faux pour 5 presets sur 8

- **Emplacement** : `README.md:507-512` (§Limitations) ;
  `regix/preprocess/geometry.py:163-175` (`body_mask`) ;
  `regix/pipeline.py:240-245` (masque QC) contre `regix/pipeline.py:290-291` (masque de travail).
- **Gravité** : **Important**
- **Affirmation** : « Both passes now take the same −300 HU threshold, so the cause is
  resolution-dependent morphology (the closing radius in voxels, and which component
  survives `keep_largest`), **not the intensity scale**. »
- **Ce que fait le code** : `body_mask` choisit sa branche dynamiquement :
  ```python
  mask = (sitk.BinaryThreshold(img, -300.0, 4000.0, 1, 0)
          if f.GetMinimum() < -200
          else sitk.OtsuThreshold(img, 0, 1, 128))
  ```
  Le masque QC (`pipeline.py:240`) est calculé sur le volume **original** → minimum ≈ −1024 →
  branche seuil. Le masque de travail (`pipeline.py:290`) est calculé sur le volume
  **préprocessé**, donc *après* `sitk.Clamp` si le preset déclare une fenêtre HU. Or
  `ct_liver` = (−30, 180) et `ct_bone` = (−200, 1000) : après clamp, le minimum vaut
  −30 ou −200, la condition `< -200` est fausse, et la seconde passe bascule sur **Otsu**.
- **[CONFIRMÉ]** — fantôme synthétique, seuil natif vs après fenêtrage :
  ```
  natif                  -> branche « threshold -300 »
  après ct_liver (-30)   -> branche OTSU        <-- algorithme différent
  après ct_bone (-200)   -> branche OTSU        <-- algorithme différent
  après ct_lung (-1000)  -> branche « threshold -300 »
  ```
- **Presets concernés** : `ct_cbct_igrt` (fixed+moving `ct_bone`),
  `ct_ct_liver_followup` (fixed+moving `ct_liver`), `mr_ct_prostate` (fixed `ct_soft` = −160…240),
  `pet_ct_wholebody` (fixed `ct_soft`), `base` non concerné, `ct_ct_lung_4d` non concerné.
  **5 presets sur 8** déclarent au moins une fenêtre CT qui fait basculer la branche.
- **Pourquoi c'est un problème** : la limitation documentée est honnête sur *l'existence* de
  l'écart 26 365 mL / 19 114 mL, mais son **diagnostic est faux** pour la majorité des presets.
  Un diagnostic faux envoie la correction future (« untangled on a dedicated branch ») dans
  la mauvaise direction : on ira regarder le rayon de fermeture morphologique alors que le
  problème premier est un changement d'algorithme de seuillage.
- **Conséquences** : masque de critère et masque de QC potentiellement très différents sur
  un CT réel (Otsu sur des données bornées à [−30, 180] segmente « tissu plus dense que la
  moyenne », pas « le patient »). Le critère elastix est échantillonné dans un masque, la
  NCC/NMI de QC dans un autre : le chiffre du rapport ne décrit pas la région optimisée.
- **Correction recommandée** :
  1. rendre le seuil **explicite et paramétrable** : `body_mask(image, modality, hu_threshold=-300.0,
     assume_hu: bool | None = None)` ; ne basculer sur Otsu que si l'appelant déclare
     explicitement que l'échelle n'est plus HU ;
  2. calculer le masque corporel **une seule fois, sur le volume natif**, puis le
     rééchantillonner (plus proche voisin) vers la grille de travail — ce qui supprime en
     même temps la moitié de l'écart de volume documenté ;
  3. mettre à jour la limitation du README avec le diagnostic corrigé.
- **Tests nécessaires** :
  - `test_body_mask_uses_the_same_branch_before_and_after_windowing` : pour chaque fenêtre de
    `HU_WINDOWS`, `body_mask(clamp(img, w))` et `body_mask(img)` doivent produire un volume
    concordant à 5 % près sur un fantôme anthropomorphe.
  - `test_qc_mask_and_criterion_mask_are_the_same_object_resampled` (test d'intégration
    pipeline, tolérance 2 %).

---

#### A-05 — `FeatureConfig.enabled: "auto"` décrit deux conditions qui n'existent pas dans le code

- **Emplacement** : `regix/config.py:152-158` (description du champ) ;
  `regix/pipeline.py:570-581` (`_features_wanted`) ; `regix/cli.py` (options).
- **Gravité** : **Important**
- **Affirmation** (description du champ, donc visible dans `config_effective.yaml`,
  dans le manifeste et dans `regix presets`) :
  > « auto = enabled when the modalities differ, torch+anatomix are present, **and a GPU
  > (or `--allow-cpu-features`) is available**. »
- **Ce que fait le code** :
  ```python
  # _features_wanted, mode auto :
  return bool(multimodal or needs_features)
  ```
  Aucune vérification de la présence de torch/anatomix, **aucune** vérification de GPU.
- **[CONFIRMÉ]** : `rg -- "allow.cpu.features" regix/` → **0 occurrence**. L'option CLI
  `--allow-cpu-features` **n'existe pas**. Le champ correspondant est
  `features.allow_cpu`, atteignable uniquement par YAML ou `--set features.allow_cpu=true`.
- **Trois erreurs distinctes dans une seule description** :
  1. « torch+anatomix are present » — non vérifié ici (c'est `_extract_features` qui gère
     ensuite le repli, voir B-08) ;
  2. « a GPU … is available » — jamais vérifié ;
  3. « `--allow-cpu-features` » — drapeau inexistant.
- **Conséquences** : l'utilisateur ne peut pas prédire quand les features s'activent ;
  une commande copiée depuis la description échoue avec `No such option`.
- **Correction recommandée** : réécrire la description pour décrire le code
  (« auto = activé dès que les modalités diffèrent, ou qu'un stage demande une métrique
  features, ou que `deformable_engine=convexadam` ; le fournisseur effectif — anatomix ou
  MIND-SSC — est décidé au moment de l'extraction ») **et** ajouter l'option CLI
  `--allow-cpu-features` qui pose `features.allow_cpu=true`, puisqu'elle est référencée
  par ailleurs.
- **Tests nécessaires** :
  - `test_features_auto_activates_on_a_multimodal_pair_without_gpu` ;
  - `test_allow_cpu_features_flag_exists_and_sets_the_config` ;
  - un test de cohérence générique : toute option `--xxx` citée dans une description de
    champ ou un docstring doit exister dans l'app typer (cf. J-06).

---

#### A-06 — `organs/labels.py` : 3 des 5 attributs de profil annoncés ne sont jamais consommés

- **Emplacement** : `README.md:412-416` ; `regix/organs/labels.py:69-84` (`OrganProfile`) ;
  `regix/pipeline.py:625-649` (`_resolve_stages`, seul consommateur).
- **Gravité** : **Important**
- **Affirmation** : « `organs/labels.py` encodes, for each organ, its deformability, the
  relevant B-spline grid, **the HU window, the mask margin and the expected physiological
  amplitude** ».
- **[CONFIRMÉ]** par recherche exhaustive sur `regix/` :

  | Attribut du profil | Défini | Lu par `merged_profile` | **Consommé par le pipeline** |
  |---|---|---|---|
  | `deformable` | oui | oui | **oui** (`_resolve_stages`) |
  | `bspline_grid_mm` | oui | oui | **oui** (`_resolve_stages`, `cli --deformable`) |
  | `hu_window` | 25 organes | oui | **non — jamais lu** |
  | `mask_dilate_mm` | oui | oui | **non** — le pipeline passe `cfg.organs.mask_dilate_mm` |
  | `roi_margin_mm` | oui | oui | **non** — `pipeline.py:284` passe toujours `cfg.organs.roi_margin_mm`, ce qui rend mort le repli `margin_mm is None` de `plan_roi` (`roi.py:135`) |
  | `typical_motion_mm` | oui | oui | **non — jamais lu** |
  | `recommended_stage_types()` | méthode | — | **non — jamais appelée** |
  | `region`, `notes` | oui | oui | **non** (jamais affichés) |

- **Pourquoi c'est un problème** : c'est un argument de vente structurant (« registration
  organ-aware instead of applying one setting to the whole body »), et il n'est tenu qu'à
  40 %. Les ~30 valeurs `hu_window="ct_liver"` etc. donnent l'illusion d'un système de
  profils vivant ; c'est une table de données morte à 60 %.
- **Conséquences** : (a) documentation trompeuse ; (b) `typical_motion_mm` aurait fourni le
  garde-fou naturel qui manque au §gates (comparer `displacement_statistics.p95_mm` à
  l'amplitude physiologique attendue — voir F-09) ; (c) toute maintenance de ces valeurs est
  du travail sans effet.
- **Correction recommandée** : *soit* câbler, *soit* supprimer. Câbler est peu coûteux et
  a de la valeur :
  1. `combined_mask(..., dilate_mm=cfg.organs.mask_dilate_mm if explicitement défini
     else profile.mask_dilate_mm)` ;
  2. `plan_roi(..., margin_mm=None)` quand l'utilisateur n'a pas fixé `roi_margin_mm`,
     ce qui réanime le repli existant ;
  3. `typical_motion_mm` → nouvelle porte `qc.gates.max_displacement_ratio` (F-09) ;
  4. `hu_window` : soit l'utiliser comme défaut de `preprocess.<side>.window` quand un
     organe est ciblé sur un CT et qu'aucune fenêtre n'est déclarée, soit le supprimer —
     attention, l'appliquer entrerait en conflit avec l'invariant « intensités natives »
     (§A-04, D-03) : c'est probablement `hu_window` qu'il faut supprimer, et documenter
     pourquoi.
- **Tests nécessaires** :
  - `test_organ_profile_margin_is_used_when_the_user_did_not_set_one` ;
  - `test_every_organ_profile_field_is_read_somewhere` : test réflexif qui échoue si un
    attribut de `OrganProfile` n'apparaît nulle part hors `labels.py`.

---

#### A-07 — Le README se contredit sur le nombre de tests (122 vs 90) et donne 2 décomptes par fichier faux

- **Emplacement** : `README.md:8` (badge) contre `README.md:448-454` (§Testing).
- **Gravité** : **Mineur**
- **[CONFIRMÉ]** :

  | Source | Total | units | pipeline | cli | dicom | internals |
  |---|---|---|---|---|---|---|
  | Badge `README.md:8` | **122** | — | — | — | — | — |
  | §Testing `README.md:448-454` | **90** | 37 | 11 | 16 | 7 | 19 |
  | `pytest --collect-only` | **122** | **67** | **13** | 16 | 7 | 19 |

- **Conséquences** : mineur en soi, mais c'est un **indicateur** : la section Testing n'a pas
  été régénérée depuis longtemps alors que le badge l'a été. Elle affirme aussi
  « ~2 min » sans que rien ne le vérifie.
- **Correction recommandée** : générer la section depuis `pytest --collect-only -q` dans un
  petit script `tools/refresh_readme_counts.py`, et le faire vérifier par la CI
  (job `lint`, non bloquant au début).
- **Tests nécessaires** : `test_readme_test_counts_match_collection` — parse le tableau du
  README, compare à `pytest --collect-only -q --no-header` par fichier.

---

#### A-08 — Le docstring de `io/dicom.py` promet trois comportements absents

- **Emplacement** : `regix/io/dicom.py:1-12`.
- **Gravité** : **Important**
- **Affirmations du docstring de module** :
  1. « incomplete series / **duplicate InstanceNumbers** » ;
  2. « patient metadata **never copied verbatim** into the outputs » ;
  3. (implicite) « several series in the same directory … » — celle-ci **est** tenue.
- **Ce que fait le code** :
  1. **[LECTURE]** aucune fonction ne lit `InstanceNumber` (`rg -- "0020|0013"` → 0 occurrence)
     ni ne détecte une série incomplète. Le seul contrôle est
     `_check_slice_regularity` (écart-type des positions), qui détecte *l'irrégularité*
     d'espacement, pas la duplication ni l'incomplétude ;
  2. **[CONFIRMÉ]** `write_derived_dicom` (`io/writers.py:162`) part de
     `ds = template.copy()` — donc **tous** les tags du DICOM source, `PatientName`,
     `PatientID`, `PatientBirthDate`, `OtherPatientIDs`, tags privés constructeur inclus,
     sont recopiés tels quels. Recherche de déidentification dans la fonction : aucune.
     `write_spatial_registration_dicom` recopie explicitement 8 tags identifiants
     (`io/writers.py:269-280`).
- **Nuance importante** : le comportement (2) est **cliniquement correct** — une série
  dérivée doit porter l'identité patient pour être classée dans le PACS, et une SRO doit
  vivre dans l'étude du patient. Le défaut n'est pas le comportement, c'est **l'affirmation
  inverse** dans le docstring, qui pourrait conduire un intégrateur à croire que les
  sorties DICOM de Regix sont pseudonymisées et à les router hors du périmètre de soin.
- **Conséquences** : risque de fuite de données de santé par mauvaise interprétation
  documentaire. Un `dicom_registered/` ou un `spatial_registration.dcm` traité comme
  « anonyme » sur la foi du docstring est une violation.
- **Correction recommandée** :
  - remplacer par : « les sorties DICOM **conservent délibérément** l'identité patient de la
    série source (obligatoire pour le classement PACS). Les journaux, le manifeste et le
    rapport HTML, eux, sont pseudonymisés — voir `logging_utils.pseudonymize`. » ;
  - supprimer la mention « duplicate InstanceNumbers » ou implémenter la détection
    (facile : `len(set(InstanceNumber)) != n_files` → warning dans `_probe_series`) ;
  - documenter le même point dans le README §Privacy, qui souffre du symptôme inverse (C-02).
- **Tests nécessaires** :
  - `test_derived_series_keeps_the_patient_identity` (verrouille le comportement voulu) ;
  - `test_duplicate_instance_numbers_are_reported` (si implémenté).

---

#### A-09 — `logging_utils` annonce des « input hashes » dans le manifeste : `file_digest` n'est jamais appelée

- **Emplacement** : `regix/logging_utils.py:6-9` (docstring) et `:81-94` (`file_digest`) ;
  `regix/logging_utils.py:140-194` (`RunManifest`, aucun champ de hash).
- **Gravité** : **Mineur** (mais avec une conséquence de traçabilité réelle)
- **Affirmation** : « one JSON manifest per run: library versions, effective configuration,
  **input hashes**, duration of each step. That file is what you re-read six months later
  to know what actually ran. »
- **[CONFIRMÉ]** : `file_digest` — 1 seule occurrence dans tout le dépôt (sa définition).
  Le `run_manifest.json` de `e2e_out/` ne contient aucun champ de hachage ; `inputs`
  contient la sortie de `Volume.describe()` (géométrie, statistiques d'intensité, chemin).
- **Pourquoi c'est un problème** : le hash d'entrée est précisément ce qui permet, six mois
  plus tard, de savoir que le fichier relu est bien celui qui a servi. Sans lui, la
  traçabilité repose sur un chemin (qui bouge) et une taille de grille (non discriminante).
  La fonction est écrite, testée nulle part, et branchée nulle part : c'est une promesse
  à moitié construite.
- **Conséquences** : traçabilité plus faible qu'annoncée ; code mort qui suggère le contraire.
- **Correction recommandée** : appeler `file_digest` dans `_load` (avec
  `max_bytes=64<<20` pour les gros volumes, le préfixe `partial:` étant déjà prévu) et
  stocker dans `manifest.inputs[side]["sha256"]`. Coût : ~0,3 s pour 64 Mo.
  Pour une série DICOM, hacher la concaténation des SOPInstanceUID triés est plus utile et
  plus rapide que de lire 800 fichiers.
- **Tests nécessaires** :
  - `test_manifest_records_an_input_digest` ;
  - `test_digest_is_stable_across_runs_and_changes_with_the_data`.

---

#### A-10 — Le docstring de `registration/__init__.py` justifie une isolation que le package viole lui-même

- **Emplacement** : `regix/registration/__init__.py:5-7` ; `regix/registration/warp.py:21`.
- **Gravité** : **Cosmétique**
- **Affirmation** : « this package deliberately re-exports nothing, so that pulling in one
  module does not import the six others — **`convexadam` in particular reaches for torch**. »
- **Ce que fait le code** : (a) `convexadam` n'importe **pas** torch au niveau module —
  l'import est local à `adam_instance_optimization` (`convexadam.py:64-71`), ce qui est bien
  fait ; la justification donnée est donc factuellement inexacte ; (b) `warp.py:21` importe
  `convexadam` **au niveau module**, si bien que tout import de `regix.registration.warp`
  (donc tout import de `regix.pipeline`) charge `convexadam` de toute façon.
- **Conséquences** : aucune conséquence fonctionnelle (l'objectif « ne pas tirer torch » est
  atteint, par un autre mécanisme que celui décrit). C'est une justification qui ne tient pas,
  ce qui la rend trompeuse pour un mainteneur.
- **Correction recommandée** : reformuler (« torch n'est importé que dans le corps de
  `adam_instance_optimization` ; c'est ce qui garantit que `import regix.pipeline` ne le
  tire pas ») et déplacer `displacement_field_from_transform` — utilitaire purement
  géométrique — vers `preprocess/geometry.py` (voir D-04).
- **Tests nécessaires** : `test_importing_the_pipeline_does_not_import_torch` :
  `assert "torch" not in sys.modules` après `import regix.pipeline` dans un sous-processus.

---

#### A-11 — Le commentaire de CI « regix doctor exits non-zero when the engine is missing » est vrai mais l'étape ne teste pas ce qu'elle prétend

- **Emplacement** : `.github/workflows/ci.yml:62-65`.
- **Gravité** : **Mineur**
- **Affirmation** : « `regix doctor` exits non-zero when the registration engine is missing:
  this is **a real smoke test of the itk-elastix installation**, not decoration. »
- **Ce que fait le code** : `cli.doctor` appelle `engine_available()` →
  `require_itk()`, qui teste `import itk` et `hasattr(itk, "ElastixRegistrationMethod")`.
  C'est un test de **présence de la classe**, pas de l'installation elastix : il ne construit
  aucun filtre, ne charge aucune bibliothèque native elastix, et ne détecte donc pas une
  roue `itk-elastix` cassée (DLL manquante, ABI incompatible) — cas qui se manifesteront au
  premier `UpdateLargestPossibleRegion()`, en production.
- **Nuance** : le job `test` exécute ensuite `tests/test_pipeline.py`, qui *lui* fait de
  vraies registrations. Le smoke test réel existe donc, mais ce n'est pas `doctor`.
- **Correction recommandée** : dans `engine_available()`, instancier réellement
  `itk.ElastixRegistrationMethod[itk.Image[itk.F,3], itk.Image[itk.F,3]].New()` dans un
  `try` — c'est ce qui déclenche le chargement natif. Et corriger le libellé de version
  retourné (voir K-05).
- **Tests nécessaires** : `test_engine_available_instantiates_the_filter` (monkeypatch d'un
  `itk` factice sans la classe → `(False, hint)`).

---

#### A-12 — Le README annonce que le manifeste contient « les versions de toutes les bibliothèques qui influencent numériquement le résultat » — elastix, pydicom et matplotlib manquent

- **Emplacement** : `README.md:386-387` ; `regix/logging_utils.py:97-137`
  (`environment_report`).
- **Gravité** : **Mineur**
- **Ce que fait le code** : `environment_report` collecte `SimpleITK, itk, numpy, torch,
  monai, anatomix`, plus `itk.Version.GetITKVersion()`. **[CONFIRMÉ]** sur
  `e2e_out/run_manifest.json` : clés = `anatomix, cuda_available, cuda_device, itk, machine,
  monai, numpy, platform, python, regix, simpleitk, torch`.
- **Manquent** :
  - la version du paquet **`itk-elastix`** lui-même. `itk.Version.GetITKVersion()` renvoie
    `5.4.6` = la version **ITK**, pas celle de la liaison elastix. Or elastix est *le* moteur :
    c'est la version la plus déterminante pour le résultat, et la seule que le manifeste
    ne peut pas restituer ;
  - **pydicom** (influence les sorties DICOM, et sa version 2 vs 3 change le comportement —
    voir J-01) ;
  - **matplotlib** (figures du rapport) et **PyYAML** (parsing de config), plus mineurs ;
  - **TotalSegmentator**, alors que le docstring de `organs/segmenter.py:8-9` vante
    « a pip-pinnable version **that the run manifest can record** » — ce que le manifeste
    ne fait pas.
- **Conséquences** : impossible de reproduire un run à l'identique en repartant du
  manifeste, ce qui est pourtant sa raison d'être annoncée.
- **Correction recommandée** : remplacer la boucle par
  `importlib.metadata.version(dist)` sur une liste explicite
  `["itk-elastix", "SimpleITK", "numpy", "pydicom", "PyYAML", "matplotlib", "torch",
  "monai", "TotalSegmentator", "anatomix"]`, en gardant `itk.Version.GetITKVersion()`
  comme entrée séparée `itk_core`.
- **Tests nécessaires** :
  - `test_environment_report_records_the_elastix_binding_version` ;
  - `test_environment_report_never_raises_when_a_package_is_absent`.

---

#### A-13 — Le README documente `regix segment … --backend totalsegmentator` ; l'option n'existe pas

- **Emplacement** : `regix/cli.py:9` (docstring de module) ; `regix/cli.py:491-496`
  (signature réelle de `segment`).
- **Gravité** : **Mineur**
- **Ce que fait le code** : `segment` accepte `image`, `-o/--output`, `--organ`. Il n'y a
  **pas** de `--backend` : `TotalSegmentatorSegmenter` est instancié en dur (`cli.py:507`).
- **[CONFIRMÉ]** par lecture de la signature ; `--backend` n'apparaît que dans le docstring.
- **Conséquences** : commande copiée depuis la doc → `No such option: --backend`.
  Accessoirement, `regix segment` est le seul chemin où `external` n'a aucun sens
  (segmenter suppose de *produire* les masques), donc l'option n'aurait pas d'utilité :
  c'est la doc qu'il faut corriger, pas le code.
- **Correction recommandée** : retirer `--backend` de la ligne 9 du docstring.
  Ajouter en revanche `--device` et `--fast/--no-fast`, qui manquent réellement (voir B-06).
- **Tests nécessaires** : couvert par le test générique J-06.

---

#### A-14 — `InitConfig.transform_file` annonce accepter un `.txt` elastix ; `sitk.ReadTransform` ne sait pas les lire

- **Emplacement** : `regix/config.py:235` (description du champ) ;
  `regix/registration/initialize.py:287`.
- **Gravité** : **Important**
- **Affirmation** : `transform_file: Path | None = Field(..., description="file mode:
  **elastix .txt** or ITK .tfm.")`
- **Ce que fait le code** : `t = sitk.ReadTransform(str(config.transform_file))`.
  `sitk.ReadTransform` lit les formats ITK (`.tfm`, `.mat`, `.h5`, et les *Insight Transform
  File* `.txt`). Il **ne sait pas** lire un `TransformParameters.0.txt` elastix, dont la
  syntaxe est `(Key "value")`.
- **[CONFIRMÉ] indirectement** — même mécanisme que B-03, où la lecture croisée des deux
  formats `.txt` a été reproduite et provoque une `RuntimeError` du parseur.
- **Aggravation** : les deux formats portent l'extension `.txt`, et Regix écrit lui-même
  **les deux** dans le même run (`elastix/stageNN/TransformParameters.0.txt` d'un côté,
  `transform/stageNN_rigid.txt` de l'autre). La confusion est structurellement invitée.
- **Conséquences** : `init.mode=file` avec un fichier elastix échoue par une exception ITK
  brute, non interceptée par `build_candidates`… — en réalité **si**, elle l'est
  (`except Exception` ligne 291) : le candidat est simplement « écarté » avec un warning, et
  le run continue **avec une initialisation géométrique par défaut**. L'utilisateur a demandé
  une initialisation précise, il en reçoit une autre, et un simple warning le signale.
  C'est le pire des trois comportements possibles.
- **Correction recommandée** :
  1. corriger la description : « ITK transform file (`.tfm`, `.h5`, ou Insight `.txt`) » ;
  2. ajouter un chargeur unifié `transforms.load_any_transform(path)` qui renifle le
     contenu (`(Transform "` en tête → elastix ; sinon ITK) et route vers
     `parameter_map_to_transform` ou `sitk.ReadTransform` — Regix a déjà les deux briques.
     Ce chargeur sert aussi à `regix apply` (B-03) ;
  3. **ne pas** avaler l'échec d'un candidat explicitement demandé : quand
     `config.mode is not MULTISTART`, l'échec du candidat unique doit être fatal, pas
     silencieusement remplacé (voir F-02).
- **Tests nécessaires** :
  - `test_init_from_an_elastix_parameter_file` et `test_init_from_an_itk_tfm` ;
  - `test_a_failing_explicit_init_mode_is_fatal_not_silently_replaced`.

---

#### A-15 — `voxel_normalize` documenté comme « requis par les variantes anatomix-dev », appliqué en réalité à toutes

- **Emplacement** : `regix/features/reduce.py:24-25` ;
  `regix/features/anatomix.py:176` ; `regix/config.py:166`.
- **Gravité** : **Mineur** *(impact numérique **À VÉRIFIER**)*
- **Affirmation** : « Normalise each voxel across channels (**required by the anatomix-dev
  variants**). »
- **Ce que fait le code** : `AnatomixExtractor.extract` appelle inconditionnellement
  `voxel_normalize(features, self.config.voxel_normalize)`, dont le défaut est `"l2"`.
  La variante `anatomix` (celle par défaut, `config.py:159`) subit donc une normalisation
  L2 par voxel que le docstring présente comme spécifique aux variantes *dev*.
- **Pourquoi c'est un problème** : (a) contradiction doc/code ; (b) **[À VÉRIFIER]** si
  l'usage de référence d'anatomix n'applique pas cette normalisation à la variante de base,
  Regix modifie le descripteur avant de le confier à elastix — modification silencieuse, non
  mesurée, sur le chemin qui porte l'argument principal du projet (« CT vs MR devient
  monomodal »). Impossible de trancher ici : les poids anatomix ne sont pas installés.
- **Conséquences** : écart potentiel non quantifié avec la méthode publiée ; risque de
  reproduire un résultat différent de la référence.
- **Correction recommandée** : (1) corriger le docstring ; (2) documenter le choix à l'endroit
  concerné (`anatomix.py:176`) en indiquant s'il est délibéré et sur quelle base ;
  (3) idéalement, mesurer sur une paire de référence l'effet de `voxel_normalize="none"`
  contre `"l2"` sur la TRE, et consigner le chiffre.
- **Tests nécessaires** (marqués `@pytest.mark.gpu`) :
  - `test_voxel_normalize_default_matches_the_documented_scope` ;
  - test de non-régression numérique sur un fantôme, avec un seuil de TRE.

---

#### A-16 — `README:499` affirme « 0.000 mm point-wise over 400 points » sans aucun test ni script correspondant dans le dépôt

- **Emplacement** : `README.md:495-500` (§Limitations, déterminisme).
- **Gravité** : **Mineur**
- **Affirmation** : « (measured: **0.000 mm point-wise over 400 points**, and identical
  similarity metrics across separate processes) ».
- **[CONFIRMÉ]** : aucun test du dépôt ne fait tourner deux registrations dans deux processus
  et ne compare 400 points. `rg -- "400"` dans `tests/` → aucune occurrence pertinente.
- **Nuance à l'avantage du projet** : le paragraphe est par ailleurs **exemplairement
  honnête** — il précise que ce n'est pas une garantie Regix, que c'est une propriété du
  build elastix, et qu'il ne faut pas s'y fier sans revérifier. C'est le bon ton.
- **Pourquoi c'est quand même un problème** : un chiffre précis non reproductible est
  invérifiable par un tiers. La consigne d'audit est explicite : « toute affirmation non
  vérifiable est un problème à part entière ».
- **Correction recommandée** : ajouter `tests/test_determinism.py` marqué `@pytest.mark.slow`
  qui exécute deux fois la même configuration via `subprocess` et compare 400 points
  échantillonnés dans le FOV, avec un `pytest.skip` documenté si elastix est absent.
  Le test peut être hors CI ; il rend le chiffre reproductible.
- **Tests nécessaires** : le test lui-même est la correction.

---

#### A-17 — Le README annonce 4 formats de transform ; en pratique 3 seulement sont systématiques

- **Emplacement** : `README.md:39-41` et `README.md:297-302` ;
  `regix/pipeline.py:955-971`.
- **Gravité** : **Mineur**
- **Affirmation** : « the transform in **four formats** (including a DICOM Spatial
  Registration Object a planning system can consume) ».
- **Ce que fait le code** : la SRO n'est écrite que si **les deux** entrées sont des
  répertoires DICOM (`_is_dicom_dir(fixed.source) and _is_dicom_dir(moving.source)`).
  Pour une paire NIfTI — le cas de tous les exemples du README §Usage sauf un, et le cas
  de la CI — seuls 3 formats sortent, sans que rien ne le signale : l'échec éventuel est
  capté par un `except Exception` réduit à `log.warning` (`pipeline.py:970`), et l'absence
  de la condition ne produit **aucun** message.
- **Conséquences** : un utilisateur attend `transform/spatial_registration.dcm` et ne le
  trouve pas, sans explication.
- **Correction recommandée** : formuler « trois formats, plus une SRO DICOM lorsque les deux
  entrées sont des séries DICOM », et ajouter dans le pipeline un
  `manifest.warn("SRO non écrite : les deux entrées doivent être des séries DICOM")` quand
  `output.write_transform` est vrai et que la condition n'est pas remplie.
- **Tests nécessaires** : `test_sro_absence_is_reported_for_nifti_inputs`.

---

#### A-18 — Le docstring de `resample_to_spacing` affirme préserver l'étendue physique ; il préserve l'origine

- **Emplacement** : `regix/preprocess/geometry.py:58` ;
  `tests/test_units.py:755` (`test_resampling_preserves_the_physical_extent`).
- **Gravité** : **Cosmétique**
- **Ce que fait le code** : `new_size = round(src_size * src_spacing / target)` et
  l'origine est conservée telle quelle. L'étendue `new_size * target` diffère donc de
  l'étendue source d'au plus un voxel par axe (effet d'arrondi). C'est le comportement
  correct et attendu ; c'est le mot « preserving » qui est trop fort.
- **Conséquences** : aucune en pratique. Le test existant passe parce qu'il utilise une
  tolérance. Le risque est qu'un lecteur en déduise une garantie exacte.
- **Correction recommandée** : « Resample to a target spacing, keeping the origin and the
  direction; the physical extent is preserved to within one voxel per axis (rounding of
  the grid size). »
- **Tests nécessaires** : renommer le test existant en
  `test_resampling_preserves_the_physical_extent_to_within_one_voxel` et rendre la
  tolérance explicite dans son nom.

---

### B. Correction — bugs et comportements faux

---

#### B-01 — **Un masque « poumon gauche » ne couvre qu'un lobe** : collision de noms dans `ORGAN_ALIASES`

- **Emplacement** : `regix/organs/labels.py:38-42` (alias lobes → poumon) ;
  `regix/organs/segmenter.py:66-71` (`label_of`, renvoie **le premier** label) ;
  `regix/organs/segmenter.py:181-204` (`_from_directory`, un label par fichier).
- **Gravité** : **Critique**
- **Le mécanisme** : `ExternalSegmenter._from_directory` — utilisé aussi bien pour un
  répertoire de masques tiers **que pour la sortie de TotalSegmentator**
  (`segmenter.py:237`) — attribue un label entier distinct par fichier, puis nomme chaque
  label par `canonical_organ_name(fichier)`. Or les 5 fichiers de lobes de TotalSegmentator
  v2 se replient sur **2 noms seulement** :
  ```
  lung_upper_lobe_left  -> lung_left
  lung_lower_lobe_left  -> lung_left     <-- doublon
  lung_upper_lobe_right -> lung_right
  lung_middle_lobe_right-> lung_right    <-- doublon
  lung_lower_lobe_right -> lung_right    <-- doublon
  ```
  `label_of()` parcourt `label_names` et renvoie **le premier** label dont le nom
  correspond. `mask_for(["lung_left"])` ne collecte donc **qu'un seul label**.
- **[CONFIRMÉ]** — reproduction exacte sur un label map synthétique de 5 lobes de 200 voxels
  chacun :
  ```
  label_names        -> {1:'lung_left', 2:'lung_left', 3:'lung_right', 4:'lung_right', 5:'lung_right'}
  label_of('lung_left')            -> 1        (devrait couvrir 1 ET 2)
  mask_for(['lung_left'])          -> 200 voxels sur 400   <-- LA MOITIÉ DU POUMON
  present_organs()                 -> ['lung_left','lung_left','lung_right','lung_right','lung_right']
  organ_volumes_ml()               -> {'lung_left': 0.2, 'lung_right': 0.2}   (au lieu de 0.4 et 0.6)
  ```
- **Pourquoi c'est critique** : ce masque tronqué alimente **quatre** consommateurs
  simultanément, et aucun ne peut détecter l'anomalie :
  1. le **masque de critère** elastix (`combined_mask`) → elastix n'échantillonne que le
     lobe supérieur : la registration du poumon est optimisée sur la moitié de l'organe ;
  2. le **recadrage ROI** (`plan_roi`) → le volume est découpé autour d'un lobe, l'autre
     est purement et simplement exclu du calcul ;
  3. l'**initialisation** `organ_centroid` / `organ_moments` → centroïde du mauvais objet ;
  4. le **Dice de QC** → mesuré sur un lobe, présenté comme le Dice du poumon.
     Il sera **anormalement bon** (un lobe est plus facile à recaler qu'un poumon entier),
     donc la porte `min_dice: {lung_left: 0.95}` du preset `ct_ct_lung_4d` passera en
     donnant une fausse assurance.
- **Le preset `ct_ct_lung_4d` est directement touché** : `targets: [lung_left, lung_right]`,
  `backend: totalsegmentator`, `roi_crop: true`, `min_dice: 0.95` sur les deux poumons.
  C'est le scénario nominal de ce preset.
- **Aggravation** : `organ_volumes_ml` (contrôle de plausibilité annoncé par son propre
  docstring) écrase les doublons dans un dictionnaire → il **sous-déclare** le volume
  pulmonaire, donc le contrôle de plausibilité ne détecte pas le problème non plus.
- **Correction recommandée** :
  1. **`label_of` doit devenir `labels_of` et renvoyer une liste.** C'est la correction de
     fond. `mask_for` unit tous les labels retournés ; `organ_volumes_ml` somme ;
     `present_organs` déduplique.
     ```python
     def labels_of(self, organ: str) -> list[int]:
         key = canonical_organ_name(organ)
         return sorted(int(v) for v, name in self.label_names.items() if name == key)
     ```
     Conserver `label_of` comme `labels_of(...)[0] if ... else None`, mais **le retirer de
     tous les chemins de masquage** ; l'appel de `pipeline.py:744-746`
     (`moving_seg.label_of(name) == label`, filtre de nomenclature du Dice) doit devenir
     une comparaison d'ensembles.
  2. Alternative complémentaire : dans `_from_directory`, **fusionner** les fichiers dont le
     nom canonique coïncide dans un même label, plutôt que d'en créer plusieurs. C'est plus
     simple mais perd l'information lobaire ; à choisir explicitement.
  3. Dans tous les cas, ajouter un warning si `len(set(names.values())) != len(names)`.
- **Tests nécessaires** :
  - `test_a_directory_of_lung_lobes_yields_one_complete_lung_mask` (le test qui aurait
    attrapé ce bug : 5 fichiers de lobes → `mask_for(["lung_left"])` couvre les 2 lobes
    gauches) ;
  - `test_organ_volumes_sum_duplicate_names` ;
  - `test_present_organs_has_no_duplicates` ;
  - `test_dice_nomenclature_filter_handles_multi_label_organs` (intégration pipeline).

---

#### B-02 — Le traitement des volumes 4D produit une image **2D**, pas « le premier point temporel »

- **Emplacement** : `regix/io/volume.py:154-156`.
- **Gravité** : **Important**
- **Le code** :
  ```python
  if image.GetDimension() == 4:
      log.warning("4D volume detected (%s): extracting the first time point", p.name)
      image = image[..., 0]
  ```
- **[CONFIRMÉ]** — sur une image 4D SimpleITK de taille `(6,5,4,3)` :
  ```
  image[..., 0]  ->  taille (5, 4), dimension 2
  ```
  Le résultat est une **coupe 2D**, pas un volume 3D. L'`Ellipsis` de
  `SimpleITK.Image.__getitem__` ne se comporte pas comme en numpy : il consomme ici deux
  axes au lieu d'un.
- **Conséquences** :
  - le message de log **affirme** avoir extrait le premier point temporel : c'est faux, et
    c'est le pire cas — l'utilisateur est informé, mais mal ;
  - en aval, `sitk_to_itk` lève `ValueError: expected a 3D volume, got 2D`
    (`itk_bridge.py:68-69`), donc le run échoue — mais **tard**, après le chargement, la
    segmentation et le préprocessing, avec un message qui ne mentionne pas la 4D ;
  - pire, `Volume.describe()`, `reorient` (qui renvoie l'image inchangée si
    `GetDimension() != 3`) et `resample_to_spacing` (qui suppose 3 axes via `(float(s),)*3`)
    manipulent tous cette image 2D sans se plaindre.
- **Correction recommandée** :
  ```python
  if image.GetDimension() == 4:
      n_t = image.GetSize()[3]
      log.warning("volume 4D (%s, %d points temporels) : extraction du premier", p.name, n_t)
      image = sitk.Extract(image, [*image.GetSize()[:3], 0], [0, 0, 0, 0])
  ```
  `sitk.Extract` avec une taille nulle sur le dernier axe **effondre** cet axe et conserve
  origine/spacing/direction 3D. Vérifier explicitement `image.GetDimension() == 3` après.
  Ajouter aussi un refus explicite pour les dimensions 1, 2 et ≥ 5, qui passent
  actuellement sans contrôle.
- **Tests nécessaires** :
  - `test_a_4d_volume_yields_the_first_3d_time_point` : écrire un NIfTI 4D, vérifier
    `GetDimension() == 3`, la taille, et l'égalité pixel à pixel avec le volume t=0 ;
  - `test_a_2d_image_is_refused_with_a_clear_message`.

---

#### B-03 — `regix apply` sur le `.txt` que Regix écrit lui-même plante avec une exception ITK brute

- **Emplacement** : `regix/cli.py:477-480` ; `regix/registration/warp.py:56-71`
  (`ElastixAppliedTransform`) ; `regix/pipeline.py:926-932` (écriture de
  `transform/final_transform.txt`).
- **Gravité** : **Important**
- **Le mécanisme** : la commande route sur l'extension seule :
  ```python
  if transform.suffix.lower() == ".tfm":
      applied = SitkAppliedTransform(sitk.ReadTransform(str(transform)))
  else:
      applied = ElastixAppliedTransform(transform)      # <-- tout le reste
  ```
  Or un run de Regix écrit dans `transform/` **des `.txt` au format Insight** —
  `final_transform.txt`, `stage00_rigid.txt`, `stage01_affine.txt` — dont le README vante
  qu'ils « se chargent directement dans 3D Slicer » (`README.md:300`). Les donner à
  `regix apply` les envoie au parseur de fichiers de paramètres elastix.
- **[CONFIRMÉ]** — `regix apply final_transform.txt m.nii.gz --reference m.nii.gz -o w.nii.gz` :
  ```
  exit=1  RuntimeError
  D:\a\im\build\...\elx-src\Common\ParameterFileParser\itkParameterFileParser.cxx:42: ITK
  ```
  Le même fichier au format `.tfm` fonctionne (`exit=0`).
- **Conséquences** : le répertoire de sortie contient 5 fichiers `.txt` sur lesquels la
  commande officielle de propagation de contours échoue par *traceback*, sans message
  exploitable. C'est le scénario le plus probable pour un utilisateur : il ouvre
  `transform/`, voit `final_transform.txt`, le passe à `regix apply`.
- **Correction recommandée** : introduire `transforms.load_any_transform(path)` qui **renifle
  le contenu** plutôt que l'extension —
  ```python
  head = Path(path).read_text(errors="replace")[:4096]
  is_elastix = "(Transform " in head and "(TransformParameters " in head
  ```
  — puis router. Emballer les erreurs de lecture dans un
  `typer.BadParameter` explicitant les deux formats acceptés. Le même utilitaire corrige
  A-14. Ajouter `exists=True, dir_okay=False` aux arguments `Path` de `apply`.
- **Tests nécessaires** :
  - `test_apply_accepts_every_transform_file_a_run_produces` — paramétré sur
    `final_transform.tfm`, `final_transform.txt`, `stage00_rigid.txt`,
    `elastix/stage00_rigid/TransformParameters.0.txt`, avec vérification que les
    résultats de `.tfm` et `.txt` coïncident au voxel près ;
  - `test_apply_rejects_a_non_transform_file_with_a_clear_message`.

---

#### B-04 — Le garde-fou de quantification `_warn_on_quantisation` ne peut **jamais** se déclencher

- **Emplacement** : `regix/registration/engine.py:165-174` (reconstruction du `ParamContext`) ;
  `regix/registration/params.py:368-400` (`_warn_on_quantisation`) ;
  `regix/pipeline.py:337-347` (contexte correctement construit, avec `intensity_range`).
- **Gravité** : **Important**
- **Le mécanisme** : le pipeline construit un `ParamContext` complet, `intensity_range`
  compris :
  ```python
  context = ParamContext(..., intensity_range=_intensity_range(fixed_work.image))
  ```
  Puis `ElastixEngine.run` **reconstruit un contexte par stage**, en recopiant 8 champs sur 9 :
  ```python
  stage_ctx = ParamContext(dimension=..., n_channels=..., working_spacing_mm=...,
                           has_mask=..., fixed_modality=..., moving_modality=...,
                           features_available=..., n_voxels=...)
                           # intensity_range : OUBLIÉ -> None (valeur par défaut)
  ```
  Et `_warn_on_quantisation` commence par :
  ```python
  if ctx.intensity_range is None:
      return
  ```
- **[CONFIRMÉ]** : `"intensity_range" in inspect.getsource(ElastixEngine.run)` → **False**.
- **Pourquoi c'est important** : ce garde-fou est **le sujet d'une section entière du
  README** (§Native intensities reach elastix, `README.md:344-374`) et de 33 lignes de
  commentaire dans `params.py:104-112`. Il est présenté comme le filet de sécurité qui
  reste après la correction du bug de normalisation :
  > « `_warn_on_quantisation` below **still checks the combination**, because a user can
  > re-create it by asking for `normalize: minmax` explicitly. »
  Il ne vérifie rien.
- **Pourquoi les tests ne l'ont pas vu** :
  `tests/test_units.py:336::test_an_integer_pixel_type_on_rescaled_data_is_reported`
  appelle `build_parameter_map` **directement**, avec un `ParamContext` qu'il construit
  lui-même en fournissant `intensity_range`. Le test valide la fonction en isolation ;
  personne ne teste que le pipeline lui transmet la donnée. **C'est un trou de test
  d'intégration, pas un défaut de la fonction.**
- **Conséquences** : le scénario documenté (fichier de zoo en `short` + `normalize: minmax`
  demandé explicitement) repasse silencieux. Il reste rattrapé en aval par la porte
  `min_abs_final_metric` — c'est-à-dire par un **FAIL après coup**, sans le message
  d'explication qui aurait dit quoi faire.
- **Correction recommandée** :
  1. **ne pas reconstruire le contexte** : `stage_ctx = replace(context, n_channels=...,
     features_available=...)` (`dataclasses.replace`). C'est la correction robuste : elle
     supprime la classe entière de bugs « champ oublié à la recopie », et le
     `ParamContext` est déjà une dataclass ;
  2. par sécurité, faire de `intensity_range` un champ **sans valeur par défaut** pour
     forcer l'appelant à le fournir — ou logguer en `DEBUG` quand il est absent au lieu de
     sortir en silence.
- **Tests nécessaires** :
  - `test_the_quantisation_warning_fires_through_the_full_pipeline` (le test manquant) :
    run complet avec `normalize: minmax` + un `parameter_file` déclarant
    `(FixedInternalImagePixelType "short")`, `caplog` doit contenir le message ;
  - `test_stage_context_carries_every_field_of_the_pipeline_context` : test réflexif
    comparant les champs de `ParamContext` transmis.

---

#### B-05 — La porte `min_abs_final_metric` fait échouer à tort une registration MSE quasi parfaite

- **Emplacement** : `regix/qc/gates.py:141-173` ; `regix/config.py:341-352`.
- **Gravité** : **Important**
- **Le mécanisme** : la porte refuse tout stage dont `|final_metric| < 1e-6`, avec
  la justification :
  > « Not a quality threshold -- a floor: **any real criterion is above 1e-3** ».
  C'est vrai pour `AdvancedMattesMutualInformation` et pour
  `AdvancedNormalizedCorrelation` (elastix minimise, donc une bonne NCC tend vers **−1**).
  C'est **faux pour `AdvancedMeanSquares`** : une somme de carrés tend vers **0** quand
  l'alignement est parfait. Idem pour `features_mse` sur des canaux normalisés L2, dont
  l'amplitude est petite par construction.
- **[CONFIRMÉ]** :
  ```
  evaluate_gates(QCGates(), stages=[{"stage":"bspline","final_metric":3e-7}])  ->  FAIL
  ```
- **Conséquences** : `metric: mse` et `metric: features_mse` sont des valeurs
  documentées de l'énumération (`config.py:31-32`), mentionnées dans le README
  (« AdvancedMeanSquares »). Les utiliser sur une paire bien alignée — cas d'un recalage
  intra-session, ou de la seconde passe d'un raffinement — produit un **FAIL**, donc un
  `exit code 2` du CLI et un rapport rouge, sur un résultat correct. Le message affiché
  (« degenerate criterion: elastix reported success but optimised nothing ») envoie
  l'utilisateur enquêter sur un type de pixel entier inexistant.
- **Nuance** : aucun preset livré n'utilise `mse`, ce qui explique que le défaut n'ait
  jamais été rencontré. Le risque est donc latent, mais l'énumération est publique.
- **Correction recommandée** : rendre la porte **consciente du sens de la métrique**.
  `StageResult.description["metric_elastix"]` est déjà disponible et transmis
  (`pipeline.py:843` passe `s.to_dict() | {"final_metric": ...}`) :
  ```python
  ZERO_IS_OPTIMAL = {"AdvancedMeanSquares", "TransformBendingEnergyPenalty",
                     "DistancePreservingRigidityPenalty"}
  if stage.get("metric_elastix") in ZERO_IS_OPTIMAL:
      continue          # + un Check(WARN/PASS) explicite disant pourquoi la porte est inapplicable
  ```
  Documenter la limite dans la description du champ, qui affirme aujourd'hui une
  généralité fausse.
- **Tests nécessaires** :
  - `test_min_abs_final_metric_does_not_fail_a_perfect_mse_stage` ;
  - `test_min_abs_final_metric_still_catches_a_degenerate_mi_stage`
    (non-régression du cas Par0008) ;
  - test paramétré sur les 5 métriques de l'énumération.

---

#### B-06 — `TotalSegmentatorSegmenter` : `task` et `fast` ne sont pas configurables, ce qui rend la segmentation MR impossible

- **Emplacement** : `regix/organs/segmenter.py:213-226` (constructeur) ;
  `regix/organs/segmenter.py:371-376` (`build_segmenter`, n'en transmet aucun) ;
  `regix/config.py:184-217` (`OrganConfig`, ne les expose pas) ;
  `regix/organs/segmenter.py:8` (docstring : « CT (**and MR in v2**) »).
- **Gravité** : **Important**
- **Le mécanisme** : `build_segmenter` ne passe que `roi_subset`, `device` et `cache_dir`.
  `task` reste `"total"` et `fast` reste `True`, en dur.
- **Conséquences** :
  1. **Segmentation d'un volume MR impossible.** Le pipeline segmente **les deux** côtés
     (`pipeline.py:517`). Sur un preset comme `mr_ct_prostate` ou `ct_mr_abdomen` passé en
     `backend: totalsegmentator`, le côté MR est envoyé à `task="total"`, entraîné sur du
     CT. TotalSegmentator ne lèvera pas d'erreur : il produira des masques **plausibles et
     faux**, qui alimenteront ensuite le centroïde d'initialisation, le masque de critère,
     la ROI et le Dice. Le docstring du module annonce pourtant la prise en charge MR
     (`task="total_mr"` dans TotalSegmentator v2). C'est le cas le plus dangereux :
     **résultat silencieusement faux**, pas une erreur ;
  2. `fast=True` en dur impose le modèle 3 mm basse résolution ; aucun moyen d'obtenir la
     précision nominale, y compris quand elle importe (organes sub-centimétriques :
     surrénales, dont le profil existe pourtant dans `ORGAN_PROFILES`) ;
  3. `OrganConfig.device` **est** transmis mais n'apparaît nulle part dans la doc ;
     `regix segment` ne l'expose pas non plus (A-13).
- **Correction recommandée** :
  1. ajouter à `OrganConfig` : `ts_task: Literal["auto","total","total_mr"] = "auto"` et
     `ts_fast: bool = True`, avec descriptions ;
  2. `build_segmenter(config, side, cache_dir, modality)` — passer la modalité du volume et
     résoudre `"auto"` → `total_mr` si `modality == "MR"`, sinon `total` ;
  3. **refuser explicitement** (ou avertir fortement) `task="total"` sur un volume non-CT,
     plutôt que de produire des masques faux ;
  4. inclure `task` et `fast` dans `_cache_key` — c'est déjà le cas (`segmenter.py:285`),
     bien vu ;
  5. exposer `--organ-device` et `--fast/--no-fast` sur `regix segment`.
- **Tests nécessaires** :
  - `test_build_segmenter_selects_the_mr_task_for_an_mr_volume` (mock de l'API TS) ;
  - `test_segmenting_an_mr_with_the_ct_task_is_refused` ;
  - `test_ts_task_and_fast_reach_the_python_api` (mock, assertions sur les kwargs).

---

#### B-07 — L'échec d'un candidat d'initialisation **explicitement demandé** est avalé et remplacé en silence

- **Emplacement** : `regix/registration/initialize.py:264-296`.
- **Gravité** : **Important**
- **Le code** :
  ```python
  for mode in modes:
      try:
          ...
      except Exception as exc:
          log.warning("candidate '%s' discarded: %s", mode.value, exc)
  if not candidates:
      log.warning("no candidate could be built: falling back to grid-centre alignment")
      candidates.append(InitCandidate("geometry", geometry_init(fixed, moving)))
  ```
- **Le problème** : la boucle traite identiquement deux situations très différentes :
  - en `multistart`, l'échec d'**un** candidat parmi plusieurs est normal et doit être
    absorbé — le comportement actuel est correct ;
  - en mode **unique** (`mode=organ_centroid`, `mode=file`, …), `modes` ne contient qu'une
    entrée. Son échec vide la liste, et le repli produit une initialisation
    **complètement différente de celle demandée**, avec pour seule trace deux `log.warning`
    (pas même un `manifest.warn`, donc **rien dans le rapport HTML ni dans le manifeste**).
- **Chemins réels menant là** :
  - `mr_ct_prostate` (`init.mode: organ_centroid`, `backend: external`) lancé sans
    `--fixed-mask/--moving-mask` : `build_segmenter` renvoie `None`, `fixed_seg` est `None`,
    le candidat lève `ValueError("segmentations are required for organ_centroid")` → repli
    géométrie ;
  - `ct_ct_liver_followup` (`init.mode: organ_centroid`, `backend: totalsegmentator`) sur
    une installation cœur (TotalSegmentator absent, ce qui est le défaut de
    `pip install -e .`) : même issue ;
  - `init.mode=file` avec un fichier illisible (A-14).
- **Conséquences** : la §« Initialization is half the work » du README insiste sur le fait
  que le point de départ *est* le problème. Le remplacer silencieusement, sur les presets
  qui en dépendent le plus, contredit le principe affiché n°3 (« A failure is labelled,
  never hidden »). Le run se termine en PASS/WARN sans que le rapport mentionne que
  l'initialisation demandée n'a pas eu lieu.
- **Correction recommandée** :
  1. `build_candidates` doit recevoir le `manifest` (ou renvoyer les échecs) et les
     remonter en `manifest.warn`, donc dans le rapport ;
  2. quand `config.mode is not InitMode.MULTISTART` et que le candidat unique échoue :
     **lever** `RegistrationFailure`. Le principe affiché du projet est de ne pas
     substituer silencieusement ;
  3. ajouter `init_report["requested"] = config.mode.value` à côté de
     `init_report["chosen"]`, et une porte QC `initialization_as_requested` qui passe en
     WARN si les deux diffèrent. Le rapport a déjà une table d'initialisation
     (`report.py:457-469`) où l'afficher.
- **Tests nécessaires** :
  - `test_organ_centroid_without_segmentation_is_fatal_not_silently_replaced` ;
  - `test_multistart_still_tolerates_a_failing_candidate` (non-régression du comportement
    voulu — le test `test_unusable_candidate_is_dropped_not_fatal` existant couvre ce cas
    et devra être restreint au multistart) ;
  - `test_the_report_states_when_the_initialization_differs_from_the_request`.

---

#### B-08 — Installer torch **dégrade** le repli CPU : on perd MIND-SSC au profit des intensités brutes

- **Emplacement** : `regix/pipeline.py:583-623` (`_extract_features`) ;
  `regix/features/anatomix.py:56-72` (`resolve_device`) ; `README.md:252`.
- **Gravité** : **Important**
- **Affirmation** : « **automatic fallback** to MIND-SSC (analytical, CPU, 12 channels)
  when torch/anatomix are absent ».
- **Ce que fait le code** — trois cas, dont le second est contre-intuitif :

  | Situation | `anatomix_available()` | `provider` | Résultat |
  |---|---|---|---|
  | torch **absent** | `False` | `"mind"` | **MIND-SSC** — conforme |
  | torch + anatomix **présents**, **pas de GPU**, `allow_cpu=False` | `True` | `"auto"` → `"anatomix"` | `resolve_device` **lève** → capté par le `except Exception` de `_extract_features` → `return None, None, {"provider":"none"}` → **registration sur les intensités brutes** |
  | torch + GPU | `True` | `"anatomix"` | anatomix — conforme |

- **Pourquoi c'est un problème** : la deuxième ligne est un **régression par ajout de
  dépendance**. Un site qui installe `regix[features]` sur des postes sans GPU passe de
  MIND-SSC (12 canaux invariants au contraste, exactement le repli conçu pour ce cas) à de
  la simple MI/NCC sur intensités. C'est l'inverse de l'intention, et l'affirmation du
  README ne couvre pas ce cas puisqu'elle est formulée sur « absent ».
- **Aggravation** : le message de log est trompeur —
  « feature extraction failed (RuntimeError: no GPU detected…) -- **falling back to
  intensities** » alors que MIND-SSC est disponible, installé, sans dépendance, et à deux
  lignes de là.
- **Correction recommandée** : dans `_extract_features`, remplacer le repli global par un
  repli **en cascade** :
  ```python
  for provider in (chosen, "mind"):       # mind toujours en dernier recours
      try:
          pair = extract_feature_pair(..., provider=provider, ...)
          break
      except Exception as exc:
          manifest.warn(f"features via {provider} indisponibles ({exc}); repli suivant")
  else:
      return None, None, {"provider": "none", ...}
  ```
  et ne renvoyer `provider: none` que si MIND lui aussi échoue. Corriger le README pour
  décrire la cascade réelle.
- **Tests nécessaires** :
  - `test_no_gpu_falls_back_to_mind_not_to_intensities` (monkeypatch `resolve_device` pour
    lever) — c'est le test qui manque ;
  - `test_features_provider_is_recorded_in_the_manifest` sur les trois issues.

---

#### B-09 — `--overwrite` n'efface rien : les sorties d'un run précédent survivent et sont présentées comme actuelles

- **Emplacement** : `regix/pipeline.py:165-169`.
- **Gravité** : **Important**
- **Le code** :
  ```python
  if out_dir.exists() and any(out_dir.iterdir()) and not cfg.output.overwrite:
      raise FileExistsError(...)
  out_dir.mkdir(parents=True, exist_ok=True)
  ```
  `overwrite` désactive le garde-fou ; **aucun nettoyage n'a lieu**.
- **Scénarios de corruption** *(mécanisme **[LECTURE]**, tracé de bout en bout)* :
  1. run n°1 : preset à 3 stages → `transform/stage02_bspline.txt`,
     `elastix/stage02_bspline/`, `deformation_field.nii.gz`.
     Run n°2 dans le même répertoire avec `--rigid-only --overwrite` → écrit
     `stage00_rigid.*`, **laisse `stage02_bspline.txt` en place**. Le répertoire
     `transform/` décrit alors une chaîne à 3 stages qui n'a pas eu lieu, à côté d'un
     `run_manifest.json` qui en décrit une à 1 stage ;
  2. `masks/moving_labelmap_registered.nii.gz`, `features/`, `jacobian.nii.gz`,
     `dicom_registered/*.dcm` : tous conditionnels, tous persistants ;
  3. `dicom_registered/` est le plus grave : les fichiers sont nommés
     `regix_00001.dcm … regix_NNNNN.dcm`. Un run n°2 produisant **moins de coupes** laisse
     les coupes surnuméraires du run n°1, avec un `SeriesInstanceUID` différent — mais dans
     le même répertoire, prêtes à être envoyées au PACS ;
  4. `regix.log` est ouvert par `logging.FileHandler` en mode **append** par défaut : les
     journaux de plusieurs runs s'accumulent dans un fichier que le manifeste présente
     comme celui du run.
- **Conséquences** : traçabilité rompue et, dans le cas (3), risque d'envoi de coupes
  hétérogènes vers un système clinique. Contredit directement l'argument « what you re-read
  six months later to know what actually ran ».
- **Correction recommandée** :
  1. lorsque `overwrite` est vrai, **supprimer** les artefacts connus produits par Regix
     avant d'écrire — liste blanche explicite (`transform/`, `elastix/`, `masks/`,
     `features/`, `dicom_registered/`, `report.html`, `run_manifest.json`, `regix.log`,
     `moving_registered.nii*`, `config_effective.yaml`, `deformation_field.nii.gz`,
     `jacobian.nii.gz`, `cache/`). **Ne jamais** faire un `rmtree(out_dir)` : l'utilisateur
     peut avoir pointé un répertoire partagé ;
  2. ouvrir `regix.log` en mode `"w"` (ou nommer `regix-{run_id}.log`) ;
  3. écrire `run_manifest.json` **en dernier** et y lister exhaustivement les fichiers
     produits, pour qu'un fichier orphelin soit détectable.
- **Tests nécessaires** :
  - `test_overwrite_removes_stale_artifacts_from_a_previous_run` : run 3 stages, puis run
    `--rigid-only --overwrite`, assertion que `transform/stage02_bspline.txt` a disparu ;
  - `test_overwrite_does_not_delete_unrelated_user_files` (un `notes.txt` déposé par
    l'utilisateur doit survivre) ;
  - `test_log_file_contains_only_the_current_run`.

---

#### B-10 — `_deep_update` ignore les valeurs `None` : impossible d'annuler une option par héritage de preset, par `with_overrides` ou par l'API HTTP

- **Emplacement** : `regix/config.py:487-495` (`_deep_update`) ;
  utilisé par `with_overrides` (`config.py:469-473`) et `_build_from_raw` (`config.py:536`).
- **Gravité** : **Important**
- **Le code** :
  ```python
  for key, value in updates.items():
      if value is None:
          continue          # <-- une valeur None n'écrase jamais rien
  ```
- **[CONFIRMÉ]** :
  ```
  base.with_overrides(preprocess={"working_spacing_mm": None}).preprocess.working_spacing_mm
      -> 2.0                (attendu : None)
  base.with_overrides(qc={"gates":{"min_abs_final_metric": None}}) ...min_abs_final_metric
      -> 1e-06              (la doc dit : « Set to null to disable »)
  load_preset("ct_cbct_igrt").preprocess.fixed.percentile_clip
      -> 'auto'             (le YAML déclare : percentile_clip: null)
  ```
- **Options rendues inatteignables**, alors que leur documentation décrit explicitement
  la valeur `null` :
  | Champ | Doc du champ |
  |---|---|
  | `preprocess.working_spacing_mm` | « **None = no resampling**. » |
  | `preprocess.orientation` | « **None = leave as is**. » |
  | `qc.gates.min_abs_final_metric` | « **Set to null to disable.** » |
  | `qc.gates.max_translation_mm`, `max_scale_deviation`, `max_tre_mm`, `min_ncc_gain`, `min_nmi_gain` | portes désactivables par `null` |
  | `preprocess.<side>.percentile_clip` | « **null means no percentile clipping at all** » — les 3 états documentés sont réduits à 2 |
  | `runtime.threads` | « None = let ITK decide. » |
- **Incohérence entre les deux mécanismes d'override** — `--set` **peut** poser `null`
  parce qu'il opère directement sur le dictionnaire, sans passer par `_deep_update` :
  ```
  --set qc.gates.min_ncc_gain=null   -> exit 0, valeur bien mise à None   [CONFIRMÉ]
  ```
  Deux chemins d'override documentés côte à côte dans le README, avec des sémantiques
  différentes. L'API HTTP (`api._build_config` → `with_overrides`) est du mauvais côté :
  **aucun client HTTP ne peut désactiver une porte QC**.
- **Circonstance atténuante mesurée** : dans les 8 presets livrés, les 6 occurrences de
  `percentile_clip: null` sont aujourd'hui **sans effet observable**, parce que
  `resolve_prep` aboutit au même résultat par un autre chemin (soit une `window` est
  déclarée et l'emporte, soit la modalité est CT dont le défaut est déjà `None`).
  **[CONFIRMÉ]** — clipping effectif résolu identique dans les 4 presets testés. Le bug est
  donc aujourd'hui latent, mais c'est un hasard : il se réveillera à la première déclaration
  `null` qui compte.
- **Pourquoi la correction n'est pas triviale** : `_deep_update` ignore `None` précisément
  pour que `cli.register` puisse construire son dictionnaire d'overrides avec des options
  typer non renseignées. Mais `cli.register` **ne met déjà que les clés renseignées**
  (`if fixed_modality: overrides[...] = ...`) : la protection est redondante côté CLI.
- **Correction recommandée** :
  1. supprimer le `if value is None: continue` de `_deep_update` ;
  2. auditer les appelants — `cli.register` construit son dict conditionnellement (OK),
     `cli.batch` idem (OK), `api._build_config` idem (OK). Le risque de régression est
     faible et couvert par la suite existante ;
  3. si une compatibilité est nécessaire, introduire un sentinel explicite
     (`UNSET`) plutôt que de surcharger `None`, qui est une **valeur légitime** dans ce
     schéma ;
  4. supprimer aussi la ligne redondante `merged["stages"] = raw["stages"]`
     (`config.py:538-539`) : `_deep_update` fait déjà ce remplacement pour une liste
     (code mort, cf. E-03).
- **Tests nécessaires** :
  - `test_a_child_preset_can_disable_a_parent_option_with_null` (le test manquant) ;
  - `test_with_overrides_can_set_a_value_to_none` ;
  - `test_set_and_with_overrides_agree_on_null` — test de parité entre les deux
    mécanismes, paramétré sur les 8 champs listés ci-dessus ;
  - `test_disabling_a_qc_gate_through_the_api_works`.

---

#### B-11 — `write_initial_transform_file` force `ComputeZYX false` sans vérifier la convention de la source

- **Emplacement** : `regix/registration/transforms.py:152` ; `:185-189`
  (`transform_to_elastix_initial`, branche `Euler3DTransform`).
- **Gravité** : **Important** *(probabilité faible, conséquence silencieuse)*
- **Le mécanisme** : `transform_to_elastix_initial` détecte un `sitk.Euler3DTransform` et
  transmet `list(e.GetParameters())` — trois angles puis trois translations. Le fichier
  écrit déclare en dur `(ComputeZYX "false")`. Or l'ordre de composition des rotations
  d'un `Euler3DTransform` dépend de son drapeau `GetComputeZYX()` : les **mêmes trois
  angles** décrivent deux rotations différentes selon sa valeur.
- **Est-ce atteignable ?** Toutes les Euler construites *par Regix* posent
  `SetComputeZYX(False)` (`initialize.py:79`, et `conftest.known_rigid`) : le chemin nominal
  est sûr. Le chemin atteignable est `init.mode=file` avec un `.tfm` produit ailleurs
  (3D Slicer, un script maison, un TPS) dont l'Euler est en ZYX. `sitk.ReadTransform`
  restaure fidèlement le drapeau ; `transform_to_elastix_initial` le jette.
- **Conséquences** : initialisation géométriquement fausse, **sans aucun signal**. Elastix
  partira d'un point erroné ; l'optimiseur convergera peut-être quand même, et la
  transformée rapportée sera cohérente — donc le rapport sera vert. Sur des angles faibles
  l'écart est faible ; sur une rotation franche il est majeur.
- **Correction recommandée** :
  ```python
  if isinstance(transform, sitk.Euler3DTransform):
      e = sitk.Euler3DTransform(transform)
      if e.GetComputeZYX():
          # convertir en affine plutôt que de mentir sur la convention
          return transform_to_elastix_initial(_to_affine(e), fixed_image, path)
      ...
  ```
  Le plus simple et le plus sûr : **supprimer la branche `Euler3DTransform`** et passer
  systématiquement par `sitk.AffineTransform(transform)`, qui est invariant à la
  convention d'angles. La branche Euler n'apporte qu'une lisibilité marginale du fichier.
  À défaut, propager `compute_zyx` en paramètre de `write_initial_transform_file`.
- **Tests nécessaires** :
  - `test_a_zyx_euler_initial_transform_round_trips` : construire une Euler avec
    `SetComputeZYX(True)` et des angles franchement différents (30°, 20°, 40°), l'écrire
    via `transform_to_elastix_initial`, la relire via `parameter_map_to_transform`, et
    comparer sur 20 points — écart < 1e-6 mm ;
  - `test_regix_never_produces_a_zyx_euler` (verrou sur le chemin nominal).

---

#### B-12 — Un organe sorti du champ après recalage n'est pas noté FAIL mais WARN

- **Emplacement** : `regix/qc/metrics.py:190-191` ; `regix/qc/gates.py:176-188`.
- **Gravité** : **Important**
- **Le mécanisme** :
  ```python
  # organ_overlap_report
  if not np.any(f_arr == label) or not np.any(w_arr == label):
      continue                      # l'organe est simplement omis du rapport
  ```
  Puis, dans `evaluate_gates`, l'absence d'entrée pour un organe sous `min_dice` produit :
  ```python
  Check(f"dice[{organ}]", WARN, None, threshold, "Dice for {organ} unavailable (organ missing from one of the volumes)")
  ```
- **Le problème** : deux situations radicalement différentes reçoivent le même verdict.
  - **Légitime** : l'organe n'a jamais été segmenté du côté mobile (FOV plus petit,
    segmenteur qui l'a manqué) → WARN est correct ;
  - **Catastrophique** : l'organe **était** présent dans le label map mobile, mais la
    transformée l'a projeté **hors de la grille fixe**. `warped_labelmap` ne le contient
    plus, `np.any(w_arr == label)` est faux, la porte devient WARN.
    C'est exactement l'échec de recalage le plus grave — un Dice de **0** — et il est
    rétrogradé de FAIL en WARN.
- **Conséquences** : `regix register` renvoie le code de sortie 0 (et non 2) pour un
  recalage qui a expulsé l'organe cible. Le principe affiché n°3 (« A failure is labelled »)
  n'est pas tenu sur son cas le plus important. Le preset `ct_ct_liver_followup`
  (`min_dice: {liver: 0.92}`) est concerné.
- **Circonstance atténuante** : d'autres portes attraperaient probablement une divergence
  aussi grossière (`max_translation_mm`, `ncc_gain`). Mais une translation *plausible*
  combinée à un organe en bord de champ ne les déclenche pas.
- **Correction recommandée** : distinguer les deux cas dans `organ_overlap_report` en
  renvoyant une entrée explicite plutôt qu'en sautant :
  ```python
  present_fixed, present_warped = np.any(f_arr == label), np.any(w_arr == label)
  if not present_fixed:
      continue                                        # rien à comparer : légitime
  if not present_warped:
      out[name] = {"dice": 0.0, "hd95_mm": float("nan"), "msd_mm": float("nan"),
                   "reason": "organ absent from the registered volume"}
      continue
  ```
  La porte produira alors `dice=0.0 < 0.92` → **FAIL**, avec le bon message.
- **Tests nécessaires** :
  - `test_an_organ_pushed_out_of_the_field_fails_the_dice_gate` (translation massive
    imposée sur un fantôme, assertion `status == "FAIL"` et `dice == 0.0`) ;
  - `test_an_organ_never_segmented_on_the_moving_side_only_warns` (non-régression).

---

#### B-13 — `evaluate_gates(deformable=...)` reçoit une expression sémantiquement fausse

- **Emplacement** : `regix/pipeline.py:842` ; `regix/qc/gates.py:229-238`.
- **Gravité** : **Mineur**
- **Le code** : `deformable=outcome.is_deformable or applied.kind == "sitk"`.
  Or `applied.kind == "sitk"` est vrai pour **toute** chaîne convertie avec succès —
  y compris une chaîne purement rigide. Le paramètre nommé `deformable` vaut donc
  quasiment toujours `True`, chemin nominal inclus.
- **Conséquence observable** : si le champ de déplacement est indisponible sur un recalage
  **rigide**, la porte ajoute
  `Check("folding_fraction", WARN, ..., "Jacobian not computed although the transform is
  deformable")` — un message faux, sur un recalage où le repliement est impossible par
  construction.
- **Correction recommandée** : passer `deformable=outcome.is_deformable or
  cfg.deformable_engine is DeformableEngine.CONVEXADAM`, qui est la question réellement
  posée : « la transformée finale peut-elle replier l'espace ? ». Renommer le paramètre en
  `can_fold` clarifierait davantage.
- **Tests nécessaires** : `test_a_rigid_run_without_a_jacobian_does_not_warn_about_folding`.

---

#### B-14 — Une série DICOM éclatée sur plusieurs sous-répertoires est chargée **tronquée**, sans avertissement

- **Emplacement** : `regix/io/dicom.py:65-92` (`list_series`).
- **Gravité** : **Important** *(mécanisme **[LECTURE]**, non reproduit faute de données)*
- **Le mécanisme** :
  ```python
  directories = [d] + [p for p in d.rglob("*") if p.is_dir()]
  seen_uids: set[str] = set()
  for sub in directories:
      for uid in reader.GetGDCMSeriesIDs(str(sub)):
          if uid in seen_uids: continue                    # <-- ici
          files = reader.GetGDCMSeriesFileNames(str(sub), uid)   # <-- NON récursif
          seen_uids.add(uid)
          found.append(_probe_series(uid, sub, list(files)))
  ```
  `GetGDCMSeriesFileNames(sub, uid)` ne liste que les fichiers **de `sub`**. Si les coupes
  d'une même série sont réparties entre `sub1/` et `sub2/`, la série est enregistrée avec
  les seuls fichiers de `sub1`, et `sub2` est ignoré par le `seen_uids`.
- **Est-ce réaliste ?** Oui : exports PACS par lots, archives découpées par un outil de
  transfert, répertoires `IM001-100/`, `IM101-200/`. C'est un cas d'usage explicitement
  visé par le projet (« several series in the same directory … because it happens all the
  time in production »).
- **Conséquences** : volume amputé, chargé sans erreur. Le seul filet est
  `_check_slice_regularity`, qui signalera peut-être un `max_gap_mm` élevé — mais seulement
  si la coupure crée un trou *interne*. Une coupure aux extrémités passe totalement
  inaperçue. Toutes les métriques et la transformée porteront sur un demi-volume.
- **Correction recommandée** : inverser la boucle — collecter les fichiers **par UID à
  travers tous les répertoires**, puis fusionner :
  ```python
  by_uid: dict[str, list[str]] = defaultdict(list)
  for sub in directories:
      for uid in reader.GetGDCMSeriesIDs(str(sub)):
          by_uid[uid].extend(reader.GetGDCMSeriesFileNames(str(sub), uid))
  # dédupliquer, puis retrier par ImagePositionPatient (GDCM le fait par répertoire seulement)
  ```
  Le retri global est indispensable après fusion : les listes concaténées ne sont plus
  ordonnées. Ajouter un warning quand une série couvre plus d'un répertoire.
- **Tests nécessaires** :
  - `test_a_series_split_across_two_directories_is_loaded_whole` : générer 20 coupes
    synthétiques, en écrire 10 dans `a/` et 10 dans `b/`, vérifier
    `n_slices == 20` et la monotonie des positions ;
  - `test_slice_order_is_correct_after_merging_directories`.

---

#### B-15 — Le message « pass the UID explicitly » désigne une capacité qui n'existe pas

- **Emplacement** : `regix/io/volume.py:142-150` ; `regix/io/dicom.py:146-164`
  (`load_series` **accepte** `series_uid`) ; `regix/cli.py` et `regix/api.py`
  (aucun moyen de le fournir).
- **Gravité** : **Important**
- **Le message** :
  > « %d DICOM series in %s; selecting the largest one (%s, %d slices).
  > Use `regix inspect` and **pass the UID explicitly** to remove the ambiguity. »
- **[CONFIRMÉ]** : `load_series(series, series_uid=...)` implémente bien la sélection par
  UID, mais `load_volume` **ne transmet jamais** le paramètre (`volume.py:151` :
  `load_series(series[0], ...)`), et ni le CLI (`register`, `batch`, `apply`, `segment`) ni
  l'API HTTP (`RegisterRequest`) n'exposent de champ UID.
- **Pourquoi c'est important, au-delà du message** : l'ambiguïté est **réelle et
  dangereuse**. Le tri est `found.sort(key=n_files, reverse=True)` — un tri **non total** :
  deux séries de même nombre de coupes (fréquent : reconstructions à noyaux différents,
  phases d'un 4D-CT, séries avec et sans contraste) sont départagées par l'ordre de
  `rglob`, donc par le système de fichiers. Regix choisira l'une ou l'autre, et
  l'utilisateur n'a **aucun moyen** de forcer son choix, alors que le log lui dit de le
  faire.
- **Conséquences** : sur une étude multi-séries, un recalage peut porter sur la mauvaise
  série (phase artérielle au lieu de portale, par exemple) sans recours possible autre que
  copier les fichiers dans un répertoire à part. Le manifeste enregistre bien le
  `series_uid` choisi (`Volume.series_uid`) — encore faut-il le relire.
- **Correction recommandée** :
  1. propager le paramètre : `load_volume(path, ..., series_uid: str | None = None)` →
     `load_series(..., series_uid=series_uid)` ;
  2. exposer `--fixed-series-uid` / `--moving-series-uid` sur `register`, les colonnes
     correspondantes sur `batch`, et les champs sur `RegisterRequest` ;
  3. rendre le tri **total et déterministe** :
     `sort(key=lambda s: (-s.n_files, s.acquisition_date or "", s.series_uid))` ;
  4. élever le message au rang de `manifest.warn` (aujourd'hui simple `log.warning` : il
     n'apparaît **pas** dans le rapport HTML ni dans le manifeste, alors que c'est
     exactement le genre d'ambiguïté qu'un relecteur doit voir).
- **Tests nécessaires** :
  - `test_series_uid_selects_the_requested_series_through_the_cli` ;
  - `test_series_ordering_is_deterministic_for_equal_slice_counts` ;
  - `test_multi_series_ambiguity_reaches_the_manifest`.

---

#### B-16 — Le score multi-start mélange NCC et NMI même sur une paire multimodale

- **Emplacement** : `regix/registration/initialize.py:218-239` (`score_candidate`,
  défaut `metric="auto"`) ; `:362` (appelé sans jamais préciser la métrique).
- **Gravité** : **Mineur**
- **Le code** : en mode `auto`, le score est la moyenne de la NCC et de (NMI − 1).
  `choose_initialization` n'a pas accès aux modalités et n'en passe jamais.
- **Le problème** : sur une paire CT↔MR — soit précisément le cas où le multi-start est le
  plus utile, et le cas des presets `ct_mr_abdomen`, `mr_ct_prostate`, `pet_ct_wholebody` —
  la NCC entre intensités CT et MR est dépourvue de sens. Sa valeur n'est pas nulle, elle
  est **arbitraire** : elle dépend de la corrélation fortuite des deux histogrammes dans le
  masque. Elle pèse pour moitié dans le classement des candidats.
- **Conséquences** : classement des points de départ partiellement bruité sur les paires
  multimodales, avec un risque accru de déclencher le warning « the two best starting points
  are tied » — ou pire, de ne pas le déclencher alors que le classement est arbitraire.
  Le README affirme que les candidats sont « scored on downsampled images by a metric
  **independent** of the optimiser » : c'est vrai (le mérite est réel), mais indépendant
  n'est pas pertinent.
- **Correction recommandée** : propager la modalité. `choose_initialization` reçoit déjà
  `fixed: Volume` et `moving: Volume` — il suffit d'appeler
  `same_modality(fixed.modality, moving.modality)` (déjà importable depuis
  `registration.params`) et de passer `metric="ncc"` ou `"nmi"`. Consigner la métrique
  retenue dans `init_report["scoring_metric"]` pour qu'elle apparaisse au rapport.
- **Tests nécessaires** :
  - `test_multistart_scores_a_multimodal_pair_with_nmi_only` ;
  - étendre `test_candidate_scoring_prefers_the_true_alignment` (déjà paramétré sur
    ncc/nmi/auto) à une paire CT/« MR » à contraste inversé, où `auto` doit désormais
    résoudre en `nmi`.

---

### C. Sécurité et protection des données patient

---

#### C-01 — Le sel de pseudonymisation par défaut est la constante publique `"regix"`, et le docstring affirme l'irréversibilité

- **Emplacement** : `regix/logging_utils.py:72-78` ; `README.md:393-394` ;
  `regix/config.py:410` (`pseudonymize: bool = True`).
- **Gravité** : **Critique**
- **Le code** :
  ```python
  def pseudonymize(value, salt=None, length=10) -> str:
      """Truncated salted hash of an identifier. Never reversible without the salt."""
      salt = salt if salt is not None else os.environ.get("REGIX_PSEUDONYM_SALT", "regix")
      digest = hashlib.sha256(f"{salt}::{value}".encode()).hexdigest()
      return digest[:length]
  ```
- **Trois défauts cumulés** :
  1. **Le sel par défaut est `"regix"`**, une constante publique du dépôt. Sans variable
     d'environnement — c'est-à-dire dans l'installation par défaut, en CI, et sur tout poste
     où personne n'a lu le README — la pseudonymisation est un simple SHA-256 tronqué à sel
     connu.
  2. **L'espace des identifiants patients est minuscule.** Un `PatientID` hospitalier est
     typiquement un entier de 6 à 10 chiffres ou un alphanumérique court. Énumérer
     10⁸ candidats et comparer 10 caractères hexadécimaux est une opération de quelques
     secondes sur un poste de bureau. **La ré-identification est immédiate.**
  3. **Le docstring affirme le contraire** : « Never reversible without the salt » — vrai
     au sens cryptographique de la préimage, **faux** au sens qui compte ici (attaque par
     dictionnaire sur un espace de recherche connu et petit). Aucune mise en garde n'est
     émise quand le sel par défaut est utilisé.
- **Aggravation par la troncature** : 10 caractères hexadécimaux = 40 bits. Même avec un
  sel secret, une collision devient probable vers 2²⁰ ≈ 10⁶ identifiants (paradoxe des
  anniversaires) — deux patients distincts partageant le même pseudonyme dans une base
  d'un million. Pour un usage de traçabilité clinique, c'est court.
- **Affirmation du README** : « **Privacy.** No patient identifier in clear text in logs or
  reports: pseudonymisation by salted hash (`REGIX_PSEUDONYM_SALT`), **verified by test**. »
  Le test existe bien (`test_dicom_io.py:137::test_patient_identifier_is_pseudonymised`),
  mais il vérifie que la sortie **diffère** de l'entrée — pas qu'elle résiste à une attaque.
  Une affirmation de sécurité « vérifiée par test » par un test qui ne teste pas la
  propriété de sécurité est plus dangereuse qu'aucun test.
- **Conséquences** : un `run_manifest.json` ou un `report.html` transmis hors du périmètre
  de soin (par courriel — ce que le README encourage explicitement : « can be emailed »)
  contient un identifiant ré-identifiable en quelques secondes. Sous RGPD, une
  pseudonymisation réversible par un tiers sans effort déraisonnable ne fait pas sortir la
  donnée du régime des données de santé.
- **Correction recommandée** :
  1. **Ne pas fournir de sel par défaut.** Si `REGIX_PSEUDONYM_SALT` est absent :
     - soit générer un sel aléatoire par run et le journaliser **hors** des sorties
       (l'utilisateur perd la corrélation inter-runs, ce qui est le comportement sûr),
     - soit refuser de démarrer quand `runtime.pseudonymize` est vrai, avec un message
       indiquant comment définir le sel.
     Dans tous les cas, **émettre un `manifest.warn` visible** quand le sel est faible ou
     absent, pour que le rapport porte l'information.
  2. Passer à un **HMAC-SHA256** (`hmac.new(salt.encode(), value.encode(), 'sha256')`)
     plutôt qu'à une concaténation `salt::value`, et **ne pas tronquer sous 128 bits**
     (32 caractères hexadécimaux) — la longueur n'a aucun coût ici.
  3. Documenter dans le README que le pseudonyme est **stable pour un sel donné** (c'est la
     propriété utile pour corréler des runs) et que le sel doit être traité comme un secret,
     conservé hors du dépôt et hors des sorties.
  4. Corriger le docstring : « Résistant à l'inversion tant que le sel reste secret ; sans
     sel secret, un identifiant patient est énumérable en quelques secondes. »
- **Tests nécessaires** :
  - `test_the_default_salt_is_refused_or_warned_about` : sans `REGIX_PSEUDONYM_SALT`, soit
    une exception, soit un warning contenant « salt » dans `manifest.warnings` ;
  - `test_pseudonym_is_stable_for_a_given_salt_and_differs_across_salts` ;
  - `test_pseudonym_length_is_at_least_32_hex_chars` ;
  - **le test de sécurité qui manque** :
    `test_a_short_numeric_patient_id_is_not_trivially_enumerable` — énumérer 10⁵
    identifiants avec le sel *par défaut* et vérifier qu'aucun ne retrouve le pseudonyme
    (ce test **doit échouer** aujourd'hui, et passer après correction).

---

#### C-02 — Les chemins des données source figurent en clair dans `run_manifest.json`, dans `config_effective.yaml` et dans `regix.log`

- **Emplacement** : `regix/io/volume.py:102` (`describe()["source"]`) ;
  `regix/pipeline.py:211-213` (`manifest.inputs`) ; `regix/pipeline.py:175, 265`
  (`manifest.config`) ; `regix/pipeline.py:437` (`config_effective.yaml`) ;
  `regix/io/dicom.py:143-150` (log du chemin) ; `README.md:393-394`.
- **Gravité** : **Important**
- **[CONFIRMÉ]** sur le `run_manifest.json` du run réel présent dans l'arbre :
  ```
  inputs.fixed.source  = C:\Users\thibault.escobar\Desktop\dataregix\Fixed
  inputs.moving.source = C:\Users\thibault.escobar\Desktop\dataregix\Moving
  ```
- **Pourquoi c'est un problème** : dans un service d'imagerie, les chemins **sont** des
  identifiants. `\\nas\etudes\DUPONT_Jean_19540312\CT_20240115\` est une donnée de santé
  nominative complète — nom, date de naissance, date d'examen. Regix pseudonymise
  scrupuleusement le `subject_id` puis écrit le chemin à côté.
  Sont concernés :
  - `run_manifest.json` → `inputs.{fixed,moving}.source`, et `config` (chemins de masques,
    de landmarks, de checkpoint, `output.dir`) ;
  - `config_effective.yaml` → mêmes chemins ;
  - `regix.log` → `list_series` journalise le répertoire ; toute exception capturée
    (`manifest.warn(f"... ({type(exc).__name__}: {exc})")`) embarque le chemin du fichier
    fautif ;
  - **`report.html`** → par le canal `warnings` (`report.py:481-484`), qui recopie
    `manifest.warnings`. Une `FileNotFoundError` sur un fichier de landmarks fait entrer le
    chemin patient dans le rapport destiné à être **envoyé par courriel**.
- **Vérification du rapport** : **[CONFIRMÉ]** sur le `report.html` du run présent —
  aucun chemin n'y figure, parce que ce run n'a produit **aucun warning**. La fuite est
  donc conditionnelle à la survenue d'un avertissement, ce qui la rend d'autant plus
  insidieuse : elle n'apparaît pas sur les runs nominaux, donc pas en test.
- **Statut de l'affirmation README** : « No patient identifier in clear text in **logs or
  reports** ». Le manifeste et `config_effective.yaml` ne sont littéralement ni des logs ni
  des rapports — mais `regix.log` **est** un log, et `report.html` **est** le rapport.
  L'affirmation est donc fausse pour deux des quatre artefacts, de façon conditionnelle.
- **Correction recommandée** :
  1. introduire `logging_utils.redact_path(p)` : conserver le **nom de fichier terminal**
     et le **suffixe**, remplacer les composants parents par leur pseudonyme
     (`<hash>/CT_series`), et l'appliquer systématiquement à `describe()["source"]`, à
     `manifest.config`, et au formatage des exceptions rentrant dans `manifest.warn` ;
  2. conserver le chemin complet **uniquement** dans `regix.log` (fichier local, dans le
     répertoire de sortie, non destiné à circuler) et le documenter comme tel — ou le
     rédiger aussi quand `runtime.pseudonymize` est vrai ;
  3. écrire dans le manifeste, à la place du chemin, le **digest** de A-09 : c'est
     l'information de traçabilité réellement utile, et elle ne fuit rien ;
  4. reformuler le README : préciser exactement quels artefacts sont pseudonymisés et
     lesquels ne le sont pas (les sorties DICOM ne le sont délibérément pas — cf. A-08).
- **Tests nécessaires** :
  - `test_no_source_path_reaches_the_manifest_when_pseudonymize_is_on` : run avec des
    entrées dans un répertoire nommé `SECRET_PATIENT_NAME`, assertion que la chaîne
    n'apparaît dans **aucun** des fichiers produits ;
  - `test_a_warning_carrying_a_path_is_redacted_in_the_report` : provoquer une
    `FileNotFoundError` sur un landmark, vérifier l'absence du chemin dans `report.html` ;
  - test de « canari » généralisé, à faire tourner sur tous les artefacts d'un run.

---

#### C-03 — L'API HTTP accepte des chemins arbitraires en lecture **et en écriture**, sans authentification

- **Emplacement** : `regix/api.py:42-56` (`RegisterRequest`), `:174-191` (`register`),
  `:1-19` (docstring) ; `README.md:171-174`.
- **Gravité** : **Important** *(risque assumé et documenté — mais non atténué)*
- **Le mécanisme** : `fixed`, `moving`, `output_dir`, `fixed_labelmap`, `moving_labelmap`,
  `landmarks_fixed`, `landmarks_moving` sont des chaînes de chemin libres. La seule
  validation est `Path(path).exists()` sur `fixed` et `moving`. Il n'y a **ni allowlist de
  racines, ni normalisation, ni authentification, ni limite de débit**.
- **Ce qu'un client non authentifié peut faire** :
  - **lire** n'importe quel volume lisible par le processus (`fixed: "/etc/…"` échouera au
    décodage, mais tout NIfTI/DICOM du système est accessible) ;
  - **écrire** dans n'importe quel répertoire accessible en écriture : `output_dir` reçoit
    `report.html`, `run_manifest.json`, `moving_registered.nii.gz`, `elastix/`, …
    `overwrite` est **forcé à `True`** (`api.py:112`), donc le garde-fou de non-écrasement
    est neutralisé côté API ;
  - **exfiltrer** : soumettre un job dont `fixed` pointe une étude patient et lire ensuite
    `GET /jobs/{id}` — qui renvoie `metrics` (dont `landmarks.per_landmark_mm`, des
    coordonnées patient) et `outputs` (des chemins) ;
  - **provoquer une fuite de chemins** via `error`, qui contient `f"{type(exc).__name__}: {exc}"`
    — donc des chemins de fichiers complets pour toute exception d'E/S.
- **Statut « assumé »** : le docstring du module et le README sont **explicites** :
  « no authentication … never expose the API directly on a clinical network ». C'est
  honnête, et cela réduit la gravité. Mais **documenter n'est pas atténuer** : il n'existe
  aucun moyen, même optionnel, de restreindre les chemins. Un intégrateur qui suit le
  conseil (proxy authentifié) reste exposé à la traversée de répertoires **par tout
  utilisateur authentifié**, ce que le proxy ne peut pas filtrer.
- **Aggravation dans le README** : l'exemple de démarrage est
  `uvicorn regix.api:app --port 8000`, **sans `--host`** — uvicorn écoute alors sur
  `127.0.0.1` par défaut, ce qui est correct ; mais le docstring du module, lui, montre
  `--host 127.0.0.1` explicitement. La bonne pratique est dans le docstring, pas dans le
  README que les gens copient. À harmoniser (et à rendre explicite dans les deux).
- **Correction recommandée** :
  1. ajouter une **allowlist de racines**, lue dans l'environnement
     (`REGIX_API_ALLOWED_ROOTS`, séparée par `os.pathsep`), et refuser tout chemin dont
     `Path(p).resolve()` ne descend pas d'une racine autorisée — `Path.is_relative_to` en
     3.9+. Si la variable est absente : refuser de démarrer, ou n'accepter que le répertoire
     de travail, en le journalisant ;
  2. ne **jamais** renvoyer `str(exc)` au client : journaliser l'exception complète
     côté serveur et renvoyer un identifiant d'erreur + un type
     (`{"error": "ProcessingError", "trace_id": "..."}`) ;
  3. ajouter un jeton statique optionnel (`REGIX_API_TOKEN`, comparé en temps constant avec
     `hmac.compare_digest`) — quelques lignes, et cela transforme « ne jamais exposer » en
     « exposition contrôlée » ;
  4. ne pas forcer `overwrite=True` : l'exposer dans `RegisterRequest`, défaut `False` ;
  5. borner `_JOBS` (voir F-11).
- **Tests nécessaires** :
  - `test_a_path_outside_the_allowlist_is_refused_with_400` ;
  - `test_a_symlink_escaping_the_allowlist_is_refused` (résolution avant vérification) ;
  - `test_job_error_does_not_leak_a_filesystem_path` ;
  - `test_the_api_requires_the_token_when_one_is_configured`.

---

#### C-04 — `regix batch` : le nom de cas issu du CSV construit un chemin sans validation

- **Emplacement** : `regix/cli.py:403-406` (`name = row.get("name") or f"case{index:04d}"` ;
  `case_dir = output / name`).
- **Gravité** : **Mineur** *(CLI local — mais le CSV peut être généré par un tiers)*
- **[CONFIRMÉ]** :
  ```
  Path("batch") / "../../escape"        -> batch/../../escape
  Path("batch") / "C:/Windows/Temp/x"   -> C:/Windows/Temp/x     <-- le préfixe disparaît
  ```
  Un composant absolu dans `name` **annule** entièrement `output`. Le pipeline crée ensuite
  ce répertoire et y écrit, avec `overwrite=True` forcé (`cli.py:415`).
- **Réalisme** : le CSV d'appariement est typiquement **produit par un script d'export** ou
  par un autre service, à partir de métadonnées DICOM — un `PatientID` ou une
  `SeriesDescription` contenant `/`, `\` ou `..` n'a rien d'exotique. Ce n'est pas une
  attaque, c'est un accident de données.
- **Conséquences** : écriture hors du répertoire de sortie, écrasement silencieux
  (`overwrite=True`), et un `summary.csv` dont la colonne `output` pointe ailleurs.
- **Correction recommandée** :
  ```python
  name = _safe_case_name(row.get("name") or f"case{index:04d}")
  # -> refuse les séparateurs, '..', les composants absolus et les noms réservés Windows
  #    (CON, PRN, AUX, NUL, COM1-9, LPT1-9), tronque à 100 caractères
  case_dir = (output / name).resolve()
  if not case_dir.is_relative_to(output.resolve()):
      raise typer.BadParameter(f"nom de cas invalide : {name!r}")
  ```
  Vérifier également l'unicité des noms sur l'ensemble des lignes (deux cas homonymes
  s'écrasent aujourd'hui sans un mot).
- **Tests nécessaires** :
  - `test_batch_refuses_a_case_name_that_escapes_the_output_directory` (paramétré :
    `../x`, `/abs/x`, `C:\x`, `a/b`, `CON`) ;
  - `test_batch_refuses_duplicate_case_names`.

---

#### C-05 — La racine d'UID DICOM est une constante « de test » partagée, sans moyen de la configurer

- **Emplacement** : `regix/io/writers.py:28`.
- **Gravité** : **Important**
- **Le code** :
  ```python
  REGIX_UID_ROOT = "1.2.826.0.1.3680043.10.1337"  # test root; replace with the site's own root
  ```
  Le commentaire reconnaît le problème — mais **aucun mécanisme de remplacement n'existe** :
  pas de champ de configuration, pas de variable d'environnement, pas d'option CLI. La
  seule façon de « remplacer » est d'éditer le source.
- **Pourquoi c'est un problème** : cette racine préfixe **tous** les UID générés —
  `SOPInstanceUID` et `SeriesInstanceUID` de la série dérivée, `SOPInstanceUID`,
  `SeriesInstanceUID` et `ImplementationClassUID` de la SRO. Deux sites différents utilisant
  Regix génèrent des UID sous **la même racine**, avec le même générateur pydicom. Le
  préfixe `1.2.826.0.1.3680043.10.` appartient à l'espace gratuit Medical Connections ;
  `.1337` n'est enregistré pour ce projet nulle part.
- **Conséquences** : (a) **collision d'UID** entre établissements — un PACS qui reçoit deux
  objets homonymes en rejette un ou, pire, en écrase un ; (b) non-conformité :
  la norme DICOM exige que la racine soit propre à l'organisation émettrice ; (c) impossible
  de tracer l'origine d'un objet à partir de son UID, ce qui est l'une des raisons d'être
  de la racine.
- **Correction recommandée** :
  1. ajouter `output.dicom_uid_root: str` à `OutputConfig`, avec pour défaut
     `os.environ.get("REGIX_DICOM_UID_ROOT")` puis, à défaut, la racine actuelle **assortie
     d'un `manifest.warn` explicite** (« racine d'UID de test : les objets DICOM produits ne
     doivent pas entrer dans un PACS de production ») ;
  2. valider le format (chiffres et points, ≤ 64 caractères au total avec le suffixe généré,
     pas de composant à zéro non significatif) ;
  3. ajouter `ds.Manufacturer = "Regix"`, `ds.ManufacturerModelName`,
     `ds.SoftwareVersions = __version__` aux deux écrivains DICOM — absents aujourd'hui, et
     ce sont les tags par lesquels un PACS identifie la provenance.
- **Tests nécessaires** :
  - `test_dicom_uid_root_is_configurable_and_used_by_both_writers` ;
  - `test_the_default_test_uid_root_triggers_a_warning` ;
  - `test_generated_uids_are_valid_dicom_ui` (≤ 64 car., pas de composant vide).

---

#### C-06 — L'objet de registration spatiale (SRO) ne référence qu'**une seule coupe** par série et omet des attributs requis

- **Emplacement** : `regix/io/writers.py:285-322` (`_registration_item`, `RegistrationSequence`).
- **Gravité** : **Important**
- **Défaut 1 — une seule coupe référencée.** `pipeline._export_transforms` transmet la
  **liste complète** des fichiers de chaque série (`f_series.files`, `m_series.files`), mais
  l'appel les tronque :
  ```python
  _registration_item(fixed_ds, fixed_reference_files[:1], None),
  _registration_item(moving_ds, moving_reference_files[:1], M),
  ```
  Le `ReferencedImageSequence` de chaque item ne contient donc **qu'un SOPInstanceUID**, sur
  les 300 à 800 d'une série CT.
- **Défaut 2 — code mort structurel.** Les objets `studies` et `series` sont construits
  (avec `RTReferencedSeriesSequence`, un attribut de **RT Structure Set**, hors sujet pour
  une SRO) puis **jamais rattachés** à `item`. Neuf lignes qui ne produisent rien
  (voir aussi E-08).
- **Défaut 3 — attributs manquants.** Par rapport au *Spatial Registration IOD* :
  - pas de `ReferencedSeriesSequence` (0008,1115) au niveau racine du dataset ;
  - pas de `RegistrationTypeCodeSequence` dans les items de `MatrixRegistrationSequence` ;
  - `PatientName`, `PatientID`, `StudyInstanceUID` ne sont copiés que `if tag in fixed_ds` :
    si le DICOM source est incomplet, la SRO sort **sans attributs de type 2**, donc
    non conforme ;
  - `ds.StudyInstanceUID` est **lu** ligne 292 alors qu'il n'a peut-être jamais été affecté
    → `AttributeError` non capturée si `fixed_ds` n'a pas de `StudyInstanceUID` ;
  - `ds.StudyDate`/`ds.StudyTime` sont écrasés par **la date du jour** tout en conservant le
    `StudyInstanceUID` d'origine : deux objets de la même étude déclarent des dates d'étude
    différentes, ce que certains PACS rejettent ;
  - pas de `SeriesDate`, `InstanceCreationDate`, `Manufacturer`, `SoftwareVersions`.
- **Conséquences** : le README présente cet objet comme « what treatment planning systems
  and fusion workstations consume » (`README.md:301`). Une SRO qui ne référence qu'une coupe
  et à laquelle il manque des séquences requises risque d'être **rejetée à l'import** ou,
  plus insidieusement, **acceptée et mal associée** — la matrice serait appliquée à la
  mauvaise série. C'est le livrable dont la validité compte le plus, et c'est celui dont la
  conformité est la moins établie.
- **Ce que le test existant couvre** : `test_dicom_io.py:227::test_spatial_registration_object_is_valid`
  vérifie le SOP Class, la modalité `REG` et la matrice. Il **ne vérifie ni** le nombre de
  coupes référencées, **ni** la présence des séquences requises. Le nom du test
  (« is_valid ») promet plus qu'il ne mesure.
- **Correction recommandée** :
  1. référencer **toutes** les coupes : supprimer les `[:1]` ; regrouper les instances par
     `SeriesInstanceUID` réel de chaque fichier plutôt que de supposer une série unique ;
  2. ajouter `ReferencedSeriesSequence` racine et `RegistrationTypeCodeSequence` ;
  3. supprimer `studies`/`series` (code mort) ou les remplacer par la séquence correcte ;
  4. garantir les type 2 : `ds.PatientName = getattr(fixed_ds, "PatientName", "")` etc.,
     et `ds.StudyInstanceUID = getattr(fixed_ds, "StudyInstanceUID", generate_uid(...))`
     **avant** toute lecture ;
  5. ne pas écraser `StudyDate`/`StudyTime` : les reprendre de `fixed_ds` et n'écrire que
     `ContentDate`/`ContentTime`/`InstanceCreationDate` avec l'horodatage courant ;
  6. **valider avec un outil externe** (`dciodvfy` de dicom3tools, ou `pydicom`+`dcmvalidate`)
     dans un test marqué `slow`, plutôt que de se contenter d'assertions maison.
- **Tests nécessaires** :
  - `test_sro_references_every_slice_of_both_series` ;
  - `test_sro_has_the_iod_required_sequences` ;
  - `test_sro_keeps_the_original_study_date` ;
  - `test_sro_is_type2_complete_when_the_source_lacks_patient_tags` ;
  - `test_sro_passes_dciodvfy` (`@pytest.mark.slow`, `skipif` si l'outil est absent) — c'est
    le seul test qui justifierait le mot « valid ».

---

#### C-07 — `save_as(..., enforce_file_format=...)` : les écrivains DICOM exigent pydicom ≥ 3.0 alors que `pyproject` autorise ≥ 2.4

Traité en **J-01** (contrainte de dépendance), avec lequel il se confond.

---

#### C-08 — La série DICOM dérivée conserve le `SOPClassUID` d'origine et les séquences de référence obsolètes

- **Emplacement** : `regix/io/writers.py:162-209`.
- **Gravité** : **Mineur**
- **Le mécanisme** : `ds = template.copy()` conserve tous les tags de la coupe modèle. Sont
  notamment conservés et **désormais faux** :
  - `SourceImageSequence` / `ReferencedImageSequence` du template, qui pointent vers les
    images d'origine avec une géométrie qui n'est plus la bonne ;
  - `SOPClassUID` inchangé (par ex. *CT Image Storage*) alors que l'objet est un
    `DERIVED\SECONDARY\REGISTERED` rééchantillonné sur une autre grille. C'est défendable
    (beaucoup d'outils font ainsi) mais mérite un choix explicite plutôt qu'un héritage ;
  - `AcquisitionDate`, `AcquisitionTime`, `AcquisitionNumber`, `ExposureTime`, `KVP`,
    `ConvolutionKernel`… : des paramètres d'acquisition qui ne décrivent plus l'image
    produite ;
  - les tags privés constructeur, recopiés en bloc, dont certains contiennent des
    géométries ou des tables de correspondance obsolètes.
- **Conséquences** : image dérivée cohérente à l'affichage mais dont les métadonnées
  mentent sur son origine ; risque de mauvaise interprétation par un outil qui lit
  `SourceImageSequence`.
- **Correction recommandée** : partir d'une liste **blanche** de tags à reprendre du template
  (identité patient, étude, modalité, sémantique d'intensité) plutôt que d'une copie
  intégrale, et supprimer explicitement `SourceImageSequence`,
  `ReferencedImageSequence`, `AcquisitionNumber` et les tags privés
  (`ds.remove_private_tags()`). Renseigner `DerivationCodeSequence` en plus de
  `DerivationDescription`, déjà présent.
- **Tests nécessaires** :
  - `test_derived_series_drops_stale_reference_sequences` ;
  - `test_derived_series_has_no_private_tags`.

---

### D. Architecture et organisation

---

#### D-01 — `pipeline.py` : 1 059 lignes, une méthode de 293 lignes, 14 étapes dans un seul flux

- **Emplacement** : `regix/pipeline.py:197-489` (`_run_inner`) ; le module entier.
- **Gravité** : **Important**
- **Constat** : `_run_inner` orchestre 14 étapes numérotées en commentaire, manipule
  ~25 variables locales vivantes simultanément (`fixed`, `moving`, `fixed_seg`, `moving_seg`,
  `qc_mask_fixed`, `qc_mask_moving`, `fixed_work`, `moving_work`, `work_mask_fixed`,
  `work_mask_moving`, `fixed_channels`, `moving_channels`, `candidate`, `initial_file`,
  `outcome`, `applied`, `registered`, `metrics`, `qc_result`, `figures`, `effective`,
  `overlap`, `feature_info`, `deformable_info`, `init_report`), et passe jusqu'à
  **14 paramètres positionnels/nommés** à `_quality_control`.
- **Pourquoi c'est un problème concret** (au-delà de la longueur) :
  - **c'est la cause racine de B-04.** `ParamContext` doit être construit à un endroit puis
    reconstruit ailleurs parce qu'aucun objet ne porte l'état du run ;
  - la distinction entre les **quatre** variantes de chaque volume (`fixed`, `fixed_work`,
    `fixed.image` recadré, canaux) et les **trois** masques (`qc_mask_fixed`,
    `work_mask_fixed`, `warped_labelmap`) n'est portée que par des conventions de nommage.
    C'est précisément là que se logent A-04 (deux masques corporels divergents) et le
    problème documenté des 26 365 mL / 19 114 mL ;
  - impossible de tester une étape isolément : la suite `test_pipeline.py` ne peut faire
    que du bout-en-bout, ce qui explique qu'elle soit lente et qu'elle n'ait pas attrapé
    B-04 ni B-08.
- **Correction recommandée** — refactoring progressif, pas une réécriture :
  1. introduire une dataclass `RunState` portant les volumes, masques, canaux, contexte et
     manifeste. Elle supprime les listes de 14 paramètres et rend impossible la classe de
     bug « champ oublié » ;
  2. extraire chaque étape en fonction libre `def step_xxx(state: RunState) -> None`,
     testable isolément — l'ordre reste explicite dans `_run_inner`, qui devient une liste
     d'appels lisible ;
  3. commencer par les deux étapes les plus problématiques : **le calcul des masques**
     (unifier QC et critère, cf. A-04) et **la construction du `ParamContext`** (cf. B-04).
- **Tests nécessaires** : le refactoring est couvert par la suite bout-en-bout existante
  (c'est son intérêt principal ici). Ajouter ensuite des tests unitaires par étape, à
  commencer par `test_step_masks_produces_one_consistent_mask_pair`.

---

#### D-02 — La notion de « masque » recouvre quatre objets différents sans type distinct

- **Emplacement** : transversal — `regix/organs/roi.py:36-65` (`combined_mask`),
  `regix/pipeline.py:240-245` vs `:290-291`, `regix/qc/metrics.py`, `regix/qc/report.py`.
- **Gravité** : **Important**
- **Les quatre rôles**, tous portés par le même type `sitk.Image` :
  | Rôle | Grille | Dilaté | Produit par | Consommé par |
  |---|---|---|---|---|
  | masque de critère | travail | oui (`mask_dilate_mm`) | `_work_mask` | elastix |
  | masque de QC | fixe d'origine | oui | `combined_mask` (étape 3) | NCC/NMI, Jacobien, figures |
  | masque d'initialisation | travail | **non** (documenté « not dilated ») | *nulle part* | — |
  | boîte de ROI | travail | non | `plan_roi` | recadrage |
- **Deux problèmes concrets** :
  1. le docstring de `roi.py:1-12` distingue explicitement le « masque d'initialisation
     (non dilaté) » du masque de critère — mais **ce troisième objet n'existe pas dans le
     code**. `choose_initialization` reçoit `work_mask_fixed`/`work_mask_moving`, c'est-à-dire
     les masques de critère **dilatés**. Le centre de masse `moments_init` est donc calculé
     sur un masque élargi de 8 à 12 mm, ce qui le biaise vers l'extérieur de l'organe.
     **Documentation qui décrit une intention non implémentée** ;
  2. rien n'empêche de passer un masque de la mauvaise grille : la seule protection est
     `_paired_arrays` (`metrics.py:43-44`) qui lève, et `jacobian_statistics`
     (`metrics.py:257-259`) qui **retombe silencieusement** sur le volume entier avec un
     simple `log.debug`.
- **Correction recommandée** : introduire un type léger portant le rôle et la grille de
  référence —
  ```python
  @dataclass(frozen=True)
  class Mask:
      image: sitk.Image
      role: Literal["criterion", "qc", "initialization", "roi"]
      dilated_mm: float
  ```
  et faire porter aux signatures `Mask` plutôt que `sitk.Image | None`. Implémenter
  réellement le masque d'initialisation non dilaté, ou retirer la mention du docstring.
- **Tests nécessaires** :
  - `test_initialization_uses_an_undilated_mask` ;
  - `test_a_mask_on_the_wrong_grid_is_refused_not_ignored` (paramétré sur
    `jacobian_statistics`, `displacement_statistics`, `_centre_index`).

---

#### D-03 — L'invariant « intensités natives » est appliqué au préprocessing mais contourné par les fenêtres HU des presets

- **Emplacement** : `regix/preprocess/intensity.py:1-43` (docstring) ;
  `regix/presets/*.yaml` (5 presets déclarant `window:`).
- **Gravité** : **Mineur** *(tension de conception, pas un bug)*
- **La tension** : le module affirme, avec raison et démonstration à l'appui,
  « what Regix hands to elastix stays on its native intensity scale. **Clipping is allowed
  — it bounds values without moving them, so a Hounsfield unit remains a Hounsfield unit.** »
  C'est exact au sens strict. Mais les conséquences du *clipping* sont sous-estimées :
  - il change le minimum du volume, ce qui **bascule la branche de `body_mask`** (A-04) ;
  - `ct_liver` = (−30, 180) écrase **tout l'air à −30 HU** : il ne reste plus de contraste
    air/tissu pour l'optimiseur aux résolutions grossières de la pyramide, là où
    précisément le recalage global se joue ;
  - il rend inopérante la détection de quantification (`intensity_range` mesuré *après*
    clipping vaudrait 210 pour `ct_liver`, donc `< 50` est faux, donc pas de warning — sans
    compter que la détection est morte de toute façon, B-04).
- **Ce n'est pas une contradiction, c'est un angle mort** : la doctrine « préserver l'échelle »
  est bien appliquée à la *normalisation* ; le *clipping*, qui a ses propres effets de bord
  documentés nulle part, échappe à la même rigueur d'analyse.
- **Correction recommandée** : ajouter au docstring un paragraphe sur les effets de bord du
  clipping (branche `body_mask`, perte de contraste air/tissu aux résolutions grossières,
  interaction avec `intensity_range`), et **reconsidérer** les fenêtres serrées des presets
  `ct_ct_liver_followup` (`ct_liver`) et `ct_cbct_igrt` (`ct_bone`) : elles servent le
  contraste local mais nuisent au recalage global. Mesurer avant de trancher.
- **Tests nécessaires** :
  - `test_a_ct_window_does_not_change_the_body_mask` (recoupe A-04) ;
  - test de non-régression comparant la TRE d'un recalage fantôme avec et sans fenêtrage.

---

#### D-04 — `displacement_field_from_transform` vit dans le module GPU

- **Emplacement** : `regix/registration/convexadam.py:186-194` ;
  importé par `regix/registration/warp.py:21`.
- **Gravité** : **Cosmétique**
- **Constat** : la fonction n'utilise que `sitk.TransformToDisplacementFieldFilter` —
  aucune dépendance à torch, aucun lien avec ConvexAdam. Elle est appelée par
  `SitkAppliedTransform.displacement_field` et `ElastixAppliedTransform.displacement_field`,
  c'est-à-dire sur le chemin nominal CPU. Sa présence dans `convexadam.py` force
  `warp.py` (donc `pipeline.py`) à importer le module GPU, ce qui contredit l'intention
  affichée par `registration/__init__.py` (cf. A-10).
- **Correction recommandée** : déplacer vers `preprocess/geometry.py`, à côté de
  `resample_like` ; conserver un alias dans `convexadam` le temps d'une version.
- **Tests nécessaires** : `test_importing_the_pipeline_does_not_import_convexadam`
  (dans un sous-processus, `assert "regix.registration.convexadam" not in sys.modules`).

---

#### D-05 — Abstraction manquante : aucun chargeur de transformée unifié

- **Emplacement** : `regix/cli.py:477-480`, `regix/registration/initialize.py:287`,
  `regix/registration/transforms.py:30-73`, `:358-359`.
- **Gravité** : **Important** *(c'est la cause commune de B-03 et A-14)*
- **Constat** : Regix manipule **deux** formats de transformée qui partagent l'extension
  `.txt` — le format elastix `(Key "value")` et l'*Insight Transform File* ITK. Il écrit les
  deux dans le même run. Trois points d'entrée décident indépendamment lequel lire, chacun
  par une heuristique différente et chacun avec un défaut :

  | Point d'entrée | Décision | Défaut |
  |---|---|---|
  | `cli.apply` | extension `== ".tfm"` | B-03 : plante sur le `.txt` de Regix |
  | `init.mode=file` | `sitk.ReadTransform` inconditionnel | A-14 : échoue puis se replie en silence |
  | `engine` (interne) | toujours elastix | correct (contexte connu) |

- **Correction recommandée** : une seule fonction
  `transforms.load_any_transform(path) -> sitk.Transform`, reniflant le contenu, levant une
  erreur explicite listant les deux formats acceptés, et utilisée par les trois appelants.
  Les deux briques existent déjà (`read_parameter_file` + `parameter_map_to_transform`, et
  `sitk.ReadTransform`) : c'est une dizaine de lignes qui supprime deux bugs.
- **Tests nécessaires** : test paramétré sur les cinq formats qu'un run produit
  (`.tfm`, Insight `.txt`, `TransformParameters.0.txt`, `.h5`, `initial_transform.txt`),
  vérifiant que tous se chargent et que la transformée obtenue est identique quand elle
  décrit la même chose.

---

#### D-06 — Abstraction manquante : le nommage des sorties est dispersé en littéraux

- **Emplacement** : `regix/pipeline.py` (≈ 20 littéraux), `regix/cli.py`,
  `.github/workflows/ci.yml:128-130`, `tests/test_cli.py`, `README.md`.
- **Gravité** : **Mineur**
- **Constat** : `"report.html"`, `"run_manifest.json"`, `"config_effective.yaml"`,
  `"moving_registered.nii.gz"`, `"transform/final_transform.txt"`, `"masks/{side}_labelmap.nii.gz"`,
  `f"stage{index:02d}_{stage.name}_TransformParameters.txt"`… sont écrits en clair à
  chaque usage, et répliqués dans la CI, les tests et le README.
- **Conséquences** : renommer une sortie exige de retrouver 5 endroits ; c'est aussi ce qui
  rend B-09 (nettoyage sur `--overwrite`) délicat à écrire correctement, faute d'inventaire.
- **Correction recommandée** : un module `regix/layout.py` déclarant les chemins relatifs et
  une fonction `expected_outputs(cfg) -> dict[str, Path]`. Il sert directement à trois
  choses : le nettoyage de B-09, l'inventaire du manifeste, et un test de contrat unique.
- **Tests nécessaires** : `test_a_run_produces_exactly_the_declared_layout` — comparer
  l'arborescence réelle après un run à `expected_outputs(cfg)`, dans les deux sens
  (aucun fichier manquant, aucun fichier inattendu).

---

#### D-07 — `RegistrationPipeline` porte de l'état d'instance qui empêche sa réutilisation propre

- **Emplacement** : `regix/pipeline.py:149-151` ; `regix/pipeline.py:494-509` (`_load`).
- **Gravité** : **Mineur**
- **Constats** :
  1. `self.targets = resolve_targets(config.organs.targets)` est calculé **au constructeur**.
     `run()` peut ensuite être appelé plusieurs fois ; les cibles restent celles de la
     configuration initiale — cohérent, mais implicite ;
  2. `_load` **mute l'objet de l'appelant** quand on lui passe un `Volume` :
     ```python
     if isinstance(source, Volume):
         volume = source
         if modality: volume.modality = modality.upper()   # mutation d'un objet du client
     ```
     Un utilisateur de l'API Python qui réutilise le même `Volume` pour deux runs de
     modalités différentes verra son objet modifié sous lui ;
  3. `setup_logging` est **global** : appelé dans `run()`, il reconfigure le logger `regix`
     pour tout le processus. Deux pipelines concurrents (l'API HTTP avec
     `max_workers > 1`) écriraient dans le même fichier — c'est d'ailleurs pourquoi
     `_POOL` est limité à 1 worker, mais rien ne le documente à cet endroit.
- **Correction recommandée** : `_load` doit faire `volume = replace(source, modality=...)`
  (`Volume` est une dataclass, `dataclasses.replace` existe déjà et est utilisé par
  `with_image`). Documenter dans `RegistrationPipeline` que `run()` reconfigure le logging
  global et n'est pas sûr en concurrence — ou basculer sur un logger par run.
- **Tests nécessaires** :
  - `test_running_the_pipeline_does_not_mutate_the_input_volume` ;
  - `test_the_configuration_object_is_not_mutated_by_a_run` **existe déjà**
    (`test_units.py:741`) — bon réflexe, à étendre aux `Volume`.

---

#### D-08 — Le rapport HTML est assemblé par concaténation de chaînes, sans moteur de template — alors que `pyproject` déclare un dossier de templates

- **Emplacement** : `regix/qc/report.py:364-544` ; `pyproject.toml:75`.
- **Gravité** : **Mineur**
- **Constats** :
  1. `pyproject.toml` déclare `package-data = { regix = ["presets/*.yaml", "qc/templates/*.html"] }`
     — **le répertoire `regix/qc/templates/` n'existe pas**. Déclaration morte, vestige
     d'une approche abandonnée (cf. E-12) ;
  2. `build_html_report` construit le document par `sections.append(f"<h2>…</h2>" + _table(…))`,
     puis `_HTML_TEMPLATE.format(...)`. C'est fonctionnel et sans dépendance — un choix
     défendable pour un fichier autoportant — mais l'échappement repose sur la discipline
     de chaque site d'insertion.
- **Vérification de l'échappement** : **[LECTURE]**, revue exhaustive des points
  d'insertion — `_cell` et `_table` échappent (`html.escape`), `_status_badge` échappe, le
  titre / sous-titre / horodatage / disclaimer échappent, les warnings échappent, les
  libellés d'initialisation échappent. Les `figures[key]` sont insérés **non échappés** dans
  `src="…"` mais ce sont des data-URI base64 générés en interne. **Aucune injection HTML
  trouvée** — le code est correct, mais la propriété n'est garantie par aucun test.
- **Correction recommandée** : soit supprimer la ligne `qc/templates/*.html` du
  `pyproject.toml`, soit créer le répertoire ; et ajouter un test d'échappement.
- **Tests nécessaires** :
  - `test_report_escapes_hostile_content` : injecter `<script>alert(1)</script>` dans une
    description de série, un nom d'organe et un warning, vérifier l'absence de `<script>`
    dans la sortie ;
  - `test_declared_package_data_directories_exist`.

---

### E. Duplications, code mort et dépendances inutilisées

Inventaire établi par recherche exhaustive sur `regix/` (les usages exclusivement en test
sont signalés — un symbole utilisé seulement par ses propres tests reste du code mort du
point de vue du produit).

---

#### E-01 — `_same_grid` est réimplémenté à l'identique quatre fois

- **Emplacement** : `regix/preprocess/geometry.py:251-257`, `regix/organs/roi.py:222-228`,
  `regix/organs/segmenter.py:380-386` — **trois copies strictement identiques** (même corps,
  même tolérance 1e-4, même signature) — plus `Volume.same_grid_as`
  (`regix/io/volume.py:66-72`), qui fait la même chose au niveau `Volume` et **n'est jamais
  appelée**.
- **Gravité** : **Mineur**
- **Conséquences** : une correction de tolérance ou l'ajout d'une comparaison (le type de
  pixel, par exemple) doit être faite trois fois ; le risque de divergence est réel et
  silencieux, puisque ces fonctions arbitrent des rééchantillonnages.
- **Correction recommandée** : une seule implémentation dans `preprocess/geometry.py`,
  importée par les deux autres modules. Faire de `Volume.same_grid_as` un mince
  `return same_grid(self.image, other.image, tol)` — ou la supprimer.
- **Tests nécessaires** : `test_same_grid_detects_each_kind_of_mismatch` (taille, spacing,
  origine, direction, à la tolérance près), une seule fois.

---

#### E-02 — `_background_value` / `_background_of` : deux implémentations identiques dans deux modules

- **Emplacement** : `regix/preprocess/geometry.py:118-125` (`_background_value`) et
  `regix/pipeline.py:1033-1036` (`_background_of`), plus `regix/pipeline.py:1039-1043`
  (`_intensity_range`) qui recalcule la même chose avec le même filtre.
- **Gravité** : **Cosmétique**
- **Conséquence secondaire, non triviale** : chacune reconstruit
  `sitk.Cast(image, sitk.sitkFloat32)` — soit une **copie complète du volume** — puis exécute
  `MinimumMaximumImageFilter`. Sur le chemin nominal, `_background_of` est appelé deux fois
  (restitution et `moving_before`) et `_intensity_range` une fois, sur des volumes pleine
  résolution : trois copies float32 inutiles. Voir G-02.
- **Correction recommandée** : une fonction unique
  `geometry.intensity_range(image) -> tuple[float, float]` mise en cache par identité
  d'image, dont `background_value` renvoie le premier élément.
- **Tests nécessaires** : `test_intensity_range_matches_numpy_on_a_phantom`.

---

#### E-03 — Code mort confirmé : symboles définis et jamais utilisés en production

**[CONFIRMÉ]** par recherche sur l'ensemble de `regix/` :

| Symbole | Emplacement | Statut | Gravité |
|---|---|---|---|
| `file_digest` | `logging_utils.py:81` | jamais appelé — et le docstring du module promet des « input hashes » (A-09) | Mineur |
| `_IMAGE_SUFFIXES` | `io/volume.py:21` | constante jamais lue ; `load_volume` s'en remet à `sitk.ReadImage` | Cosmétique |
| `Volume.same_grid_as` | `io/volume.py:66` | jamais appelée (3 copies locales à la place, E-01) | Cosmétique |
| `save_landmarks` | `io/writers.py:79` | jamais appelée, jamais testée ; `load_landmarks` l'est | Cosmétique |
| `load_transform` | `registration/transforms.py:358` | jamais appelée | Cosmétique |
| `transform_points` (module) | `registration/transforms.py:346` | jamais appelée ; le pipeline utilise les méthodes d'`AppliedTransform` | Cosmétique |
| `erode_mask_mm` | `preprocess/geometry.py:207` | jamais appelée ; `StageConfig.erode_mask` ne fait qu'écrire `(ErodeMask …)` pour elastix | Cosmétique |
| `warp_landmarks_moving_to_fixed` | `registration/warp.py:171` | appelée **uniquement par ses tests** (`test_registration_internals.py:286, 301`) | Mineur |
| `OrganProfile.recommended_stage_types` | `organs/labels.py:83` | jamais appelée | Cosmétique |
| `OrganProfile.hu_window / typical_motion_mm / mask_dilate_mm / roi_margin_mm / region / notes` | `organs/labels.py` | propagés par `merged_profile`, jamais consommés (A-06) | Important |
| `OrganROI.fixed_region / moving_region` | `organs/roi.py:108-109` | renseignés par `plan_roi`, jamais relus | Mineur |
| `ElastixEngine.keep_intermediate` | `registration/engine.py:124,127` | stocké, **jamais utilisé** (E-11) | Mineur |
| `FeatureConfig.cache_dir` | `config.py:178` | jamais lu (les `cache_dir` de `segmenter.py` sont un autre champ) | Mineur |
| `OrganSegmentation.mask_for(missing="raise")` | `organs/segmenter.py:73,83-84` | la branche `raise` n'est jamais empruntée | Cosmétique |
| `studies` / `series` | `io/writers.py:290-303` | construits puis jamais rattachés (C-06) | Mineur |
| `merged["stages"] = raw["stages"]` | `config.py:538-539` | redondant : `_deep_update` a déjà remplacé la liste | Cosmétique |
| `if overrides else cfg` | `cli.py:326` | branche morte : `overrides` contient toujours `runtime.log_level` (posé ligne 324) | Cosmétique |
| `device=... if ... != "auto" else "auto"` | `pipeline.py:690` | tautologie : les deux branches donnent la même valeur | Cosmétique |
| 11 alias identité | `organs/labels.py:36-52` | `"brain": "brain"`, `"sacrum": "sacrum"`, `"femur_left": "femur_left"`, `"hip_left"`, `"hip_right"`, `"heart"`, `"adrenal_gland_left"`, `"adrenal_gland_right"`, `"inferior_vena_cava"`, `"portal_vein_and_splenic_vein"`, `"vertebrae_l1"` — `ORGAN_ALIASES.get(key, key)` les rend inopérants | Cosmétique |

- **Gravité globale** : **Mineur**
- **Pourquoi ce n'est pas anodin** : trois de ces entrées ne sont pas du simple bruit —
  `file_digest`, les champs d'`OrganProfile` et `keep_intermediate` **soutiennent une
  affirmation documentaire**. Leur inutilisation est ce qui rend ces affirmations fausses.
  Le reste est du bruit à supprimer.
- **Correction recommandée** : supprimer les entrées « Cosmétique » ; **câbler** (et non
  supprimer) `file_digest`, les champs d'`OrganProfile` et `keep_intermediate`, dont
  l'absence est un défaut fonctionnel documenté ailleurs (A-06, A-09, E-11).
  Pour `warp_landmarks_moving_to_fixed` : soit l'exposer au CLI (une commande
  `regix apply --invert` a une valeur clinique claire), soit la retirer avec ses tests.
- **Tests nécessaires** : ajouter `vulture` ou `ruff --select F401,F841` étendu (les règles
  actuelles `E,F,W,I,B,UP` ne détectent pas les fonctions publiques inutilisées) à la CI,
  avec une allowlist explicite pour les API publiques volontaires.

---

#### E-04 — Dépendances déclarées mais non utilisées, et dépendance utilisée mais non déclarée

- **Emplacement** : `pyproject.toml:28-61`.
- **Gravité** : **Mineur**
- **Analyse** de chaque dépendance du cœur :

  | Déclarée | Réellement utilisée | Verdict |
  |---|---|---|
  | `itk-elastix>=0.20` | `itk_bridge`, `engine`, `params` | ✔ |
  | `SimpleITK>=2.2` | partout | ✔ |
  | `numpy>=1.24` | partout | ✔ |
  | `pydantic>=2.5` | `config`, `api` | ✔ |
  | `PyYAML>=6.0` | `config`, `cli` | ✔ |
  | `typer>=0.12` | `cli` | ✔ |
  | `rich>=13.0` | `cli` (`Console`, `Table`) | ✔ |
  | `matplotlib>=3.7` | `qc/report` | ✔ — mais **uniquement** pour le rapport HTML |
  | `pydicom>=2.4` | `io/writers`, `io/dicom` (indirect), `pipeline` | ✔ *mais contrainte fausse* (J-01) |

- **Constats** :
  1. **`matplotlib` est une dépendance du cœur pour une fonctionnalité optionnelle.**
     `qc.report_html` peut valoir `false`, et `matplotlib` (~40 Mo avec ses dépendances)
     n'est alors jamais importé — les imports sont d'ailleurs déjà locaux aux fonctions de
     figure, ce qui est bien fait. Il devrait être un extra `report`, avec un message clair
     si `report_html` est demandé sans lui ;
  2. **`anatomix` est déclaré comme une dépendance git directe** (`pyproject.toml:50`).
     Une URL VCS dans `[project.optional-dependencies]` **empêche la publication du paquet
     sur PyPI** (PEP 440 : les dépendances directes sont interdites pour les distributions
     publiées). Le projet est déclaré « Development Status :: 4 - Beta » avec des URL
     Homepage/Repository/Issues : il vise manifestement la publication. À déplacer vers une
     instruction d'installation documentée ;
  3. `organs = ["regix[totalsegmentator]"]` — auto-référence circulaire, valide en PEP 621
     et correctement commentée, mais elle exige que le paquet soit installable par son nom :
     inopérante en `pip install -e .` depuis un checkout sans index. **[À VÉRIFIER]** : la CI
     n'installe jamais cet extra, donc le point n'est pas couvert ;
  4. `all = ["regix[features,organs,api,dev]"]` hérite du même problème, et embarque
     `dev` (pytest, ruff) dans un extra que son nom suggère destiné aux utilisateurs.
- **Correction recommandée** : extra `report = ["matplotlib>=3.7"]` ; retirer la dépendance
  git d'anatomix au profit d'une ligne d'installation dans le README ; retirer `dev` de
  `all` ou renommer `all` en `everything-including-dev` ; ajouter un job CI qui installe
  chaque extra séparément (J-05).
- **Tests nécessaires** :
  - `test_the_core_install_has_no_optional_import_at_module_level` : dans un
    sous-processus, `import regix.pipeline` puis vérifier que `matplotlib`, `torch`,
    `monai`, `totalsegmentator` ne sont pas dans `sys.modules` ;
  - job CI `pip install -e ".[features]"` / `".[api]"` / `".[totalsegmentator]"`.

---

#### E-05 — `hausdorff95` et `mean_surface_distance` recalculent deux fois les quatre mêmes filtres

- **Emplacement** : `regix/qc/metrics.py:133-168` et `:194-197`.
- **Gravité** : **Mineur** *(performance — voir aussi G-04)*
- **Le mécanisme** : `_surface_distances` exécute **quatre** filtres coûteux
  (2 × `LabelContour`, 2 × `SignedMaurerDistanceMap`). `organ_overlap_report` appelle
  `hausdorff95(m_fixed, m_warp)` **puis** `mean_surface_distance(m_fixed, m_warp)` :
  les quatre filtres tournent **deux fois par organe**, sur des volumes pleine résolution.
  Avec `qc_labels: [liver, spleen, kidney_right]`, c'est 24 exécutions au lieu de 12.
- **Correction recommandée** : appeler `_surface_distances` une fois dans
  `organ_overlap_report` et dériver les deux métriques ; conserver `hausdorff95` et
  `mean_surface_distance` comme fonctions publiques minces pour l'usage isolé.
- **Tests nécessaires** : `test_surface_distances_are_computed_once_per_organ`
  (compteur d'appels par monkeypatch) ; les valeurs sont déjà couvertes par
  `test_units.py:965`.

---

#### E-06 — Trois définitions concurrentes de la notion de « grille identique » dans les métriques

- **Emplacement** : `regix/qc/metrics.py:34-35` (lève), `:43-44` (lève),
  `:257-259` (retombe en silence), `:283-286` (retombe en silence, sans même un log),
  `regix/qc/report.py:206-208` (retourne `None`), `:318-319` (retombe sur le centre).
- **Gravité** : **Mineur**
- **Constat** : la même incohérence géométrique produit selon l'appelant une exception, un
  `log.debug`, un `None`, ou un repli muet. Aucun de ces comportements n'est faux
  isolément ; leur coexistence rend imprévisible le diagnostic d'un masque mal aligné.
  Le cas le plus discutable est `displacement_statistics` (`metrics.py:283-286`), qui
  ignore un masque incompatible **sans aucune trace** — les statistiques de déplacement
  rapportées portent alors sur le volume entier, air compris, ce qui les rend
  optimistes et fausses.
- **Correction recommandée** : une règle unique — toute incompatibilité de grille dans le
  QC produit un `manifest.warn` **et** une valeur marquée `available: false`, jamais un
  repli silencieux sur un domaine différent.
- **Tests nécessaires** : `test_qc_metrics_report_unavailable_rather_than_silently_widening`
  (paramétré sur les 4 fonctions).

---

#### E-07 — `_read_parameter_file` : indirection d'un seul appel

- **Emplacement** : `regix/registration/engine.py:394-398`.
- **Gravité** : **Cosmétique**
- **Constat** : fonction privée dont le corps est un import local suivi d'un appel à
  `params.read_parameter_file`, utilisée une seule fois (`engine.py:209`). L'import local
  n'évite aucun cycle : `engine.py:49-56` importe déjà six symboles de `params` au niveau
  module.
- **Correction recommandée** : ajouter `read_parameter_file` à l'import de tête et
  supprimer l'indirection.

---

#### E-08 — Code mort structurel dans l'écriture de la SRO

Voir **C-06, défaut 2** : `studies` et `series` (`io/writers.py:290-303`) sont construits,
peuplés d'attributs de RT Structure Set, puis jamais rattachés au dataset.
Gravité **Mineur** ; la correction est incluse dans C-06.

---

#### E-09 — Le rapport HTML sérialise un `set`, donc dans un ordre non déterministe

- **Emplacement** : `regix/pipeline.py:466-468`.
- **Gravité** : **Mineur**
- **Le code** :
  ```python
  "metrics": " / ".join({s.get("metric", "?") for s in [x.to_dict() for x in outcome.stages]}),
  ```
  L'ensemble en compréhension est un `set` : l'ordre d'itération des chaînes dépend de
  `PYTHONHASHSEED`, randomisé à chaque processus.
- **Conséquences** : deux runs strictement identiques produisent des `report.html` qui
  diffèrent (`"ncc / mi"` vs `"mi / ncc"`). Cela casse toute comparaison de rapports par
  diff ou par hachage, et contredit l'insistance du projet sur la reproductibilité.
  Accessoirement, la déduplication perd l'ordre des stages, qui est l'information utile.
- **Correction recommandée** : `" -> ".join(s.get("metric","?") for s in …)` — sans
  déduplication, dans l'ordre des stages, ce qui est plus informatif et déterministe.
- **Tests nécessaires** : `test_report_metric_summary_is_deterministic_and_ordered`
  (exécuter deux fois avec des `PYTHONHASHSEED` différents via `subprocess`).

---

#### E-10 — `environment_report` renseigne la clé `itk` deux fois

- **Emplacement** : `regix/logging_utils.py:104-122`.
- **Gravité** : **Cosmétique**
- **Constat** : la boucle affecte `report["itk"] = itk.__version__` (version du **paquet
  Python** `itk`), puis le bloc suivant l'écrase par `itk.Version.GetITKVersion()` (version
  de la **bibliothèque ITK**). La première valeur est perdue — et c'est la plus proche de ce
  qu'on veut (A-12). Deux informations distinctes se disputent une clé.
- **Correction recommandée** : deux clés, `itk_python` et `itk_core`, plus
  `itk_elastix` (A-12).

---

#### E-11 — `runtime.keep_intermediate` est documenté comme un réglage et ne fait rien

- **Emplacement** : `regix/config.py:408` ; `regix/pipeline.py:334` ;
  `regix/registration/engine.py:124,127`.
- **Gravité** : **Mineur**
- **Constat** : la valeur circule de la configuration jusqu'à `ElastixEngine.__init__`, où
  elle est stockée dans `self.keep_intermediate` — **et n'est jamais relue**. Les fichiers
  intermédiaires (répertoires `stageNN_*`, `elastix.log`, `IterationInfo.*.txt`) sont donc
  **toujours conservés**, quelle que soit la valeur.
- **[CONFIRMÉ]** sur le run réel : `e2e_out/elastix/stage00_rigid/` contient 4 fichiers
  `IterationInfo.0.R{0..3}.txt` (~100 Ko) et un `elastix.log` de **117 Ko**, alors que le
  défaut est `keep_intermediate: false`.
- **Conséquences** : (a) un réglage documenté sans effet ; (b) volumétrie sur disque non
  maîtrisée en traitement par lots — ~250 Ko de journaux elastix par cas, soit 250 Mo pour
  1 000 cas, en plus des volumes ; (c) les `elastix.log` contiennent les chemins de travail
  (C-02).
- **Correction recommandée** : implémenter le nettoyage en fin de `ElastixEngine.run`
  lorsque `keep_intermediate` est faux — en **conservant impérativement**
  `TransformParameters.0.txt` et `parameters.txt` (ce sont les livrables copiés par
  `_export_transforms`) et en ne supprimant que `IterationInfo.*` et, éventuellement, le
  corps de `elastix.log` au-delà de sa fin. Attention à ne pas casser A-03 (rejouabilité) :
  documenter que `keep_intermediate: true` est requis pour rejouer.
- **Tests nécessaires** :
  - `test_keep_intermediate_false_removes_iteration_logs_but_keeps_the_transform` ;
  - `test_keep_intermediate_true_keeps_everything`.

---

#### E-12 — `package-data` déclare un répertoire de templates inexistant

- **Emplacement** : `pyproject.toml:75`.
- **Gravité** : **Cosmétique**
- **Constat** : `regix = ["presets/*.yaml", "qc/templates/*.html"]` — `regix/qc/templates/`
  n'existe pas. Le motif est simplement ignoré par setuptools : aucun effet, mais il
  suggère au lecteur qu'un système de templates existe (cf. D-08).
- **Correction recommandée** : supprimer le motif, ou créer le répertoire si le passage à
  un template externe est prévu.
- **Tests nécessaires** : `test_declared_package_data_directories_exist` (couvre aussi D-08).

---

### F. Gestion des erreurs et cas limites

---

#### F-01 — `--set` : trois classes d'erreur utilisateur produisent un traceback non intercepté

- **Emplacement** : `regix/cli.py:50-73` (`_apply_sets`).
- **Gravité** : **Important**
- **Le code** :
  ```python
  for part in parts[:-1]:
      node = node[int(part)] if part.isdigit() else node.setdefault(part, {})
  ```
  Seule l'absence de `=` est traitée proprement (`typer.BadParameter`).
- **[CONFIRMÉ]** — quatre expressions passées à `regix register … --dry-run` :
  ```
  --set stages.9.max_iterations=10   -> exit 1, IndexError        (non intercepté)
  --set stages.0.type.x=1            -> exit 1, TypeError         (non intercepté)
  --set nope.deep=1                  -> exit 1, ValidationError   (message pydantic brut)
  --set qc.gates.min_ncc_gain=null   -> exit 0                    (fonctionne)
  ```
- **Détail des trois échecs** :
  1. `IndexError` — indice de stage hors bornes. Cas très facile à atteindre : le README
     lui-même donne `--set stages.2.final_grid_spacing_mm=12`, qui échoue sur tout preset
     à moins de 3 stages (5 presets sur 8) ;
  2. `TypeError` — descente dans une valeur scalaire ou une énumération
     (`data["stages"][0]["type"]` est un `TransformType`, qui n'a pas `.setdefault`) ;
  3. `ValidationError` pydantic brute — une clé inconnue crée un sous-dictionnaire vide
     (`setdefault(part, {})`) puis échoue à la validation avec un message qui parle de
     `extra_forbidden`, sans jamais mentionner `--set` ni suggérer l'orthographe correcte.
  À noter aussi : un indice **négatif** (`stages.-1.max_iterations`) n'est pas reconnu par
  `part.isdigit()` et sera traité comme une clé de dictionnaire, donc silencieusement
  transformé en clé `-1` puis rejeté par la validation.
- **Conséquences** : la fonctionnalité mise en avant dans le README (« Any configuration
  option is overridable without editing YAML ») échoue par traceback sur les erreurs les
  plus banales. Pour un outil destiné à des physiciens médicaux et non à des développeurs
  Python, c'est un défaut d'utilisabilité de premier ordre.
- **Correction recommandée** : encapsuler la descente et convertir toute exception en
  `typer.BadParameter` explicite, avec le contexte :
  ```python
  try:
      ...
  except (IndexError, KeyError, TypeError, AttributeError) as exc:
      raise typer.BadParameter(
          f"--set {item!r} : chemin invalide à « {part} ». "
          f"Clés disponibles à ce niveau : {sorted(node) if isinstance(node, dict) else type(node).__name__}"
      ) from exc
  ```
  Ajouter un contrôle de borne explicite pour les indices, et gérer les indices négatifs
  (ou les refuser explicitement). Enfin, convertir la `ValidationError` finale en
  `BadParameter` en listant les clés proches (`difflib.get_close_matches`).
- **Tests nécessaires** :
  - `test_set_reports_an_out_of_range_stage_index_cleanly` ;
  - `test_set_reports_an_unknown_key_with_a_suggestion` ;
  - `test_set_reports_a_descent_into_a_scalar_cleanly` ;
  - `test_set_never_raises_anything_but_badparameter` (test « fuzz » léger sur une liste
    d'expressions malformées).
  Le test existant `test_invalid_set_is_rejected` (`test_cli.py:206`) ne couvre que le cas
  « pas de `=` ».

---

#### F-02 — Les options d'énumération du CLI sont des chaînes libres : erreur de frappe = `ValidationError` pydantic brute

- **Emplacement** : `regix/cli.py:258-260` (`--organ-backend`), `:270-272` (`--init`),
  `:278` (`--log-level`), `:379` (`--log-level` de `batch`).
- **Gravité** : **Important**
- **[CONFIRMÉ]** :
  ```
  --log-level verbose      -> exit 1, ValidationError
  --init nonsense          -> exit 1, ValidationError
  --organ-backend nope     -> exit 1, ValidationError
  ```
- **Le problème** : ces trois options correspondent à des `Enum` / `Literal` déjà définis
  dans `config.py` (`OrganBackend`, `InitMode`, `RuntimeConfig.log_level`). Typer sait
  générer des choix à partir d'un `Enum` — `--help` afficherait alors les valeurs
  acceptées, la complétion fonctionnerait, et une faute de frappe donnerait
  `Invalid value for '--init': 'nonsense' is not one of 'identity', 'geometry', …`.
  En l'état, `--help` n'affiche **aucune** valeur pour `--log-level`, et un texte
  d'aide en prose pour `--init`.
- **Conséquences** : messages d'erreur illisibles pour l'utilisateur cible, aide incomplète,
  pas de complétion. Accessoirement, `--log-level` accepte n'importe quelle casse grâce au
  `.upper()` (`cli.py:324`) mais n'accepte pas `CRITICAL`, qui est pourtant un niveau
  logging standard — silence sur ce point.
- **Correction recommandée** : typer les paramètres avec les énumérations existantes
  (`init: Optional[InitMode]`, `organ_backend: Optional[OrganBackend]`) et créer un
  `LogLevel(str, Enum)` réutilisé par `RuntimeConfig`. Attention : cela impose d'importer
  `regix.config` au niveau module de `cli.py`, ce qui va à l'encontre de l'optimisation
  d'imports paresseux — mais `config.py` n'importe que `yaml` et `pydantic`, tous deux
  légers, donc le coût est négligeable. À mesurer avec
  `python -X importtime -c "import regix.cli"` avant/après.
- **Tests nécessaires** :
  - `test_invalid_enum_option_gives_a_typer_error_listing_the_choices` (paramétré sur les
    3 options) ;
  - `test_help_lists_the_accepted_values` ;
  - `test_cli_import_time_stays_under_a_budget` (garde-fou pour l'optimisation d'imports).

---

#### F-03 — `_export_transforms` : un `shutil.copy2` sans garde peut faire échouer un run par ailleurs réussi

- **Emplacement** : `regix/pipeline.py:891-898`.
- **Gravité** : **Mineur**
- **Le code** :
  ```python
  for index, stage in enumerate(outcome.stages):
      shutil.copy2(stage.transform_parameter_file, transform_dir / f"stage{index:02d}_…")
  ```
  Aucun `try`, aucun contrôle d'existence — contrairement à la ligne suivante, qui vérifie
  `if params.exists()`. L'asymétrie suggère un oubli plutôt qu'un choix.
- **Cas atteignables** : disque plein, répertoire de travail nettoyé par un antivirus ou un
  `tmpwatch` pendant un long run par lots, verrou Windows sur un fichier ouvert par un
  autre outil, chemin trop long (limite `MAX_PATH` sur Windows, atteignable avec des noms
  de stage personnalisés via `StageConfig.label`).
- **Conséquences** : le recalage est terminé et correct, la transformée est en mémoire, mais
  l'exception remonte de l'étape `exports` → `_run_inner` → `run()` → le run est marqué
  `ERROR` et **`RegistrationResult` n'est jamais renvoyé**. Le travail est perdu pour une
  copie de fichier. `manifest.save()` est bien appelé (bon réflexe, `pipeline.py:184`), donc
  la trace subsiste, mais l'appelant Python reçoit une exception au lieu du résultat.
- **Correction recommandée** : envelopper chaque copie et dégrader en
  `manifest.warn(...)` ; plus généralement, **toute** l'étape `exports` devrait être
  non fatale — elle vient après la production du résultat. Le seul export dont l'échec
  mérite peut-être de remonter est `config_effective.yaml` (traçabilité).
- **Tests nécessaires** :
  - `test_a_failed_transform_copy_degrades_to_a_warning` (monkeypatch de `shutil.copy2`) ;
  - `test_the_result_is_returned_even_if_exports_fail`.

---

#### F-04 — `read_parameter_file` : plusieurs formes valides de fichier elastix provoquent une erreur cryptique ou une perte silencieuse

- **Emplacement** : `regix/registration/params.py:517-550` ; `_validate` `:423-472`.
- **Gravité** : **Mineur**
- **Cas non couverts** :
  1. **Entrée sans valeur** — une ligne `(WriteResultImage)` produit `pmap["WriteResultImage"] = ()`.
     Si la clé concernée est `NumberOfResolutions`, `_validate` fait
     `int(pmap["NumberOfResolutions"][0])` → **`IndexError` non interceptée**, sans mention
     du fichier ni de la ligne ;
  2. **Entrée multi-lignes** — elastix tolère qu'une liste de valeurs s'étende sur plusieurs
     lignes. Le parseur exige `line.startswith("(") and line.endswith(")")` : une entrée
     multi-lignes est **silencieusement ignorée**, et la clé disparaît de la map sans un mot.
     Sur `ImagePyramidSchedule`, cela produit exactement l'écart que `_validate` cherche à
     signaler — mais par un autre chemin, non détecté ;
  3. **`//` à l'intérieur d'une chaîne** — `line.split("//")[0]` tronque
     `(ResultImageFormat "a//b")`. Cas rare mais réel avec des chemins UNC ;
  4. **Clé dupliquée** — la dernière gagne, sans avertissement. Un fichier de zoo édité à la
     main contient fréquemment deux `(MaximumNumberOfIterations …)` dont l'une est commentée
     à moitié. Elastix, lui, se comporte différemment selon les versions.
- **Conséquences** : le README promet un accueil soigné des fichiers du zoo elastix
  (« those files *are* the interchange format of the elastix world »). Un fichier
  légèrement inhabituel est soit rejeté par une `IndexError` nue, soit accepté avec des
  clés manquantes.
- **Correction recommandée** :
  1. accumuler les lignes jusqu'à la parenthèse fermante avant de découper ;
  2. tenir compte des guillemets dans le retrait des commentaires (le tokeniseur
     `_split_values` sait déjà le faire — appliquer la même logique au retrait de `//`) ;
  3. avertir sur clé dupliquée ;
  4. dans `_validate`, remplacer les accès directs `pmap[k][0]` par un helper
     `_first_value(pmap, key, path)` levant un `ValueError` nommant le fichier et la clé.
- **Tests nécessaires** — un fichier de fixture par cas :
  - `test_a_parameter_entry_without_a_value_is_reported_not_crashed` ;
  - `test_a_multiline_parameter_entry_is_read` ;
  - `test_a_duplicated_key_is_reported` ;
  - `test_a_comment_inside_a_quoted_value_is_preserved`.

---

#### F-05 — `_quote` écrit `nan` et `inf` sans guillemets dans les fichiers de paramètres

- **Emplacement** : `regix/registration/params.py:506-514`.
- **Gravité** : **Cosmétique**
- **Le code** :
  ```python
  try:
      float(text); return text          # non quoté
  except ValueError:
      return f'"{text}"'
  ```
  `float("nan")`, `float("inf")`, `float("Infinity")` et `float("1_000")` réussissent tous
  en Python. Les valeurs correspondantes sont donc écrites **sans guillemets** dans le
  fichier elastix, où `nan`, `inf` et `1_000` ne sont pas des littéraux numériques valides.
- **Atteignable via** `stage.extra: {SomeKey: "nan"}` ou une valeur calculée devenue `NaN`.
- **Conséquences** : fichier de paramètres invalide, refusé par le parseur elastix avec un
  message de bas niveau ; ou, pour `1_000`, une valeur mal interprétée.
- **Correction recommandée** : n'accepter comme numérique non quoté que ce qui correspond à
  une expression décimale stricte —
  `re.fullmatch(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?", text)`.
- **Tests nécessaires** : `test_quote_rejects_non_finite_and_underscored_numbers`
  (paramétré : `nan`, `inf`, `-inf`, `Infinity`, `1_000`, `1e5`, `-3.4`, `true`, `Compose`).

---

#### F-06 — `itk_transform_to_sitk` : le garde-fou de conversion peut passer sans avoir rien vérifié

- **Emplacement** : `regix/registration/itk_bridge.py:107-128`.
- **Gravité** : **Mineur**
- **Le code** :
  ```python
  probes = [(0,0,0), (57,-31,93), (-42,68,-17)]
  errors = []
  for probe in probes:
      try:
          ...
          errors.append(...)
      except Exception:
          continue                      # point hors du support de la B-spline
  if errors and max(errors) > 1e-3:     # <-- « if errors »
      raise ValueError(...)
  log.debug("... (max discrepancy %.2e mm)", max(errors) if errors else 0.0)
  ```
  Si **les trois** sondes lèvent, `errors` est vide, la condition est fausse, la conversion
  est déclarée valide, et le log affiche `max discrepancy 0.00e+00 mm` — un écart nul qui
  n'a jamais été mesuré.
- **Pourquoi c'est important malgré la faible probabilité** : c'est le seul contrôle
  garantissant l'affirmation centrale du README, « its conversion to a `sitk.Transform`
  through HDF5 is **exact** (zero discrepancy on probed points) ». Un garde-fou qui peut
  passer à vide, en journalisant précisément le chiffre que le README cite, est
  trompeur.
- **Défaut lié** : les trois sondes sont des **coordonnées physiques fixes**, sans relation
  avec le champ de vue des images. En coordonnées patient LPS, (0,0,0) est proche de
  l'isocentre et (57,−31,93) tombe généralement dans le patient — ce qui explique que le
  cas dégénéré ne se produise pas en pratique. Mais rien ne le garantit pour une géométrie
  inhabituelle (petit animal, ROI très décentrée, coordonnées en mètres).
- **[À VÉRIFIER]** : je n'ai pas construit de cas où les trois sondes échouent. Le risque
  est identifié par lecture, non reproduit. En revanche, la faiblesse symétrique de
  `linear_matrix_from_transform` / `flatten_linear` (sondes fixes) **a été testée et n'a
  pas reproduit de faux positif** : une B-spline dont le support ne couvrait pas les sondes
  a bien été refusée.
- **Correction recommandée** :
  1. `if not errors: raise ValueError("conversion invérifiable : aucune sonde évaluable")` ;
  2. dériver les sondes de la **géométrie de l'image fixe** — centre, et les 8 coins du FOV
     ramenés à 90 % vers le centre — plutôt que de les fixer en dur. Le `work_dir` passé à
     la fonction identifie déjà le contexte ; il suffit de lui passer l'image de référence.
- **Tests nécessaires** :
  - `test_conversion_guard_fails_when_no_probe_is_evaluable` (transformée factice dont
    `TransformPoint` lève toujours) ;
  - `test_conversion_guard_probes_inside_the_image_field_of_view`.

---

#### F-07 — `load_landmarks` interprète un fichier elastix `index` comme des millimètres

- **Emplacement** : `regix/io/writers.py:54-76` ; `regix/config.py:367-370`.
- **Gravité** : **Important**
- **Le code** :
  ```python
  if lines and lines[0].lower() in ("point", "index"):
      lines = lines[2:]      # on saute l'en-tête... et on continue comme si de rien n'était
  ```
  Le format de points elastix comporte un en-tête d'un mot qui **déclare le système de
  coordonnées** : `point` = coordonnées physiques (mm), `index` = **indices de voxel**.
  Regix reconnaît les deux mots-clés, les saute tous les deux, et parse les valeurs comme
  des millimètres dans les deux cas.
- **Conséquences** : un fichier `index` — format parfaitement légitime, produit par de
  nombreux outils, et explicitement accepté par le docstring de la fonction (« also elastix
  point files ») — est lu comme des millimètres. Les points sont alors totalement faux
  (un indice de 128 devient 128 mm), et **la TRE calculée est absurde**. Or la TRE est
  décrite par le README comme « the only genuinely independent measure » et pilote la porte
  `max_tre_mm` (3 mm dans `mr_ct_prostate`).
  Le résultat le plus probable est un `FAIL` inexplicable ; le résultat le plus dangereux
  est une TRE qui passe par coïncidence.
- **Correction recommandée** :
  ```python
  header = lines[0].lower() if lines else ""
  if header == "index":
      raise ValueError(
          f"{path} déclare des coordonnées en indices de voxel. Regix attend des "
          "coordonnées physiques (mm, LPS). Convertissez avec "
          "TransformIndexToPhysicalPoint, ou fournissez un fichier 'point'."
      )
  if header == "point":
      lines = lines[2:]
  ```
  Mieux : accepter `index` en le convertissant, ce qui exige de passer l'image de référence
  — l'appelant (`_quality_control`) l'a sous la main.
  Ajouter aussi un garde-fou de plausibilité : avertir si l'enveloppe des points est hors du
  FOV de l'image fixe, ce qui attrape aussi bien ce cas qu'une erreur de convention RAS/LPS.
- **Tests nécessaires** :
  - `test_an_index_point_file_is_refused_or_converted` ;
  - `test_landmarks_outside_the_image_field_of_view_are_reported` ;
  - `test_point_header_and_plain_xyz_give_the_same_result`.

---

#### F-08 — `organ_moments_init` ne détecte pas la dégénérescence des axes principaux

- **Emplacement** : `regix/registration/initialize.py:135-191` ;
  `regix/preprocess/geometry.py:297-317` (`principal_axes`).
- **Gravité** : **Mineur**
- **Le mécanisme** : l'appariement des axes repose sur `np.dot(V_f[:,i], V_m[:,i]) < 0` pour
  corriger le signe, puis sur un contrôle de déterminant pour éviter le miroir. Ces deux
  garde-fous supposent que **l'ordre** des axes propres est comparable entre les deux
  volumes. Or `np.linalg.eigh` trie par valeur propre : si deux longueurs d'inertie sont
  proches (organe quasi sphérique — rein, prostate, vessie pleine), l'ordre entre elles est
  déterminé par le bruit. Les axes 1 et 2 peuvent être **permutés** entre fixe et mobile.
- **Conséquences** : `R = V_m @ V_f.T` devient une rotation d'environ 90°, avec un
  déterminant positif (donc non détectée par le contrôle de miroir). L'initialisation part
  d'une orientation franchement fausse. Le mode `organ_moments` n'est pas dans les
  candidats par défaut du multistart, donc l'erreur n'est pas rattrapée par un score
  concurrent : en mode unique, elle est adoptée telle quelle (et B-07 fait que même un
  échec ne se verrait pas).
- **Correction recommandée** : mesurer la séparation des valeurs propres et refuser (ou
  dégrader vers `organ_centroid`) quand deux longueurs d'inertie sont à moins de ~10 % l'une
  de l'autre :
  ```python
  ratios = len_f[:-1] / np.maximum(len_f[1:], 1e-6)
  if np.any(ratios < 1.10):
      log.warning("axes principaux de %s quasi dégénérés (%s) : repli sur le centroïde", chosen, len_f)
      return organ_centroid_init(...)
  ```
  Consigner `axis_separation` dans `info` pour que le rapport le montre.
- **Tests nécessaires** :
  - `test_organ_moments_falls_back_when_the_axes_are_degenerate` (sphère quasi parfaite) ;
  - `test_organ_moments_still_aligns_an_elongated_organ` (non-régression) ;
  - le test existant `test_organ_moments_initialization_never_mirrors`
    (`test_registration_internals.py:100`) couvre déjà le miroir — bon test, à conserver.

---

#### F-09 — Aucune porte QC ne confronte le déplacement mesuré à l'amplitude physiologique attendue

- **Emplacement** : `regix/qc/gates.py:79-314` ; `regix/qc/metrics.py:279-295`
  (`displacement_statistics`, calculé) ; `regix/organs/labels.py:80`
  (`typical_motion_mm`, défini).
- **Gravité** : **Mineur**
- **Constat** : `displacement_statistics` produit `mean_mm`, `p95_mm`, `max_mm` et les place
  dans `metrics["displacement"]` — mais `evaluate_gates` **ne reçoit jamais** ce
  dictionnaire (il n'est pas dans sa signature). Aucune porte ne l'examine. En parallèle,
  chaque profil d'organe déclare `typical_motion_mm` (20 mm pour le foie, 25 pour les
  poumons, 0 pour un fémur), et cette valeur n'est lue nulle part (A-06).
- **Ce qui manque** : la conjonction évidente des deux. Un B-spline qui déplace le foie de
  80 mm alors que l'amplitude respiratoire attendue est de 20 mm est presque certainement
  faux — et **aucune porte actuelle ne l'attrape** : le Jacobien peut rester sain (une
  translation locale ample ne replie rien), le Dice peut être bon (l'organe est bien
  superposé, au mauvais endroit), et `max_translation_mm` ne regarde que la partie
  **linéaire**.
- **Conséquences** : un trou de couverture dans le jeu de portes, sur un mode de défaillance
  réaliste des recalages déformables. Le README annonce pourtant que le profil encode
  « the expected physiological amplitude ».
- **Correction recommandée** :
  1. passer `displacement=metrics.get("displacement")` à `evaluate_gates` ;
  2. ajouter `QCGates.max_displacement_p95_mm: float | None = None` ;
  3. lorsque des organes sont ciblés et que la porte n'est pas fixée explicitement, la
     dériver du profil : `2.5 * merged_profile(targets).typical_motion_mm`, en WARN (pas en
     FAIL — la marge est indicative) ;
  4. pour un profil `deformable: False` avec `typical_motion_mm == 0`, tout déplacement
     non linéaire significatif devrait être un FAIL.
- **Tests nécessaires** :
  - `test_an_implausible_displacement_amplitude_is_flagged` ;
  - `test_the_displacement_gate_is_derived_from_the_organ_profile` ;
  - `test_a_rigid_organ_profile_rejects_a_deformable_displacement`.

---

#### F-10 — `combined_mask` : le repli sur l'union de tous les labels est trop permissif et masqué

- **Emplacement** : `regix/organs/roi.py:54-60`.
- **Gravité** : **Mineur**
- **Le code** :
  ```python
  try:
      mask = seg.mask_for(wanted)
  except ValueError:
      log.warning("target organs missing: falling back to the union of all labels")
      mask = seg.mask_for(None)
  ```
- **Le problème** : l'utilisateur a demandé `--organ prostate`. Aucune prostate n'est
  trouvée. Regix construit alors le masque de **tous** les organes segmentés — chez un
  patient TotalSegmentator, plus de 100 structures couvrant tout le tronc. Le masque de
  critère passe donc de « la prostate + 12 mm » à « pratiquement tout le corps ». Le
  recalage n'est plus du tout celui demandé, et le seul signe est un `log.warning` qui
  **n'atteint ni le manifeste ni le rapport**.
- **Conséquences** : recalage silencieusement transformé en recalage global. Combiné à
  B-07 (l'initialisation `organ_centroid` échoue aussi et se replie sur la géométrie), un
  preset centré organe peut se dégrader intégralement en recalage corps entier, avec un
  statut PASS et deux lignes de log.
- **Correction recommandée** : remonter en `manifest.warn` (donc au rapport), et distinguer
  les cas — si **aucun** organe demandé n'est trouvé, le repli sur le corps entier
  (`body_mask`) est plus honnête que l'union de toutes les structures ; si **certains** sont
  trouvés, n'utiliser que ceux-là et nommer les manquants. Ajouter dans le rapport une ligne
  « organes ciblés / organes réellement utilisés ».
- **Tests nécessaires** :
  - `test_a_missing_target_organ_reaches_the_manifest_and_the_report` ;
  - `test_partially_available_targets_use_only_the_available_ones`.

---

#### F-11 — API : `_JOBS` croît sans borne et le pool n'est jamais fermé

- **Emplacement** : `regix/api.py:73-75`, `:185-191`, `:202-205`.
- **Gravité** : **Mineur**
- **Constats** :
  1. `_JOBS: dict[str, JobStatus]` n'est jamais purgé. Chaque job conserve `metrics`
     — un dictionnaire complet incluant `landmarks.per_landmark_mm` et l'ensemble des
     mesures. Un service tournant en continu accumule indéfiniment ;
  2. `GET /jobs` renvoie **tous** les jobs, sans pagination : la réponse croît linéairement
     et devient un vecteur de fuite d'information (chemins de sortie de tous les patients
     traités) ;
  3. `_POOL = ThreadPoolExecutor(max_workers=1)` est créé à l'import et jamais arrêté :
     pas de `shutdown` sur l'événement d'arrêt FastAPI. Un job en cours est tué brutalement
     à l'arrêt du serveur, laissant un répertoire de sortie partiel ;
  4. il n'existe aucun moyen d'annuler un job.
- **Conséquences** : croissance mémoire non bornée, fuite d'historique, arrêt non propre.
  Cohérent avec les limitations annoncées (« no job persistence »), mais la fuite mémoire
  n'en fait pas partie.
- **Correction recommandée** : borner `_JOBS` par un `OrderedDict` de taille maximale
  configurable (défaut 1 000) avec éviction FIFO des jobs terminés ; paginer `GET /jobs`
  (`limit`/`offset`, défaut 50) ; ajouter un gestionnaire `lifespan` FastAPI appelant
  `_POOL.shutdown(wait=True, cancel_futures=True)`.
- **Tests nécessaires** :
  - `test_finished_jobs_are_evicted_beyond_the_cap` ;
  - `test_jobs_listing_is_paginated` ;
  - `test_the_pool_is_shut_down_on_application_shutdown`.

---

#### F-12 — Le cache de segmentation peut restituer la segmentation d'un **autre** patient

- **Emplacement** : `regix/organs/segmenter.py:117-124` (`_cache_key`), `:282-310` ;
  `regix/pipeline.py:515` (`cache = out_dir / "cache"`).
- **Gravité** : **Mineur** *(probabilité faible, conséquence grave)*
- **Le mécanisme** : la clé de cache est un SHA-256 de
  `f"{name}|{extra}|{size}|{spacing}|{origin}"` suivi de **`arr[::step]` tronqué à 1 Mo**,
  où `step = shape[0] // 8`. Sur un CT 512×512×512 en int16, `arr[::64]` fait 8 coupes de
  512 Ko = 4 Mo, **tronqués à 1 Mo** : la signature ne couvre donc que **les deux premières
  coupes échantillonnées**, soit les coupes 0 et 64 du volume.
- **Le scénario de collision** : deux patients scannés sur la même machine, même protocole,
  même géométrie de reconstruction — donc mêmes `size`, `spacing`, `origin`. Les coupes 0
  et 64 d'un CT thoraco-abdominal sont souvent au-dessus de l'anatomie (air, table) ou dans
  une région peu discriminante. Si elles coïncident au bit près, **la clé est identique**
  et la segmentation du premier patient est restituée pour le second.
- **Circonstances atténuantes réelles** : (a) `cache_dir` vaut `out_dir / "cache"`, donc un
  répertoire de sortie neuf par run — la collision exige de **réutiliser** le même
  répertoire de sortie pour deux patients (`--overwrite`), ce qui est déjà problématique
  par ailleurs (B-09) ; (b) le bruit d'acquisition rend une coïncidence bit-à-bit très
  improbable en pratique. Le risque est donc **faible**, mais son coût est une segmentation
  d'organe attribuée au mauvais patient.
- **Défauts secondaires** : la clé n'inclut ni la version de Regix, ni celle de
  TotalSegmentator — un cache écrit par une version antérieure est réutilisé après mise à
  jour ; `_cached` parse `names[int(idx)] = name` sans `try`, donc un fichier de cache
  corrompu fait planter le run au lieu d'être ignoré.
- **Correction recommandée** :
  1. hacher **l'intégralité** du volume (`arr.tobytes()`) — `hashlib` traite ~1 Go/s, soit
     ~0,5 s pour un CT, négligeable devant une inférence TotalSegmentator ;
  2. inclure `regix.__version__` et `importlib.metadata.version("TotalSegmentator")` dans la
     clé ;
  3. envelopper `_cached` dans un `try/except` qui invalide le cache au lieu de propager ;
  4. documenter l'existence du cache — aujourd'hui totalement absent du README et de la
     configuration.
- **Tests nécessaires** :
  - `test_cache_key_changes_when_a_single_voxel_changes_anywhere` (y compris dans une coupe
    non échantillonnée aujourd'hui — c'est le test qui échoue actuellement) ;
  - `test_cache_key_changes_with_the_totalsegmentator_version` ;
  - `test_a_corrupt_cache_file_is_ignored_not_fatal`.

---

#### F-13 — `_run` de TotalSegmentator : `except ImportError` couvre l'appel entier, masquant les erreurs internes

- **Emplacement** : `regix/organs/segmenter.py:250-279`.
- **Gravité** : **Mineur**
- **Le code** :
  ```python
  try:
      from totalsegmentator.python_api import totalsegmentator
      totalsegmentator(input=..., output=..., ...)     # <-- dans le même try
      return
  except ImportError:
      pass                                              # -> bascule vers la CLI
  ```
  L'appel lui-même est dans le `try`. Une `ImportError` levée **à l'intérieur** de
  TotalSegmentator (nnU-Net absent, poids non téléchargés, dépendance CUDA manquante) est
  interprétée comme « le module n'est pas installé », et Regix bascule silencieusement sur
  l'exécutable — qui échouera de la même façon, avec un message encore plus obscur.
- **Défauts liés** : `subprocess.run(cmd, check=True)` sans `timeout` (une segmentation
  bloquée bloque le run indéfiniment) et sans `capture_output` (la sortie de TS se mélange
  au flux de Regix, hors du fichier de log).
- **Correction recommandée** : restreindre le `try/except ImportError` au seul import ;
  passer `timeout=` (paramétrable, défaut 30 min) et `capture_output=True` en journalisant
  `stderr` en cas d'échec.
- **Tests nécessaires** :
  - `test_an_import_error_inside_totalsegmentator_is_not_mistaken_for_absence` ;
  - `test_the_cli_backend_has_a_timeout`.

---

#### F-14 — `mask_bounding_box_mm` : message d'erreur trompeur sur un label map

- **Emplacement** : `regix/preprocess/geometry.py:214-236`.
- **Gravité** : **Cosmétique**
- **Constat** : la fonction lève `ValueError("empty mask: no bounding box")` dès que le
  label `1` est absent. Sur une image dont les valeurs sont, par exemple, `{0, 3, 7}` —
  un label map passé par erreur au lieu d'un masque binaire — le message affirme que le
  masque est vide alors qu'il ne l'est pas. Le message remonte tel quel à l'utilisateur via
  `plan_roi` (`roi.py:147-149`).
- **Correction recommandée** : binariser en entrée (`sitk.Greater(mask, 0)`) ou distinguer
  les deux cas dans le message (« aucun voxel non nul » vs « le label 1 est absent ;
  labels présents : … »).
- **Tests nécessaires** : `test_bounding_box_of_a_multilabel_map_reports_clearly`.

---

#### F-15 — `dilate_mask_mm` / `erode_mask_mm` : le rayon minimal d'un voxel sur-dilate

- **Emplacement** : `regix/preprocess/geometry.py:198-211`.
- **Gravité** : **Cosmétique**
- **Le code** : `radius = [max(1, int(round(mm / s))) for s in spacing]`.
  Sur un axe de 5 mm d'épaisseur avec `mask_dilate_mm = 2.0`, `round(0.4) = 0`, et le
  `max(1, …)` impose **1 voxel = 5 mm** — soit 2,5 fois la dilatation demandée, sur cet axe
  seulement. La dilatation devient fortement anisotrope sans que rien ne le signale.
- **Cas réels** : CT thoraco-abdominal reconstruit en 5 mm, IRM en coupes épaisses.
- **Conséquences** : masque de critère plus large que voulu selon l'axe de coupe ; effet
  discret sur l'échantillonnage elastix, et sur les volumes rapportés par `roi_overlap_report`.
- **Nuance** : `max(1, ...)` évite un rayon nul qui ferait de la dilatation un no-op — le
  choix se défend. Le défaut est l'absence de signalement.
- **Correction recommandée** : conserver `max(1, …)` mais journaliser en `DEBUG` le rayon
  effectif en millimètres par axe, et avertir quand il dépasse 1,5 × la valeur demandée.
  Le test `test_dilation_in_mm_respects_anisotropy` (`test_units.py:788`) existe : le
  compléter avec le cas de la coupe épaisse.
- **Tests nécessaires** : `test_dilation_warns_when_the_voxel_floor_overshoots`.

---

#### F-16 — `QCConfig.n_slices` et plusieurs champs numériques n'ont pas de contrainte de borne

- **Emplacement** : `regix/config.py:372` (`n_slices`), `:166-176`
  (`sw_overlap`, `patch_size`, `pca_max_voxels`, `sw_batch_size`), `:213` (`roi_margin_mm`),
  `:207` (`mask_dilate_mm`), `:117` (`denoise_sigma_mm`).
- **Gravité** : **Mineur**
- **[CONFIRMÉ]** : `QCConfig(n_slices=-3)` est accepté sans broncher.
- **Conséquences** : `overlay_figure` fait alors `plt.subplots(3, -6)` → exception, capturée
  par le `try/except` de `_quality_control` → « QC figures not generated » : le rapport perd
  toutes ses figures pour une valeur de configuration que la validation aurait dû refuser.
  De même, `sw_overlap` hors de [0, 1) fait échouer monai en cours d'inférence ;
  `mask_dilate_mm` négatif est silencieusement traité comme 0.
- **Contraste** : `StageConfig` est exemplaire à cet égard (`ge=1, le=8`, `gt=0.0, le=1.0`,
  `ge=64`…). L'incohérence est d'autant plus visible.
- **Correction recommandée** : aligner les autres modèles — `n_slices: int = Field(3, ge=1, le=12)`,
  `sw_overlap: float = Field(0.5, ge=0.0, lt=1.0)`, `sw_batch_size: int = Field(1, ge=1)`,
  `pca_max_voxels: int = Field(200_000, ge=1_000)`, `mask_dilate_mm: float = Field(8.0, ge=0.0)`,
  `roi_margin_mm: float = Field(20.0, ge=0.0)`, `denoise_sigma_mm: float | None = Field(None, gt=0.0)`.
- **Tests nécessaires** : `test_out_of_range_config_values_are_refused`, paramétré sur
  l'ensemble des champs numériques du schéma.

---

#### F-17 — `RegistrationResult.status` vaut `"WARN"` quand le QC est désactivé

- **Emplacement** : `regix/pipeline.py:481` ; `regix/cli.py:365-366`.
- **Gravité** : **Cosmétique**
- **Constat** : `status=qc_result.get("status", "WARN") if cfg.qc.enabled else "WARN"`.
  Avec `qc.enabled: false`, le run renvoie `WARN` — donc `result.ok` est faux et le rapport
  affiche un badge orange — alors qu'aucune mesure n'a été faite. Le CLI sort en 0
  (seul `FAIL` donne 2), ce qui est cohérent, mais le statut est ambigu : `WARN` signifie
  ailleurs « une mesure est indisponible ou limite », pas « rien n'a été mesuré ».
- **Correction recommandée** : introduire un quatrième statut `"NOT_EVALUATED"` (ou
  `"SKIPPED"`), documenté et propagé dans le manifeste, l'API et le rapport. C'est la
  distinction que le principe affiché n°3 (« a failure is labelled ») appelle.
- **Tests nécessaires** : `test_qc_disabled_yields_an_explicit_not_evaluated_status`.

---

### G. Performances

Le projet est destiné à des volumes cliniques (CT thoraco-abdominal 512 × 512 × 800,
soit ~210 M voxels). Les évaluations ci-dessous prennent ce volume de référence.

---

#### G-01 — Le QC d'intensité alloue jusqu'à 4 copies float64 du volume complet, quatre fois de suite

- **Emplacement** : `regix/qc/metrics.py:33-46` (`_paired_arrays`), appelé par
  `normalized_cross_correlation` et `normalized_mutual_information`, eux-mêmes appelés
  **quatre fois** par `similarity_report` (`metrics.py:96-107`).
- **Gravité** : **Important**
- **Le calcul** : chaque appel à `_paired_arrays` produit
  ```
  GetArrayFromImage(Cast(a, float32))  ->  copie float32   (840 Mo)
  .astype(np.float64)                  ->  copie float64   (1,7 Go)
  idem pour b                          ->  1,7 Go
  valid = isfinite(a) & isfinite(b)    ->  2 × bool (210 Mo) + résultat
  arr_a[valid], arr_b[valid]           ->  2 copies supplémentaires
  ```
  Soit un pic de l'ordre de **4 à 5 Go** pour un seul appel sur le volume de référence.
  `similarity_report` enchaîne NCC(after), NMI(after), NCC(before), NMI(before) :
  **quatre exécutions complètes**, chacune refaisant l'intégralité des conversions sur les
  mêmes données.
- **Conséquences** : consommation mémoire disproportionnée sur un poste clinique standard
  (16 Go), avec un risque réel de `MemoryError` ou de recours au fichier d'échange, à un
  moment où le recalage — la partie utile — est déjà terminé. Temps de calcul également
  multiplié par 4 sans nécessité.
- **Nuance** : le choix de `GetArrayFromImage` plutôt que `GetArrayViewFromImage` est
  **correct et bien commenté** (`metrics.py:36-38`) — une vue sur une image temporaire
  provoque une violation d'accès. Le problème n'est pas la copie, c'est sa répétition et la
  promotion en float64.
- **Correction recommandée** :
  1. extraire une fois les vecteurs masqués et les réutiliser :
     ```python
     def paired_samples(fixed, before, after, mask) -> dict[str, np.ndarray]:
         # un seul passage, un seul masque calculé, 3 vecteurs float32
     ```
     `similarity_report` prend alors ces vecteurs et calcule les 4 métriques dessus ;
  2. **rester en float32** : la NCC de Pearson en float32 sur 10⁸ échantillons est
     largement suffisante (accumuler la somme en float64 via `np.dot(a, b, dtype=np.float64)`
     suffit à éviter la perte de précision) ;
  3. **sous-échantillonner** au-delà d'un seuil : la NCC et la NMI sur 5 × 10⁶ voxels tirés
     au hasard (graine `runtime.seed`) sont statistiquement indiscernables du calcul complet
     et coûtent 40 fois moins. À rendre explicite dans le manifeste
     (`n_voxels_evaluated` existe déjà — parfait pour y consigner l'échantillonnage).
- **Tests nécessaires** :
  - `test_similarity_report_computes_the_masked_arrays_once` (compteur d'appels) ;
  - `test_subsampled_metrics_match_the_full_computation` (tolérance 1e-3 sur un fantôme) ;
  - `test_similarity_report_peak_memory_stays_bounded` (`tracemalloc`, marqué `slow`).

---

#### G-02 — `Volume.describe()` copie et parcourt le volume complet, deux fois par run

- **Emplacement** : `regix/io/volume.py:83-103` ; appelé par `pipeline.py:211-212`.
- **Gravité** : **Important**
- **Le code** :
  ```python
  arr = self.array(np.float32)          # copie complète (840 Mo)
  finite = arr[np.isfinite(arr)]        # bool (210 Mo) + copie compactée (840 Mo)
  ... np.percentile(finite, 1) ... np.percentile(finite, 99)   # 2 tris partiels
  ```
  `np.percentile` sur 210 M éléments effectue une sélection partielle coûteuse
  (`np.partition`), **deux fois** — les deux appels sont indépendants et ne partagent pas
  le travail.
- **Coût** : ~2 Go de pic et plusieurs secondes **par volume**, soit deux fois par run,
  uniquement pour produire une ligne descriptive du manifeste.
- **Correction recommandée** :
  1. un seul appel `np.percentile(finite, [1, 99])` (l'API accepte une liste — c'est déjà
     fait ailleurs dans le projet, par ex. `report.py:41`) ;
  2. sous-échantillonner : les percentiles 1/99 d'un volume estimés sur 10⁶ voxels tirés au
     hasard sont exacts à la troisième décimale ;
  3. `sitk.StatisticsImageFilter` donne min/max/moyenne/écart-type sans copie numpy ;
     ne recourir à numpy que pour les percentiles et le comptage des non-finis.
- **Tests nécessaires** :
  - `test_describe_is_bounded_in_time_on_a_large_volume` (`slow`) ;
  - `test_describe_percentiles_match_the_exact_computation` (tolérance).

---

#### G-03 — `principal_axes` appelle `TransformContinuousIndexToPhysicalPoint` une fois **par voxel**

- **Emplacement** : `regix/preprocess/geometry.py:297-317`.
- **Gravité** : **Important**
- **Le code** :
  ```python
  idx = np.argwhere(arr > 0)
  points = np.asarray(
      [m.TransformContinuousIndexToPhysicalPoint([float(i[2]), float(i[1]), float(i[0])])
       for i in idx], dtype=np.float64)
  ```
  Une boucle Python avec un franchissement de frontière SWIG **par voxel du masque**.
  Un foie à 2 mm compte ~200 000 voxels ; à pleine résolution, ~1,5 million.
- **Coût estimé** : ~5 à 10 µs par appel SWIG → **1 à 15 secondes par organe et par côté**,
  soit 2 à 30 s pour `organ_moments_init` qui appelle la fonction deux fois. Sur le chemin
  d'initialisation, censé coûter « a few hundred milliseconds per candidate ».
- **Correction recommandée** : la transformation index → physique est une affine, calculable
  d'un coup :
  ```python
  direction = np.asarray(m.GetDirection()).reshape(3, 3)
  spacing   = np.asarray(m.GetSpacing())
  origin    = np.asarray(m.GetOrigin())
  idx_xyz   = idx[:, ::-1].astype(np.float64)          # (z,y,x) -> (x,y,z)
  points    = idx_xyz * spacing @ direction.T + origin
  ```
  Gain attendu : trois à quatre ordres de grandeur. **Le même motif** existe dans
  `organ_centroids` (`roi.py:86`) — mais là il n'est appelé qu'une fois par organe, sur le
  centroïde déjà moyenné : c'est correct, et c'est justement la preuve que l'auteur connaît
  la bonne forme.
- **Tests nécessaires** :
  - `test_principal_axes_vectorised_matches_the_reference` (comparaison à la boucle sur un
    petit masque, tolérance 1e-9) ;
  - `test_principal_axes_is_fast_on_a_realistic_organ` (borne de temps, marqué `slow`).

---

#### G-04 — Les distances de surface sont calculées deux fois par organe

Voir **E-05**. Coût réel : `SignedMaurerDistanceMap` sur un volume 512³ prend de l'ordre de
la seconde ; le doublon coûte donc ~2 secondes supplémentaires **par organe**, soit ~6 s pour
le preset `ct_ct_liver_followup` (3 organes de QC). Gravité **Mineur**.

---

#### G-05 — `MIND-SSC` alloue ~20 volumes float32 simultanément, sans le garde-fou dont dispose anatomix

- **Emplacement** : `regix/features/mind.py:72-86`.
- **Gravité** : **Important**
- **Le décompte** pour un volume de travail de N voxels (float32, 4 octets) :
  ```
  shifted   = np.stack([_shift(vol, …) for … in 6 voisins])   ->  6 N
  distances = np.empty((12,) + vol.shape)                     -> 12 N
  mind      = np.exp(-distances / variance)                   -> 12 N
  + variance (N), + les copies temporaires de _shift (np.roll + .copy())
  ```
  Soit un pic voisin de **30 N × 4 octets**. Pour un volume de travail à 2 mm d'un
  thoraco-abdomen (≈ 256 × 256 × 400 = 26 M voxels) : **~3,1 Go**, avant la PCA.
- **Aggravation par la PCA** : `joint_pca_reduce._project` fait
  `(flat - mean) @ basis.T` où `flat` est `(V, C)` en float32 et `mean` en **float64** —
  la soustraction promeut tout le tableau en float64, soit `26 M × 12 × 8 = 2,5 Go`
  supplémentaires, pour un résultat immédiatement reconverti en float32.
- **Asymétrie révélatrice** : `AnatomixExtractor.extract` **estime et signale** son coût
  (`anatomix.py:155-163`, avertissement au-delà de 4 Go, avec le conseil d'augmenter
  `working_spacing_mm` ou d'activer `roi_crop`). Le chemin MIND — celui qui s'exécute
  précisément sur les machines **sans GPU**, donc souvent avec moins de mémoire —
  n'a **aucun** garde-fou équivalent.
- **Conséquences** : `MemoryError` sur le chemin de repli par défaut, sur les machines les
  plus modestes. C'est le scénario nominal d'un poste clinique sans GPU face à une paire
  multimodale.
- **Correction recommandée** :
  1. reproduire l'estimation d'anatomix dans `mind_ssc_features` (même seuil, même conseil) ;
  2. calculer les 12 canaux **par blocs** plutôt que d'allouer `distances` en entier :
     la boucle sur `_PAIRS` peut écrire directement dans le tableau de sortie et libérer
     `shifted` par paire ;
  3. dans `_project`, forcer `mean = mean.astype(np.float32)` et projeter par tranches
     (`for chunk in np.array_split(...)`), ce qui borne la mémoire à quelques centaines de Mo.
- **Tests nécessaires** :
  - `test_mind_warns_about_memory_on_a_large_volume` ;
  - `test_pca_projection_stays_in_float32` ;
  - `test_mind_output_is_unchanged_after_the_chunked_rewrite` (non-régression numérique
    stricte, à ajouter **avant** le refactoring).

---

#### G-06 — `list_series` lance un scan GDCM par sous-répertoire, sans limite de profondeur

- **Emplacement** : `regix/io/dicom.py:73-89`.
- **Gravité** : **Mineur**
- **Le code** : `directories = [d] + [p for p in d.rglob("*") if p.is_dir()]`, puis un
  `GetGDCMSeriesIDs` par répertoire. Sur une archive patient organisée en
  `étude/série/instance/`, cela peut représenter des milliers d'appels, chacun ouvrant et
  lisant l'en-tête de tous les fichiers du répertoire.
- **Défaut lié** : `rglob("*")` matérialise la liste complète avant de commencer, suit les
  liens symboliques (risque de boucle), et n'a pas de limite de profondeur.
- **Correction recommandée** : ajouter un paramètre `max_depth` (défaut 4, suffisant pour
  toutes les organisations courantes), ne descendre que dans les répertoires contenant des
  fichiers, et journaliser le nombre de répertoires visités. Combiner avec la correction de
  B-14 (fusion par UID), qui restructure de toute façon cette boucle.
- **Tests nécessaires** :
  - `test_list_series_respects_max_depth` ;
  - `test_list_series_does_not_follow_symlink_loops`.

---

#### G-07 — Le rapport HTML pèse ~4 Mo pour 6 Ko de contenu

- **Emplacement** : `regix/qc/report.py:78-86` (`_figure_to_base64`, `dpi=110`) ;
  `:124-126` (`figsize=(3.1 * columns * n_slices, 9.0)`).
- **Gravité** : **Mineur**
- **[CONFIRMÉ]** sur le rapport réel :
  ```
  report.html : 3,92 Mo   |   contenu hors images : 6,4 Ko
  ```
  Soit **99,8 %** du fichier en PNG base64. Avec `n_slices=3` et deux colonnes
  (avant/après), la figure d'overlay fait 18,6 pouces de large à 110 dpi, soit ~2 050 px.
- **Conséquences** : le README met en avant que le rapport « can be emailed » — 4 Mo passe,
  mais dépasse la limite de pièce jointe de certaines messageries hospitalières (souvent
  2 à 5 Mo après encodage MIME, qui ajoute encore 33 %). En traitement par lots, 1 000 cas
  produisent 4 Go de rapports.
- **Correction recommandée** :
  1. encoder les figures en **WebP** ou en PNG 8 bits palettisé — les rendus en niveaux de
     gris avec une superposition « hot » compressent très bien ; gain typique 3 à 5× ;
  2. ramener `dpi` à 90 et rendre le couple (dpi, format) configurable
     (`qc.figure_dpi`, `qc.figure_format`) ;
  3. offrir `qc.report_images: "embedded" | "sidecar"` — en mode *sidecar*, écrire les PNG
     à côté du HTML, ce qui divise la taille par 100 pour l'archivage, l'embarqué restant
     le défaut pour l'envoi par courriel.
- **Tests nécessaires** :
  - `test_report_size_stays_under_a_budget` (par exemple 2 Mo sur le fantôme) ;
  - `test_sidecar_mode_writes_images_next_to_the_html`.

---

#### G-08 — Le masque corporel et les statistiques d'intensité sont recalculés à pleine résolution

- **Emplacement** : `regix/pipeline.py:240-246` (masques QC pleine résolution) ;
  `regix/preprocess/geometry.py:151-184` (`body_mask`).
- **Gravité** : **Mineur**
- **Constat** : `body_mask` enchaîne `BinaryMorphologicalClosing` (boule de rayon 5 mm),
  `BinaryFillhole` et `ConnectedComponent` **sur le volume à pleine résolution**, deux fois
  (fixe et mobile), avant même le préprocessing. La fermeture morphologique 3D avec un
  élément structurant sphérique est l'une des opérations les plus coûteuses de SimpleITK.
- **Coût estimé** : plusieurs secondes à plusieurs dizaines de secondes par volume sur un
  CT complet, pour un masque dont l'usage (délimiter le corps) ne demande aucune précision
  sub-centimétrique.
- **Correction recommandée** : calculer le masque corporel sur une version sous-échantillonnée
  à 4 mm, puis le rééchantillonner (plus proche voisin) sur la grille cible. Cela résout
  **simultanément** l'écart de volume documenté dans les limitations du README et le
  problème A-04, puisqu'il n'y aurait plus qu'un seul masque, calculé une fois.
- **Tests nécessaires** :
  - `test_body_mask_at_4mm_matches_the_full_resolution_mask` (tolérance 3 % en volume) ;
  - `test_body_mask_is_computed_once_per_volume` (compteur d'appels).

---

#### G-09 — Le champ de déplacement dense est matérialisé même pour une transformée purement linéaire

- **Emplacement** : `regix/pipeline.py:770-786` ;
  `regix/registration/convexadam.py:186-194`.
- **Gravité** : **Mineur**
- **Le mécanisme** : dès que `qc.jacobian` est vrai (défaut), le pipeline appelle
  `applied.displacement_field(fixed.image)`, qui matérialise un
  `sitk.Image` de vecteurs **float64** sur la grille fixe complète — soit
  `210 M × 3 × 8 = 5 Go` pour le volume de référence. Puis
  `DisplacementFieldJacobianDeterminant` en produit un autre (scalaire float64, 1,7 Go).
- **Or, pour une transformée linéaire, le déterminant du Jacobien est constant** — le code
  le sait parfaitement : `jacobian_figure` refuse d'en tracer la carte pour cette raison
  précise, avec un commentaire explicite (`report.py:249-256`). Mais le champ est quand même
  calculé intégralement pour en extraire une valeur unique.
- **Conséquences** : 7 Go d'allocations et plusieurs secondes, sur **tous** les recalages
  rigides et affines — soit 3 presets sur 8 (`ct_cbct_igrt`, `mr_mr_brain`,
  `pet_ct_wholebody`) et le preset `base`.
- **Correction recommandée** : court-circuiter le cas linéaire —
  ```python
  matrix = to_matrix_4x4(applied.as_sitk_transform())
  if matrix is not None:
      det = float(np.linalg.det(matrix[:3, :3]))
      jacobian = {"available": True, "linear": True, "det_min": det, "det_max": det,
                  "det_mean": det, "det_std": 0.0, "folding_voxels": 0,
                  "folding_fraction": 0.0 if det > 0 else 1.0, "n_voxels": None}
  ```
  Ne matérialiser le champ que si `to_matrix_4x4` renvoie `None`, ou si
  `output.write_deformation_field` est explicitement demandé.
  Passer par ailleurs le champ en **float32** : la précision est très largement suffisante
  pour un déplacement en millimètres, et cela divise la mémoire par deux.
- **Tests nécessaires** :
  - `test_a_linear_transform_reports_a_constant_jacobian_without_materialising_a_field`
    (compteur d'appels sur `TransformToDisplacementFieldFilter`) ;
  - `test_the_analytical_and_dense_jacobians_agree_for_an_affine` (tolérance 1e-6).

---

### H. Types et contrats d'interface

---

#### H-01 — `target_registration_error` : un contrat « callable » détourné par un lambda qui ignore son argument

- **Emplacement** : `regix/qc/metrics.py:205-237` ; appelé par `regix/pipeline.py:800`.
- **Gravité** : **Mineur**
- **Le contrat déclaré** : « `transform_fixed_to_moving` is either a `sitk.Transform` **or a
  callable mapping a point** from the fixed frame to the moving frame ».
- **L'appel réel** :
  ```python
  mapper = applied.transform_points(pts_fixed)          # un tableau (N, 3) déjà calculé
  landmarks = target_registration_error(pts_fixed, pts_moving, lambda _p: mapper)
  ```
  Le lambda **ignore son argument** et renvoie un tableau pré-calculé. La fonction fait
  ensuite `transform_fixed_to_moving(f)` — le contrat est formellement respecté (un
  callable prenant un tableau et renvoyant un tableau), mais la sémantique annoncée
  (« mapping a point ») ne l'est pas.
- **Le risque concret** : le paramètre est nommé au singulier et documenté au singulier.
  Un appelant qui suivrait la documentation à la lettre passerait une fonction
  point-par-point ; `transform_fixed_to_moving(f)` l'appellerait avec un tableau `(N, 3)`
  entier, et la plupart des implémentations naïves renverraient un résultat de forme
  incorrecte — attrapé par `.reshape(-1, 3)` seulement si la taille totale coïncide, sinon
  silencieusement mal interprété.
- **Détail supplémentaire** : la discrimination
  `if callable(x) and not hasattr(x, "TransformPoint")` est fragile — un objet exposant les
  deux (un `sitk.Transform` enveloppé dans un `functools.partial`, par exemple) prendrait la
  mauvaise branche.
- **Correction recommandée** : remplacer le paramètre par une **union explicite** de deux
  types et une surcharge claire :
  ```python
  def target_registration_error(fixed_points, moving_points,
                                mapped_points: np.ndarray | None = None,
                                transform: sitk.Transform | None = None) -> dict:
  ```
  Le pipeline passe alors `mapped_points=mapper`, ce qui est ce qu'il veut réellement dire,
  et le lambda disparaît. Vérifier `len(mapped) == len(fixed_points)` explicitement.
- **Tests nécessaires** :
  - `test_tre_accepts_precomputed_points_and_a_transform_and_agrees` ;
  - `test_tre_refuses_a_mapped_array_of_the_wrong_length`.

---

#### H-02 — `AppliedTransform` : un contrat à trois implémentations aux comportements divergents

- **Emplacement** : `regix/registration/warp.py:27-153`.
- **Gravité** : **Mineur**
- **Divergences non documentées entre `SitkAppliedTransform` et `ElastixAppliedTransform`** :

  | Méthode | `SitkAppliedTransform` | `ElastixAppliedTransform` |
  |---|---|---|
  | `transform_points` | tableau `(N,3)` | **`None`** si `_linear` absent |
  | `as_sitk_transform` | la transformée | **`None`** si `_linear` absent |
  | `displacement_field` | toujours un champ | `None` si transformix échoue |
  | `resample(is_label=True)` | plus proche voisin | interpolation d'ordre 0 + `Round` + `Cast` |
  | effets de bord | aucun | **écrit sur disque** dans `work_dir` |

  Le type de retour `np.ndarray | None` force chaque appelant à traiter le `None`, et
  `_quality_control` le fait correctement (`pipeline.py:794-798`, avec repli sur le champ).
  Mais rien dans le contrat abstrait n'indique **quand** `None` survient, ni que `resample`
  peut écrire des fichiers.
- **Conséquence concrète** : `warp_landmarks_moving_to_fixed` (`warp.py:171-189`) appelle
  `applied.as_sitk_transform()` puis `t.GetInverse()`. Pour un `SitkAppliedTransform`
  portant un `CompositeTransform` contenant une B-spline, `GetInverse()` **lève** —
  correctement capturé, et le refus de renvoyer une inverse approchée est un **bon choix**
  documenté par le README. Mais la même méthode renvoie `None` pour trois raisons
  différentes (pas de transformée, inversion impossible, transformée non linéaire) sans que
  l'appelant puisse les distinguer.
- **Correction recommandée** : documenter dans la classe abstraite les conditions exactes de
  `None` pour chaque méthode ; ajouter une propriété
  `capabilities: frozenset[Literal["points","inverse","field","sitk"]]` que chaque
  implémentation renseigne, pour que l'appelant interroge plutôt que de tester `None` ;
  déclarer explicitement que `ElastixAppliedTransform.resample` écrit dans `work_dir`.
- **Tests nécessaires** :
  - `test_applied_transform_capabilities_match_the_actual_behaviour` (paramétré sur les
    deux implémentations et sur des transformées linéaire / B-spline / dense) — c'est un
    test de contrat, celui qui manque à cette hiérarchie.

---

#### H-03 — `StageConfig.extra` : un type qui accepte `bool` par accident et perd la distinction

- **Emplacement** : `regix/config.py:299-307`.
- **Gravité** : **Cosmétique**
- **Le type** : `dict[str, list[str | float | int] | str | float | int]`.
  En Python, `bool` est une sous-classe de `int` : `{"WriteResultImage": True}` est accepté
  et stringifié en `"True"` par `_as_tuple`, alors qu'elastix attend `"true"` (minuscule).
  `_quote` corrige le cas (`text.lower() in ("true","false")` → `"true"`), donc **le
  résultat final est correct** — mais par un rattrapage en aval, non par le type.
- **Détail lié** : le commentaire du champ explique bien pourquoi les nombres n'ont pas
  besoin de guillemets (« YAML parses a numeric list as ints ») — bonne justification.
  Il ne mentionne pas les booléens.
- **Correction recommandée** : ajouter `bool` explicitement à l'union pour rendre l'intention
  visible, et documenter la conversion en minuscules.
- **Tests nécessaires** : `test_extra_accepts_a_python_bool_and_emits_lowercase`.
  Le test `test_extra_accepts_a_numeric_list` (`test_units.py:389`) existe et couvre le
  cas voisin.

---

#### H-04 — `Volume.with_image` partage le dictionnaire `meta` avec l'objet d'origine

- **Emplacement** : `regix/io/volume.py:75-76`.
- **Gravité** : **Mineur**
- **Le code** : `return replace(self, image=image, **overrides)` — `dataclasses.replace`
  effectue une copie **superficielle** : le nouveau `Volume` partage la **même instance**
  de `meta` que l'original.
- **Statut actuel** : aucun appelant ne mute `meta` en place. `apply_intensity_prep`
  (`intensity.py:229-230`) fait `out.meta = {**volume.meta, "intensity_prep": applied}` —
  une **réaffectation**, donc sans effet de bord. Le code est correct aujourd'hui.
- **Le risque** : c'est un piège latent. Un futur `work.meta["warnings"].append(...)`
  — motif naturel, et `meta["warnings"]` **existe déjà** (`dicom.py:209`) — corromprait
  silencieusement le volume d'origine, et donc `manifest.inputs`, déjà sérialisé à partir
  de `describe()`.
- **Correction recommandée** : `replace(self, image=image, meta=dict(self.meta), **overrides)`,
  ou geler `meta` en `MappingProxyType`. Coût nul, piège supprimé.
- **Tests nécessaires** : `test_with_image_does_not_share_mutable_metadata`.

---

#### H-05 — `compose([t])` renvoie l'objet d'origine, pas une copie

- **Emplacement** : `regix/registration/transforms.py:85-99`.
- **Gravité** : **Cosmétique**
- **Le code** : `if len(valid) == 1: return valid[0]`. La transformée retournée est
  **la même instance** que celle passée. Un appelant qui la modifierait (`SetParameters`,
  `SetCenter`) modifierait aussi la source.
- **Cas atteignable** : `engine.run` fait
  `outcome.final_linear_transform = compose(chain)` ; avec un seul stage et pas
  d'initialisation, `final_linear_transform` **est** `outcome.stages[0].transform`. Les deux
  champs du même objet `RegistrationOutcome` sont alors des alias, ce que rien n'indique.
- **Correction recommandée** : renvoyer une copie
  (`sitk.Transform(valid[0])` ou le constructeur de copie du type concret), ou documenter
  explicitement l'aliasing dans le docstring.
- **Tests nécessaires** : `test_compose_of_one_returns_an_independent_transform`.

---

#### H-06 — `ParamContext` : tous les champs ont une valeur par défaut, ce qui autorise les contextes incomplets

- **Emplacement** : `regix/registration/params.py:124-138`.
- **Gravité** : **Mineur**
- **Constat** : la dataclass donne un défaut à ses neuf champs (`dimension=3`,
  `n_channels=1`, `fixed_modality="UNKNOWN"`, `intensity_range=None`…). Construire
  `ParamContext()` est donc légal et produit un contexte plausible mais vide de sens.
  **C'est précisément ce qui a rendu B-04 possible et invisible** : l'oubli de
  `intensity_range` n'a produit ni erreur de type, ni exception, ni avertissement — juste
  un `None` silencieux qui désactive un contrôle de sécurité.
- **Correction recommandée** : retirer les valeurs par défaut des champs qui décrivent le
  run (`fixed_modality`, `moving_modality`, `n_voxels`, `intensity_range`,
  `working_spacing_mm`) pour forcer leur transmission ; ne garder de défaut que pour
  `dimension`. Combiner avec l'usage de `dataclasses.replace` recommandé en B-04, qui rend
  l'oubli structurellement impossible.
- **Tests nécessaires** :
  - `test_param_context_requires_the_run_describing_fields` ;
  - `test_stage_context_carries_every_field` (déjà listé en B-04).

---

#### H-07 — Les métriques renvoient `float("nan")` là où `None` serait le contrat juste

- **Emplacement** : `regix/qc/metrics.py:115-116` (`_round`) et l'ensemble du module.
- **Gravité** : **Mineur**
- **Constat** : `_round` renvoie `float("nan")` pour toute valeur non finie ou absente.
  Ces `NaN` remontent dans `metrics`, puis dans `RunManifest.metrics`, puis dans
  `json.dumps`.
- **`json.dumps` accepte `NaN` par défaut** (`allow_nan=True`) et écrit le jeton littéral
  `NaN`, **qui n'est pas du JSON valide** (RFC 8259). Un consommateur strict
  (`JSON.parse` en JavaScript, `jq`, la plupart des parseurs Java/Go/Rust) rejette le
  fichier.
- **[CONFIRMÉ]** sur le manifeste réel : **0 jeton `NaN`**, analyse stricte réussie — parce
  que ce run nominal n'a produit aucune métrique indisponible. Le défaut est donc
  **conditionnel** : il se manifeste exactement sur les runs dégradés, c'est-à-dire ceux
  qu'on cherche le plus à analyser après coup.
- **Chemins produisant un `NaN`** : NCC sur moins de 64 voxels, NMI sur moins de 256,
  Dice sur deux masques vides, HD95 / MSD sur un masque vide, toute métrique dont la
  transformée est non convertible.
- **Effet sur l'API** : FastAPI sérialise `JobStatus.metrics` ; selon la version de
  pydantic/starlette, un `NaN` produit soit un JSON invalide, soit une erreur de
  sérialisation renvoyée au client.
- **Correction recommandée** : renvoyer `None` plutôt que `NaN` depuis `_round`, et adapter
  les tests de finitude (`np.isfinite(gain)` devient `gain is not None and np.isfinite(gain)`,
  motif déjà utilisé dans `gates.py:100`). Passer `allow_nan=False` à `json.dumps` dans
  `RunManifest.save` **comme filet** : mieux vaut une exception au moment de l'écriture
  qu'un manifeste invalide.
- **Tests nécessaires** :
  - `test_manifest_is_strict_json_even_when_metrics_are_unavailable` (forcer un masque
    minuscule pour produire une NCC indisponible, puis
    `json.loads(raw, parse_constant=raise)`) — **c'est le test qui manque** ;
  - `test_the_api_serialises_unavailable_metrics_as_null`.

---

#### H-08 — `evaluate_gates` : sept paramètres optionnels sans structure, dont aucun n'est validé

- **Emplacement** : `regix/qc/gates.py:79-88`.
- **Gravité** : **Cosmétique**
- **Constat** : la signature prend `similarity`, `organ_overlap`, `jacobian`,
  `linear_analysis`, `landmarks`, `deformable`, `stages` — sept dictionnaires libres
  (`dict[str, Any]`), tous optionnels, dont la structure attendue n'est décrite nulle part.
  Un renommage de clé côté producteur (par exemple `ncc_gain` → `ncc_delta`) ne casse rien :
  la porte passe simplement en WARN « non calculable ». **Une régression silencieuse
  dégrade le QC au lieu de le faire échouer.**
- **C'est le même mécanisme que B-04**, transposé : un contrat implicite entre deux modules,
  garanti par aucun type.
- **Correction recommandée** : des `TypedDict` (ou des dataclasses légères) pour
  `SimilarityReport`, `JacobianStats`, `LinearAnalysis`, `LandmarkReport`, `StageSummary`.
  Aucun coût à l'exécution, et un `mypy` en CI (J-04) détecterait alors ces désaccords.
- **Tests nécessaires** :
  - `test_gates_fail_loudly_on_an_unexpected_metric_schema` — passer un dictionnaire aux
    clés renommées et vérifier qu'une erreur est levée plutôt qu'un WARN.

---

#### H-09 — Aucune vérification de types statique n'est exécutée, alors que le code est intégralement annoté

- **Emplacement** : `pyproject.toml:95-107` (ruff : `E, F, W, I, B, UP`) ;
  `.github/workflows/ci.yml:21-39` (job `lint`).
- **Gravité** : **Mineur**
- **Constat** : l'ensemble du code porte des annotations soignées
  (`from __future__ import annotations` partout, unions modernes, `Literal`, génériques).
  Il y a même des `# type: ignore[arg-type]` ciblés (`geometry.py:59`,
  `params.py:310`, `segmenter.py:172`) — signe que **mypy a été exécuté à un moment**.
  Mais aucun vérificateur de types ne figure dans les dépendances `dev` ni dans la CI.
- **Ce que mypy aurait attrapé dans cet audit** : B-04 (champ manquant : non — les défauts
  le masquent, cf. H-06), H-01 (contrat callable ambigu : partiellement),
  et surtout tout futur désaccord de schéma entre modules (H-08).
  Avec les `TypedDict` de H-08, le gain deviendrait substantiel.
- **Correction recommandée** : ajouter `mypy` aux extras `dev`, avec une configuration
  progressive (`ignore_missing_imports = true` pour `itk`, `SimpleITK`, `monai`,
  `totalsegmentator` ; `strict = false` au départ, module par module ensuite), et un job CI
  `continue-on-error: true` dans un premier temps — exactement le traitement déjà réservé à
  `ruff format` (`ci.yml:39`), qui est le bon compromis.
- **Tests nécessaires** : le job CI est la vérification.

---

### I. Tests et scénarios non couverts

La suite est **substantielle et de bonne facture** (122 tests, fantômes numériques bien
conçus, vérité terrain exacte, aucun besoin de données patient). Les critiques ci-dessous
portent sur des **trous précis**, pas sur la qualité générale — qui est traitée en §4.

---

#### I-01 — `regix/api.py` n'est couvert par **aucun** test : 0 ligne exécutée

- **Emplacement** : `regix/api.py` (206 lignes) ; `tests/` (aucun fichier `test_api.py`).
- **Gravité** : **Important**
- **[CONFIRMÉ]** — lecture du fichier `.coverage` du dépôt :
  ```
  regix/api.py  ->  0 ligne mesurée
  ```
  Recherche de `regix.api`, `fastapi`, `TestClient` dans `tests/` : **aucune occurrence**.
- **Ce qui n'est donc jamais exercé** : la construction de configuration
  (`_build_config`, qui contient la logique d'override et le `overwrite=True` forcé), la
  validation d'entrée de `POST /register`, le cycle de vie des jobs, la sérialisation de
  `JobStatus` (donc le problème `NaN` de H-07), `GET /health`, `GET /presets`,
  `GET /jobs`, la gestion d'erreur de `_run_job`.
- **Le README est honnête** sur le principe : « The uncovered quarter is concentrated in the
  paths that need hardware or third-party weights this project does not redistribute —
  anatomix inference, the GPU deformable stage, the TotalSegmentator call, **and the HTTP
  service**. » Mais l'API est la seule des quatre qui **ne nécessite ni matériel ni poids** :
  `fastapi` et `httpx` s'installent en CPU en quelques secondes, et `TestClient` permet de
  tester l'ensemble sans réseau. Le ranger avec les chemins « hardware-dependent » n'est pas
  exact.
- **Conséquences** : la surface la plus exposée du projet (C-03) est aussi la seule
  totalement non testée. Une régression y passerait inaperçue jusqu'au déploiement.
- **Correction recommandée** : créer `tests/test_api.py` avec
  `fastapi.testclient.TestClient`, marqué `pytest.importorskip("fastapi")` pour rester
  optionnel, et ajouter un job CI `pip install -e ".[api,dev]"` qui l'exécute. Couverture
  visée : les 6 routes, les 4 branches de `_build_config`, l'échec de job, et les
  vérifications de sécurité de C-03.
- **Tests nécessaires** (le fichier entier) : voir la liste sous C-03 et F-11, plus
  `test_register_returns_202_and_a_job_id`, `test_polling_an_unknown_job_returns_404`,
  `test_health_reports_degraded_without_the_engine`,
  `test_presets_endpoint_lists_every_bundled_preset`.

---

#### I-02 — Le test de la SRO s'appelle `..._is_valid` mais ne vérifie aucune exigence de l'IOD

- **Emplacement** : `tests/test_dicom_io.py:227` ; l'objet testé est
  `regix/io/writers.py:215-328`.
- **Gravité** : **Important**
- **Constat** : le test vérifie le SOP Class, `Modality == "REG"` et la matrice. Il ne
  vérifie **ni** le nombre d'instances référencées (C-06, défaut 1 : une seule coupe sur
  plusieurs centaines), **ni** la présence des séquences requises, **ni** la complétude
  des attributs de type 2.
- **Pourquoi c'est un problème de test et pas seulement de code** : le nom du test affirme
  la validité. Un mainteneur qui le voit passer conclut que l'objet est conforme. Un test
  dont le nom promet plus que ses assertions est un faux filet de sécurité — c'est ce qui
  explique que C-06 ait survécu.
- **Correction recommandée** : soit renommer en
  `test_spatial_registration_object_has_the_expected_matrix_and_modality`, soit — de
  préférence — étendre les assertions et ajouter la validation externe décrite en C-06.
- **Tests nécessaires** : la liste de C-06.

---

#### I-03 — Le garde-fou de quantification est testé en isolation, jamais à travers le pipeline

- **Emplacement** : `tests/test_units.py:336`
  (`test_an_integer_pixel_type_on_rescaled_data_is_reported`).
- **Gravité** : **Important**
- **Constat** : c'est le trou de test qui a laissé passer **B-04**. Le test construit
  lui-même un `ParamContext` en fournissant `intensity_range`, appelle
  `build_parameter_map`, et vérifie le warning. Il valide donc parfaitement la fonction —
  et pas du tout son câblage. Le pipeline, lui, perd le champ en chemin.
- **Le motif est général** : plusieurs mécanismes documentés sont testés au niveau unitaire
  sans test d'intégration correspondant. Même schéma pour :
  - `_features_wanted` / le repli MIND (B-08) — testé via
    `test_multimodal_with_the_mind_descriptor`, mais uniquement dans le cas
    « torch absent », jamais « torch présent sans GPU » ;
  - `warp_landmarks_moving_to_fixed` (E-03) — testé, mais jamais appelé en production ;
  - `body_mask` — jamais testé du tout après fenêtrage (A-04).
- **Correction recommandée** : pour chaque mécanisme de sécurité documenté, exiger **un**
  test qui l'exerce à travers `RegistrationPipeline.run`. Établir la règle : « un garde-fou
  décrit dans le README a un test d'intégration, pas seulement un test unitaire ».
- **Tests nécessaires** : ceux listés en B-04, B-08 et A-04.

---

#### I-04 — Aucun test ne s'exécute sur un volume anisotrope à coupes épaisses, ni sur une acquisition oblique

- **Emplacement** : `tests/conftest.py:29-36` — le fantôme unique
  (`shape=(64,80,80)`, `spacing=(2.0,2.0,2.5)`, direction **identité implicite**).
- **Gravité** : **Important**
- **Constat** : tous les fantômes partagent une direction identité (jamais fixée, donc
  `(1,0,0, 0,1,0, 0,0,1)`) et un rapport d'anisotropie de 1,25.
- **Ce qui n'est donc jamais exercé** :
  1. **les cosinus directeurs**. C'est l'affirmation la plus insistante du projet —
     « direction cosines are honoured everywhere (`UseDirectionCosines true`) — the most
     common omission, and **fatal on oblique acquisitions** » (`README.md:294-296`), reprise
     dans `params.py`, `itk_bridge.py` et `ENFORCED_WITH_PARAMETER_FILE`. Le test
     `test_direction_cosines_are_always_enabled` (`test_units.py:160`) vérifie seulement que
     la **chaîne** `"true"` est écrite dans la carte de paramètres. **Aucun test ne recale
     jamais deux volumes obliques.** L'affirmation centrale du projet en matière de
     géométrie est donc non vérifiée de bout en bout ;
  2. la conversion SimpleITK ↔ ITK sur une direction non triviale
     (`sitk_to_itk` / `itk_to_sitk`, `itk_bridge.py:78, 89`) ;
  3. `_normalized_to_world_field` (`convexadam.py:158-183`), dont le docstring souligne
     que la direction est l'une des « trois conversions successives, chacune une source
     classique d'erreur » — et qui n'est couvert par aucun test ;
  4. l'anisotropie forte (5 mm de coupe), qui déclenche F-15 (sur-dilatation) et change
     le rayon de fermeture de `body_mask` (A-04).
- **Correction recommandée** : paramétrer `make_phantom` par une direction et une épaisseur
  de coupe, et **paramétrer les tests clés** sur trois géométries :
  identité / oblique 15° autour de x / anisotrope 1×1×5 mm. Le coût est faible (le fantôme
  est petit) et le gain considérable : c'est la classe de bugs la plus coûteuse à
  diagnostiquer en clinique.
- **Tests nécessaires** :
  - `test_rigid_recovers_the_ground_truth[oblique]` — variante paramétrée du test principal ;
  - `test_itk_bridge_round_trip_preserves_an_oblique_direction` ;
  - `test_displacement_field_conversion_honours_the_direction_cosines` ;
  - `test_body_mask_on_thick_slices`.

---

#### I-05 — Aucun test de sérialisation stricte du manifeste, ni de contrat sur les fichiers produits

- **Emplacement** : `tests/test_pipeline.py:342`
  (`test_html_report_and_manifest_are_produced`).
- **Gravité** : **Mineur**
- **Constat** : le test vérifie l'existence des fichiers et quelques clés. Il ne vérifie
  pas :
  - que le JSON est **strictement** valide (H-07 : `NaN` sur les runs dégradés) ;
  - que l'arborescence produite correspond exactement à ce qui est annoncé (D-06) ;
  - qu'aucun chemin patient n'y figure (C-02) ;
  - que le manifeste est stable entre deux runs identiques (E-09).
- **Correction recommandée** : un test de contrat unique
  `test_run_artifacts_contract` faisant les quatre vérifications, exécuté sur un run nominal
  **et** sur un run volontairement dégradé (masque minuscule → NCC indisponible, landmarks
  absents, organe manquant).
- **Tests nécessaires** : celui-ci, plus ceux listés en C-02, E-09 et H-07.

---

#### I-06 — La porte de couverture est calculée sur une exécution partielle, ce qui la rend plus lâche qu'annoncée

- **Emplacement** : `.github/workflows/ci.yml:79-83`.
- **Gravité** : **Mineur**
- **Le workflow** :
  ```yaml
  - name: Coverage gate
    if: matrix.python == '3.12'
    run: pytest -q --cov=regix --cov-report=term-missing:skip-covered --cov-fail-under=75
  ```
  Cette étape **relance toute la suite une cinquième fois** (après les 4 étapes
  `pytest tests/test_units.py`, `test_cli.py test_dicom_io.py`, `test_pipeline.py`) —
  soit un doublement du temps du job, pour la seule mesure de couverture.
- **Le commentaire affirme** : « The threshold sits just below the figure on the README
  badge, so the badge cannot silently drift: a drop in coverage turns CI red. » Le badge
  annonce 78 %, le seuil est 75 % : il y a donc **3 points de jeu**. Le badge *peut* dériver
  de 3 points sans que la CI réagisse. Le commentaire dit « cannot ».
- **Correction recommandée** : mesurer la couverture **pendant** les étapes existantes
  (`--cov=regix --cov-append` sur chacune) plutôt que de relancer la suite ; resserrer le
  seuil à la valeur réelle moins 1 point, et **régénérer le badge depuis la même mesure**
  pour supprimer la possibilité de dérive.
- **Tests nécessaires** : la CI est la vérification.

---

#### I-07 — Aucun test ne couvre le mode `batch` au-delà du cas nominal

- **Emplacement** : `tests/test_cli.py:334, 372`.
- **Gravité** : **Mineur**
- **Couvert** : le cas nominal (`test_batch_produces_a_summary`) et l'absence de colonnes
  requises.
- **Non couvert** : `--stop-on-error` ; une ligne dont le fichier n'existe pas ; les noms
  de cas dangereux (C-04) ; les noms dupliqués ; un CSV avec BOM et séparateur `;`
  (Excel français produit exactement cela — `csv.DictReader` avec le dialecte par défaut
  lira une seule colonne, et le message d'erreur parlera de colonnes manquantes sans
  mentionner le séparateur) ; le code de sortie 2 en présence d'un FAIL.
- **Correction recommandée** : ajouter la détection de dialecte
  (`csv.Sniffer().sniff(sample, delimiters=",;\t")`) — c'est un cas d'usage réel pour le
  public visé — et les tests correspondants.
- **Tests nécessaires** :
  - `test_batch_reads_a_semicolon_separated_csv` ;
  - `test_batch_stop_on_error_aborts_at_the_first_failure` ;
  - `test_batch_exit_code_is_2_when_a_case_fails` ;
  - plus ceux de C-04.

---

#### I-08 — Aucun test ne vérifie le comportement en l'absence des dépendances optionnelles

- **Emplacement** : transversal.
- **Gravité** : **Mineur**
- **Constat** : la CI n'installe **que** `.[dev]`. Tous les tests s'exécutent donc dans une
  seule configuration : torch absent, TotalSegmentator absent, fastapi absent.
  Conséquence : les chemins « dépendance présente » ne sont jamais exercés, et — plus
  gênant — les chemins « dépendance présente mais inutilisable » (B-08 : torch sans GPU)
  non plus.
  Symétriquement, rien ne garantit que le **cœur** reste indépendant : un `import torch`
  ajouté par mégarde au niveau module de `pipeline.py` ferait échouer tous les tests, ce qui
  est bien — mais un import ajouté dans une branche rarement prise passerait.
- **Correction recommandée** :
  1. une matrice CI à 2 axes : `[core, features, api]` × `[3.10, 3.12]`, en réduisant la
     matrice Python actuelle pour compenser le coût ;
  2. des tests de **simulation d'absence/présence** via `monkeypatch.setitem(sys.modules, …)`
     — c'est ce que fait déjà `test_multimodal_with_the_mind_descriptor` et le motif est bon ;
  3. le test d'isolation d'import mentionné en E-04 et D-04.
- **Tests nécessaires** : ceux de E-04, D-04, B-08.

---

#### I-09 — Le test de non-régression du fichier de zoo dépend d'une fixture dont l'historique montre la fragilité

- **Emplacement** : `tests/data/Parameters.Par0008.affine.txt` ;
  `tests/test_units.py:297` (`test_the_real_zoo_fixture_reached_the_repository`) ;
  commit `cf56b1f` « fix(ci): commit the zoo test fixture that .gitignore was swallowing ».
- **Gravité** : **Cosmétique** *(déjà corrigé — mentionné pour la complétude)*
- **Constat** : la règle `data/` du `.gitignore` avalait `tests/data/`, ce qui a rendu la
  CI rouge. La correction (`!tests/data/`) est en place, et **un test dédié vérifie
  désormais que la fixture est bien présente** (`test_the_real_zoo_fixture_reached_the_repository`).
  C'est exactement la bonne réponse : transformer un incident en test.
- **Reste à noter** : la règle `!tests/data/` ré-inclut le répertoire mais **pas les
  fichiers d'image qu'il pourrait contenir** — le commentaire du `.gitignore:32-36`
  l'explique correctement et c'est vrai. En revanche, un futur fichier de fixture avec une
  extension listée (`.mha`, `.nii.gz`) serait à nouveau avalé, avec le même symptôme.
  Le test générique manque : « tout fichier référencé par un test existe ».
- **Correction recommandée** : ajouter
  `test_every_referenced_fixture_exists` qui parcourt `tests/data/` et vérifie que chaque
  fichier attendu par la suite est présent — ou, plus simplement, `git check-ignore` sur
  `tests/data/*` dans un test.
- **Tests nécessaires** : `test_no_test_fixture_is_gitignored`.

---

#### I-10 — Absence de tests de propriété / d'invariants géométriques

- **Emplacement** : transversal.
- **Gravité** : **Mineur**
- **Constat** : la suite est presque intégralement composée de tests par l'exemple, avec
  des valeurs codées en dur. Pour un projet dont le cœur est une algèbre de transformées,
  les invariants sont nombreux et se prêtent bien à des tests de propriété
  (`hypothesis`) :
  - `compose([A, B])` appliqué à un point == `B(A(point))` — testé sur un exemple
    (`test_composition_applies_in_list_order`), jamais sur des transformées aléatoires ;
  - `to_matrix_4x4(t)` puis reconstruction == `t` sur tout point ;
  - `matrix_moving_to_fixed(t)` == `inv(to_matrix_4x4(t))` ;
  - `flatten_linear(t)` == `t` sur tout point, pour toute composition linéaire ;
  - `decompose_affine(M)` puis recomposition == `M` ;
  - un aller-retour `transform_to_elastix_initial` → `parameter_map_to_transform`
    est l'identité (c'est ce qui aurait attrapé **B-11**).
- **Correction recommandée** : ajouter `hypothesis` aux extras `dev` et écrire une poignée
  de tests de propriété sur les transformées, avec une stratégie générant des Euler /
  affines / similitudes plausibles (angles bornés, échelles dans [0,5 ; 2], translations
  dans ±300 mm). Le rapport valeur/effort est ici particulièrement élevé.
- **Tests nécessaires** : la liste ci-dessus, sous forme de propriétés.

---

### J. Configuration, variables d'environnement et déploiement

---

#### J-01 — `pydicom>=2.4` est déclaré, mais le code exige pydicom ≥ 3.0

- **Emplacement** : `pyproject.toml:40` ; `regix/io/writers.py:209` et `:326`.
- **Gravité** : **Critique**
- **Le code** :
  ```python
  ds.save_as(str(out / f"regix_{k + 1:05d}.dcm"), enforce_file_format=False)   # writers.py:209
  ds.save_as(str(p), enforce_file_format=False)                                # writers.py:326
  ```
  Le mot-clé `enforce_file_format` a été **introduit dans pydicom 3.0**. Sa signature dans
  la version installée ici le confirme :
  ```
  Dataset.save_as(self, filename, /, _Dataset__write_like_original=None, *,
                  implicit_vr=None, little_endian=None,
                  enforce_file_format: bool = False, overwrite: bool = True, **kwargs)
  ```
  En pydicom 2.x, la signature est `save_as(filename, write_like_original=True)` :
  l'appel lève **`TypeError: save_as() got an unexpected keyword argument
  'enforce_file_format'`**.
- **[CONFIRMÉ]** : pydicom 3.0.2 installé ici, l'appel fonctionne. L'incompatibilité avec
  2.x est établie par la signature de l'API, documentée en amont.
- **Détails aggravants** :
  - `ds.is_little_endian = True` et `ds.is_implicit_VR = False` (`writers.py:207-208`) sont
    des attributs **supprimés** de `Dataset` en pydicom 3.0. Les affectations ne lèvent pas
    (Python crée simplement des attributs d'instance) mais **elles ne font plus rien** :
    c'est du code mort résiduel de l'API 2.x, qui donne l'illusion de contrôler le
    transfert syntax alors que celui-ci est déterminé par `file_meta.TransferSyntaxUID` ;
  - le code mélange donc les deux API : il utilise un mot-clé exclusif à la 3.x **et** des
    attributs exclusifs à la 2.x.
- **Conséquences** : `pip install regix` sur un environnement où pydicom 2.x est déjà
  installé (très courant — de nombreux outils d'imagerie médicale y sont épinglés) satisfait
  la contrainte et produit une installation **dont toutes les sorties DICOM plantent**.
  Ni `regix doctor` (qui vérifie seulement `import pydicom`) ni la CI (qui installe la
  dernière version) ne le détectent.
- **Correction recommandée** :
  1. **porter la contrainte à `pydicom>=3.0`** — c'est la correction juste, la 3.0 étant
     sortie et stable ;
  2. supprimer `ds.is_little_endian` / `ds.is_implicit_VR` ;
  3. renforcer `regix doctor` : afficher la **version** de pydicom et signaler en rouge si
     elle est < 3.0 (aujourd'hui, la ligne affiche seulement « installed ») ;
  4. si la compatibilité 2.x est réellement souhaitée, encapsuler l'appel :
     `ds.save_as(path, **({"enforce_file_format": False} if PYDICOM_3 else {"write_like_original": False}))`.
- **Tests nécessaires** :
  - `test_dicom_writers_declare_their_minimum_pydicom_version` : comparer
    `importlib.metadata.requires("regix")` à `pydicom.__version_info__` et échouer si
    l'installation courante ne satisfait pas ce dont le code a besoin ;
  - job CI installant explicitement `pydicom==2.4.4` pour vérifier que la contrainte
    déclarée est honnête (ce job doit **échouer** aujourd'hui).

---

#### J-02 — La CI ne teste que l'installation « cœur » : quatre extras déclarés ne sont jamais installés

- **Emplacement** : `.github/workflows/ci.yml:57-60, 95` (`pip install -e ".[dev]"`) ;
  `pyproject.toml:43-61`.
- **Gravité** : **Important**
- **Constat** : les extras `features`, `totalsegmentator`, `organs`, `api` et `all` ne sont
  installés par aucun job. Or :
  - `features` contient une **dépendance git directe**
    (`anatomix @ git+https://github.com/neel-dey/anatomix.git`) dont rien ne garantit
    qu'elle se résout encore — le dépôt amont peut avoir changé de branche par défaut, de
    nom de paquet ou de fichier de build ;
  - `organs = ["regix[totalsegmentator]"]` est une **auto-référence** ; sa résolution en
    installation éditable depuis un checkout n'est pas évidente et n'est jamais vérifiée ;
  - `api` : jamais installé, donc `regix/api.py` n'est **même jamais importé** en CI —
    ce qui explique I-01, et signifie qu'une erreur de syntaxe dans ce fichier passerait
    la CI (le job `lint` la verrait, mais pas une erreur d'import) ;
  - `all` compose les quatre.
- **Conséquences** : le README documente cinq lignes d'installation
  (`README.md:83-89`) dont **quatre ne sont jamais exécutées**. Une rupture chez un
  fournisseur amont ne serait découverte que par un utilisateur.
- **Correction recommandée** : un job `install-matrix` léger, sans exécution de tests,
  faisant `pip install -e ".[X]"` puis `python -c "import regix"` pour chaque extra, avec
  `continue-on-error: true` pour `features` (la dépendance git est hors du contrôle du
  projet) et bloquant pour `api` et `organs`. Y ajouter `regix doctor`, qui rendra visible
  ce que chaque extra apporte réellement.
- **Tests nécessaires** : le job CI est la vérification. Ajouter aussi
  `test_every_documented_install_command_appears_in_ci` (parse le README, compare aux
  jobs).

---

#### J-03 — `REGIX_PSEUDONYM_SALT` est la seule variable d'environnement, et elle n'est ni documentée en détail ni validée

- **Emplacement** : `regix/logging_utils.py:29, 76` ; `README.md:393-394`.
- **Gravité** : **Mineur** *(le fond est traité en C-01)*
- **Constats de configuration** :
  1. c'est la **seule** variable d'environnement lue par le projet ; il n'existe aucun
     mécanisme général de configuration par l'environnement, alors que c'est le mode normal
     de paramétrage d'un service conteneurisé (l'API, notamment). Les besoins identifiés
     par cet audit en ajoutent au moins trois : `REGIX_DICOM_UID_ROOT` (C-05),
     `REGIX_API_ALLOWED_ROOTS` et `REGIX_API_TOKEN` (C-03) ;
  2. sa valeur n'est **jamais validée** : un sel vide (`REGIX_PSEUDONYM_SALT=""`) est traité
     comme une chaîne valide (`os.environ.get(..., "regix")` ne retourne le défaut que si la
     clé est absente, pas si elle est vide) — le sel devient alors littéralement `""` ;
  3. elle n'apparaît **pas** dans `regix doctor`, alors que c'est exactement le genre
     d'information qu'un « what is installed, what is missing, and the impact » devrait
     donner ;
  4. elle n'est **pas** enregistrée dans le manifeste (même pas son empreinte), donc rien
     ne permet de savoir *a posteriori* si deux runs ont utilisé le même sel — ce qui est
     nécessaire pour corréler leurs pseudonymes.
- **Correction recommandée** : introduire un petit module `regix/env.py` centralisant la
  lecture et la validation des variables ; ajouter une ligne « pseudonymisation » à
  `regix doctor` indiquant `sel : défini / par défaut (FAIBLE)` ; consigner dans le
  manifeste le **hachage tronqué du sel** (jamais le sel), qui suffit à établir la
  concordance entre deux runs sans rien révéler.
- **Tests nécessaires** :
  - `test_an_empty_salt_is_refused` ;
  - `test_doctor_reports_the_pseudonymisation_state` ;
  - `test_manifest_records_a_salt_fingerprint_not_the_salt`.

---

#### J-04 — Aucun verrouillage de version : `pip install -e .` d'aujourd'hui n'est pas celui de demain

- **Emplacement** : `pyproject.toml:28-41` ; `.github/workflows/ci.yml:29-30`
  (`cache-dependency-path: pyproject.toml`).
- **Gravité** : **Important**
- **Constat** : toutes les dépendances sont des bornes basses ouvertes
  (`itk-elastix>=0.20`, `SimpleITK>=2.2`, `numpy>=1.24`…). La seule exception est `ruff`,
  bornée haut — et le commentaire qui l'explique (`pyproject.toml:57-59`) est **excellent et
  exact** : « it is the one tool whose new releases add rules … so an unpinned `ruff>=x`
  turns CI red on a day nobody touched the code ».
- **Le problème** : ce raisonnement est parfaitement juste et s'applique **avec bien plus de
  force** à `itk-elastix`. Ruff qui change fait rougir la CI — c'est visible et sans
  conséquence sur les résultats. `itk-elastix` qui change de version modifie **les valeurs
  numériques produites par le recalage**, silencieusement. Le projet a raisonné sur le
  risque le moins grave et pas sur le plus grave.
- **Conséquences** :
  - **reproductibilité** : le manifeste enregistre la version d'ITK utilisée (bien), mais
    rien ne permet de la **restaurer**. « Ce que vous relisez six mois plus tard » indique
    donc une version qu'on ne sait pas réinstaller ;
  - **CI non déterministe** : un job vert aujourd'hui peut être rouge demain sans commit,
    par montée d'une dépendance. La mise en cache (`cache: pip`,
    `cache-dependency-path: pyproject.toml`) **masque** partiellement le phénomène : le
    cache n'étant invalidé que par un changement de `pyproject.toml`, une dépendance
    nouvellement publiée peut n'apparaître que des semaines plus tard, rendant l'échec
    difficile à corréler à sa cause ;
  - **numpy 2** : `numpy>=1.24` autorise numpy 2.x, dont les changements de promotion de
    types (NEP 50) modifient le comportement de plusieurs expressions du projet
    (par exemple `np.float32 + python_float`). L'environnement d'audit utilise numpy 2.3.5
    et la suite passe — mais rien ne teste numpy 1.x, pourtant autorisé.
- **Correction recommandée** :
  1. **borner haut les dépendances numériquement significatives** :
     `itk-elastix>=0.20,<0.22`, `SimpleITK>=2.2,<3`, `numpy>=1.24,<3` ;
  2. ajouter un `requirements-lock.txt` généré par `pip-compile` / `uv lock`, utilisé par
     **un** job CI « reproductible » à côté du job « dernières versions », pour distinguer
     une régression du projet d'une rupture amont ;
  3. tester explicitement les deux bornes de numpy dans la matrice.
- **Tests nécessaires** : deux jobs CI (`resolved` et `locked`) ; la divergence entre les
  deux est le signal recherché.

---

#### J-05 — La CI n'a ni build de paquet, ni vérification d'installation depuis une distribution

- **Emplacement** : `.github/workflows/ci.yml` (3 jobs : `lint`, `test`, `cli`).
- **Gravité** : **Mineur**
- **Constat** : tous les jobs utilisent `pip install -e .` (installation éditable, depuis le
  répertoire source). Rien ne construit une roue, ni ne vérifie que le paquet installé
  contient les données nécessaires.
- **Le risque concret** : `[tool.setuptools.package-data]` déclare
  `presets/*.yaml`. En mode éditable, les presets sont trouvés parce que
  `PRESET_DIR = Path(__file__).parent / "presets"` pointe vers le source. Dans une roue mal
  construite, `available_presets()` renverrait une **liste vide** et
  `load_preset("base")` lèverait `FileNotFoundError` — sans qu'aucun job ne s'en aperçoive.
  Le job `cli` valide bien les presets, mais toujours en mode éditable.
  (Rappel : `qc/templates/*.html` est déjà déclaré à tort, E-12 — signe que cette
  déclaration n'est vérifiée par rien.)
- **Correction recommandée** : un job `package` faisant
  `python -m build`, `pip install dist/*.whl` **dans un environnement vierge, hors du
  répertoire du dépôt**, puis `regix presets` et `python -c "from regix.config import
  available_presets; assert len(available_presets()) == 8"`. Ajouter `twine check dist/*`.
- **Tests nécessaires** : le job CI ; plus
  `test_preset_directory_is_packaged` (vérifie `PRESET_DIR.exists()` et le décompte).

---

#### J-06 — Rien ne vérifie la cohérence entre la documentation et le code

- **Emplacement** : transversal — c'est le défaut de processus qui sous-tend la majorité de
  la section A.
- **Gravité** : **Important**
- **Constat** : le README est très riche en affirmations vérifiables mécaniquement, et
  **aucune** ne l'est. Cet audit en a trouvé onze fausses (A-01 à A-06, A-07, A-12, A-13,
  A-14, A-17) — toutes détectables par un test de quelques lignes :

  | Affirmation | Test mécanique possible |
  |---|---|
  | décomptes de tests (A-07) | comparer au `--collect-only` |
  | options CLI citées (A-05, A-13) | parcourir les docstrings/descriptions, extraire `--xxx`, vérifier contre `app` |
  | tableau des presets (A-02) | parser le tableau, comparer aux YAML chargés |
  | badge de couverture (I-06) | comparer à la mesure |
  | commande de rejeu elastix (A-03) | vérifier l'existence des fichiers cités |
  | `regix presets` avec commentaires (A-01) | compter les `#` |
  | champs de profil consommés (A-06) | test réflexif |
  | versions au manifeste (A-12) | comparer aux dépendances déclarées |
- **Correction recommandée** : un fichier `tests/test_documentation.py` regroupant ces
  vérifications. C'est le test à écrire **en premier** : il transforme une classe entière de
  défauts en échecs de CI, et il empêche leur réapparition. Coût : une centaine de lignes.
- **Tests nécessaires** : le fichier lui-même.

---

#### J-07 — Le workflow CI n'a pas de `timeout-minutes` et un `continue-on-error` masque le format

- **Emplacement** : `.github/workflows/ci.yml:21, 41, 85` (aucun `timeout-minutes`) ;
  `:39` (`continue-on-error: true` sur `ruff format --check`).
- **Gravité** : **Cosmétique**
- **Constats** :
  1. aucun job n'a de `timeout-minutes`. Le job `test` exécute de vraies registrations
     elastix ; un optimiseur qui ne converge pas ou une pathologie de sampling immobilise un
     runner jusqu'au délai par défaut de GitHub (**6 heures**). Avec
     `cancel-in-progress: true` cela reste borné en pratique, mais le coût est réel ;
  2. `ruff format --check` en `continue-on-error: true` avec le commentaire « formatting is
     advisory, not a merge blocker ». C'est un choix défendable — mais l'historique montre
     un commit `03f4917 "style: apply ruff format across the repository"`, donc le projet
     *a* adopté le format. Un contrôle qui ne bloque jamais et que personne ne lit dans les
     journaux ne sert à rien : soit il devient bloquant, soit il est retiré ;
  3. `if-no-files-found: warn` sur l'upload du rapport QC : si l'étape de registration
     échoue, l'artefact manque et le job réussit quand même — cohérent avec `if: always()`,
     mais le rapport est précisément ce qu'on veut examiner après un échec. `error` serait
     plus utile, conjugué à `if: always()`.
- **Correction recommandée** : `timeout-minutes: 30` sur `test`, `10` sur `lint` et `cli` ;
  rendre `ruff format` bloquant ; passer l'upload en `if-no-files-found: error` (le job
  `cli` ayant déjà échoué, cela n'ajoute pas de faux négatif).
- **Tests nécessaires** : la CI est la vérification.

---

#### J-08 — Le CLI ne propose ni `--version` global, ni sortie machine pour `register`

- **Emplacement** : `regix/cli.py:31-36`, `:87-90`, `:240-366`.
- **Gravité** : **Mineur**
- **Constats** :
  1. `regix version` existe (sous-commande), mais `regix --version` — la convention
     universelle, et ce que tout script d'intégration essaie en premier — n'existe pas.
     Typer le fournit en trois lignes via un callback `--version` avec `is_eager=True` ;
  2. `regix inspect --json` existe (bien), mais `regix register` n'a **aucune** sortie
     machine : le statut, les métriques et les chemins ne sont disponibles qu'en texte
     formaté par `rich`. Un script d'orchestration doit soit relire
     `run_manifest.json` (possible mais non documenté comme contrat), soit parser la sortie
     colorée. Ajouter `--json` à `register` et `batch` est peu coûteux et directement utile
     pour le public visé ;
  3. les codes de sortie ne sont documentés nulle part : `0` = OK, `1` = erreur
     d'invocation ou moteur absent, `2` = FAIL du QC. C'est un contrat d'interface qui
     mérite d'être écrit dans le README et testé.
- **Correction recommandée** : ajouter le callback `--version`, un `--json` sur `register`
  et `batch`, et une section « Codes de sortie » au README.
- **Tests nécessaires** :
  - `test_version_flag_and_subcommand_agree` ;
  - `test_register_json_output_is_valid_and_stable` ;
  - `test_exit_codes_are_documented_and_respected` (paramétré sur les trois cas).

---

### K. Documentation et expérience développeur

---

#### K-01 — Aucun fichier `CONTRIBUTING`, `CHANGELOG`, ni documentation d'architecture

- **Emplacement** : racine du dépôt (`LICENSE`, `README.md`, `pyproject.toml` seulement).
- **Gravité** : **Mineur**
- **Constat** : le README fait 28 Ko et couvre l'utilisation. Manquent :
  - **`CONTRIBUTING.md`** : comment lancer les tests, quelles conventions
    (le projet en a de fortes : justifier les choix non évidents en commentaire à l'endroit
    concerné — c'est une **excellente** convention, et elle n'est écrite nulle part) ;
  - **`CHANGELOG.md`** : la version est figée à `0.1.0` dans `pyproject.toml` et
    `__init__.py` alors que l'historique montre des changements de comportement majeurs
    (abandon de la normalisation, `min_abs_final_metric`, refonte de la segmentation).
    Un utilisateur ne peut pas savoir ce qui a changé ;
  - une **note d'architecture** : les docstrings de module sont remarquables individuellement,
    mais rien ne décrit la vue d'ensemble (le diagramme mermaid du README décrit le *flux
    de données*, pas la structure du code, les invariants transverses ou les conventions
    de coordonnées).
- **Le point le plus dommageable** : les **conventions transverses** — « elastix/ITK va du
  fixe vers le mobile », « la sortie est toujours sur la grille fixe d'origine »,
  « les masques ne sont jamais interpolés autrement qu'au plus proche voisin » — sont
  rappelées en tête de plusieurs modules (`transforms.py`, `geometry.py`, `pipeline.py`),
  ce qui est bien, mais un nouvel arrivant doit les découvrir module par module.
- **Correction recommandée** : un `CONTRIBUTING.md` court (tests, ruff, la convention de
  justification en commentaire), un `CHANGELOG.md` au format *Keep a Changelog*, et un
  `docs/ARCHITECTURE.md` d'une page listant les invariants transverses et le rôle de chaque
  paquet.
- **Tests nécessaires** : `test_version_is_consistent_across_pyproject_and_package`
  (aujourd'hui `0.1.0` est écrit à deux endroits sans contrôle).

---

#### K-02 — Le README de 28 Ko mélange trois publics sans séparation

- **Emplacement** : `README.md` (536 lignes).
- **Gravité** : **Mineur**
- **Constat** : le document sert simultanément de page d'accueil, de manuel utilisateur, de
  note de conception (« Technical notes established by measurement ») et de journal de
  décisions (« Why elastix and not SimpleElastix », « Native intensities reach elastix »).
  Ces sections sont **intéressantes et bien écrites** — c'est d'ailleurs ce qui rend le
  projet lisible — mais elles noient les informations opérationnelles : les limitations,
  qui sont importantes, arrivent à la ligne 487.
- **Effet de bord observé** : plus le README est long et affirmatif, plus le coût de le
  maintenir en accord avec le code est élevé — et c'est exactement ce qui s'est produit
  (section A).
- **Correction recommandée** : garder dans le README l'essentiel opérationnel (installation,
  usage, presets, limitations, sécurité), et déplacer les notes de conception dans
  `docs/DESIGN.md` — en conservant le renvoi. Ce qui est *vérifiable mécaniquement* doit
  rester dans le README et être testé (J-06) ; ce qui relève du raisonnement peut vivre à
  côté.
- **Tests nécessaires** : sans objet.

---

#### K-03 — `console.print` de `rich` sur du YAML arbitraire : le balisage peut être interprété

- **Emplacement** : `regix/cli.py:173` (`presets NAME`), `:360` (`register --dry-run`).
- **Gravité** : **Mineur**
- **Le mécanisme** : `console.print(cfg.to_yaml())` — `rich` interprète par défaut la syntaxe
  `[tag]` comme du balisage. Le YAML produit par `yaml.safe_dump` utilise le style bloc pour
  les listes, donc les crochets sont rares — mais ils **apparaissent** dès qu'une valeur de
  chaîne en contient. Un chemin Windows contenant `[`, une description d'organe, un libellé
  de stage personnalisé (`StageConfig.label`) : `rich` lèverait
  `MarkupError` ou, plus insidieusement, **avalerait silencieusement** le fragment.
  De plus, `rich` applique une coloration syntaxique heuristique et **replie les lignes
  longues** sur la largeur du terminal, ce qui peut produire un YAML **non ré-analysable**
  si l'utilisateur redirige la sortie de `--dry-run` vers un fichier.
- **[À VÉRIFIER]** : je n'ai pas reproduit de corruption sur les presets livrés — ils ne
  contiennent pas de crochets. Le risque est identifié par lecture de l'API de `rich`.
- **Conséquences** : `regix register … --dry-run > my_config.yaml` est un usage naturel
  (le README présente `--dry-run` comme « print the effective configuration ») et peut
  produire un fichier tronqué ou reformaté.
- **Correction recommandée** : `console.print(yaml_text, markup=False, highlight=False,
  soft_wrap=True)`, ou simplement `print(...)` / `sys.stdout.write(...)` pour toute sortie
  destinée à être redirigée. Idéalement, `rich.syntax.Syntax(yaml_text, "yaml")` quand la
  sortie est un terminal, et du texte brut sinon (`console.is_terminal`).
- **Tests nécessaires** :
  - `test_dry_run_output_is_reparseable_yaml` : capturer la sortie, `yaml.safe_load`,
    comparer à la configuration ;
  - `test_dry_run_survives_a_label_containing_brackets`.

---

#### K-04 — `regix doctor` : sorties incohérentes et import coûteux

- **Emplacement** : `regix/cli.py:93-162`.
- **Gravité** : **Mineur**
- **Constats** :
  1. **incohérence de présentation** : `torch`, `monai`, `numpy` affichent une version ;
     `anatomix`, `TotalSegmentator`, `pydicom` affichent « installed » sans version — alors
     que c'est justement la version qui importe (J-01 pour pydicom). `monai` utilise encore
     une troisième forme : `f"{report['monai'] or '[yellow]missing[/]'}"` ;
  2. **coût** : `import totalsegmentator` charge torch, nnU-Net et leurs dépendances —
     plusieurs secondes et plusieurs centaines de Mo de RSS, pour afficher une ligne.
     `importlib.util.find_spec("totalsegmentator")` répond à la même question
     instantanément et sans effet de bord ;
  3. **robustesse** : les `try/except ImportError` ne captent que `ImportError`. Une
     installation cassée qui lève `OSError` (DLL CUDA manquante — cas fréquent) ou
     `RuntimeError` fait **planter `regix doctor`**, c'est-à-dire précisément l'outil censé
     diagnostiquer ce genre de situation ;
  4. **manques** : ni la version de `itk-elastix` (A-12), ni l'état de
     `REGIX_PSEUDONYM_SALT` (J-03), ni l'espace disque disponible, ni la mémoire — trois
     informations que « what is installed, what is missing, **and the impact** » devrait
     couvrir pour un outil qui alloue plusieurs Go (section G).
- **Correction recommandée** : une table homogène `(composant, version, état, conséquence)`
  alimentée par `importlib.metadata.version` + `importlib.util.find_spec` ;
  `except Exception` avec affichage du type d'erreur ; ajout des quatre informations
  manquantes.
- **Tests nécessaires** :
  - `test_doctor_survives_a_broken_optional_dependency` (monkeypatch levant `OSError`) ;
  - `test_doctor_reports_a_version_for_every_installed_component` ;
  - `test_doctor_does_not_import_torch` (mesure de `sys.modules`).

---

#### K-05 — `engine_available()` annonce la version d'ITK sous le libellé « itk-elastix »

- **Emplacement** : `regix/registration/itk_bridge.py:49-54` ; `regix/cli.py:105-111` ;
  `regix/api.py:145-148`.
- **Gravité** : **Cosmétique**
- **Constat** : `return True, f"itk {itk.Version.GetITKVersion()}"` — c'est la version de la
  bibliothèque **ITK**, affichée dans la ligne intitulée « itk-elastix (engine) » de
  `regix doctor` et renvoyée par `GET /health` sous la clé `engine`. Un utilisateur qui
  relève « 5.4.6 » pour tracer sa version d'elastix relève autre chose.
- **Correction recommandée** : renvoyer
  `f"itk-elastix {importlib.metadata.version('itk-elastix')} (ITK {itk.Version.GetITKVersion()})"`,
  avec un repli propre si la métadonnée est absente. Recoupe A-12 et K-04.
- **Tests nécessaires** : `test_engine_detail_names_the_elastix_binding_version`.

---

#### K-06 — Les messages d'erreur de `load_preset` sont trompeurs pour un chemin de fichier

- **Emplacement** : `regix/config.py:505-520`.
- **Gravité** : **Cosmétique**
- **Le code** :
  ```python
  candidate = Path(name_or_path)
  if candidate.suffix in (".yaml", ".yml") and candidate.exists():
      path = candidate
  else:
      path = PRESET_DIR / f"{name_or_path}.yaml"
      if not path.exists():
          raise FileNotFoundError(f"unknown preset '{name_or_path}'. Available: …")
  ```
  Un utilisateur qui passe `--config /chemin/typo.yaml` (fichier inexistant) obtient
  « **unknown preset** '/chemin/typo.yaml'. Available: base, ct_cbct_igrt, … » — un message
  qui laisse croire à une erreur de nom de preset alors que c'est un chemin invalide.
  Le projet tente même de construire `PRESET_DIR / "/chemin/typo.yaml.yaml"`.
- **Correction recommandée** : distinguer les deux intentions — si l'argument ressemble à un
  chemin (contient un séparateur ou une extension `.yaml`/`.yml`), lever
  `FileNotFoundError(f"fichier de configuration introuvable : {candidate}")` ; sinon,
  le message actuel. Ajouter `difflib.get_close_matches` sur les noms de presets pour
  suggérer l'orthographe.
- **Tests nécessaires** :
  - `test_a_missing_config_file_is_reported_as_a_missing_file` ;
  - `test_a_misspelled_preset_name_suggests_the_closest_match`.

---

#### K-07 — `_build_from_raw` mute le dictionnaire de l'appelant

- **Emplacement** : `regix/config.py:523-524` (`raw.pop("extends", None)`).
- **Gravité** : **Cosmétique**
- **[CONFIRMÉ]** :
  ```
  raw = {"extends": "base", "name": "x"} ; _build_from_raw(raw, Path("."))
  raw  ->  {"name": "x"}        # la clé 'extends' a disparu chez l'appelant
  ```
- **Constat** : sans conséquence aujourd'hui (les seuls appelants passent un dictionnaire
  fraîchement produit par `yaml.safe_load`), mais `_build_from_raw` est appelée
  récursivement et pourrait, dans une évolution, recevoir un dictionnaire réutilisé.
  Un second appel sur le même dictionnaire produirait un résultat **différent** du premier.
- **Correction recommandée** : `raw = dict(raw)` en tête de fonction.
- **Tests nécessaires** : `test_build_from_raw_does_not_mutate_its_input`.

---

#### K-08 — Le message de repli de `apply_intensity_prep` sur les voxels non finis est inexact

- **Emplacement** : `regix/preprocess/intensity.py:194-199`.
- **Gravité** : **Cosmétique**
- **Le code** :
  ```python
  log.warning("%d non-finite voxels replaced with the finite minimum", n_bad)
  clean = np.nan_to_num(..., nan=<min>, posinf=<max>, neginf=<min>)
  ```
  Les `+inf` sont remplacés par le **maximum** fini, pas par le minimum. Le message
  généralise à tort.
- **Défaut secondaire** : `np.isfinite(arr)` est évalué **quatre fois** sur le volume
  complet dans ces cinq lignes (une pour le comptage, trois dans les arguments de
  `nan_to_num`), soit quatre tableaux booléens de la taille du volume. Sur le volume de
  référence, 840 Mo d'allocations inutiles — un cas ponctuel de G-01.
- **Correction recommandée** : calculer `finite_mask = np.isfinite(arr)` une fois, en
  dériver `lo`/`hi`, et corriger le message
  (« %d voxels non finis remplacés (NaN et −inf → %.3g, +inf → %.3g) »).
- **Tests nécessaires** : `test_non_finite_voxels_are_replaced_by_the_correct_bound`
  (volume contenant `nan`, `+inf` et `-inf`).

---

#### K-09 — Duplication du même avertissement de recouvrement de champ de vue

- **Emplacement** : `regix/organs/roi.py:200-206` (`log.warning`) et
  `regix/pipeline.py:248-251` (`manifest.warn`, qui journalise aussi).
- **Gravité** : **Cosmétique**
- **Constat** : la même condition (`min(fov_fixed, fov_moving) < 0.25`) est évaluée et
  journalisée deux fois, avec deux formulations différentes. L'utilisateur voit deux
  avertissements consécutifs pour un seul problème ; le seuil `0.25` est codé en dur aux
  deux endroits, sans constante partagée.
- **Correction recommandée** : conserver uniquement le `manifest.warn` du pipeline (qui
  atteint le rapport) ; `roi_overlap_report` se contente de renvoyer la mesure. Extraire le
  seuil en constante de module, ou mieux, en champ de configuration
  (`qc.gates.min_fov_overlap`), puisqu'il s'agit conceptuellement d'un critère
  d'acceptabilité — il n'en est aujourd'hui pas un : **aucune porte QC ne l'évalue**, le
  recouvrement de champ de vue produit seulement un avertissement.
- **Tests nécessaires** :
  - `test_low_fov_overlap_is_reported_once` ;
  - `test_fov_overlap_can_be_made_a_blocking_gate`.

---

#### K-10 — Le calcul de recouvrement du champ de vue utilise des boîtes alignées aux axes

- **Emplacement** : `regix/organs/roi.py:172-219` (`roi_overlap_report`, `_physical_box`).
- **Gravité** : **Mineur**
- **Le mécanisme** : `_physical_box` calcule bien les 8 coins en tenant compte de la
  direction (le docstring le souligne : « all 8 corners, **direction included** ») — mais
  renvoie ensuite `pts.min(axis=0), pts.max(axis=0)`, c'est-à-dire la **boîte englobante
  alignée aux axes** (AABB). Le recouvrement est ensuite l'intersection de deux AABB.
- **Pourquoi c'est faux pour une acquisition oblique** : l'AABB d'un volume incliné est
  strictement plus grande que le volume lui-même, et l'intersection de deux AABB est
  strictement plus grande que l'intersection des deux volumes. Le recouvrement est donc
  **surestimé**, et d'autant plus que l'inclinaison est forte.
- **Conséquences** : le recouvrement de champ de vue est présenté comme « a diagnosis »
  (docstring du pipeline) et comme « the primary cause of silent failure ». Un
  recouvrement réel de 20 % peut être rapporté à 60 % sur une paire oblique, et
  l'avertissement à 25 % ne se déclenche pas — précisément dans le cas où il serait le plus
  utile, puisque l'obliquité est le contexte que le projet met en avant.
- **Correction recommandée** : deux options, par coût croissant :
  1. **échantillonnage** — tirer N points (10⁴ suffisent) uniformément dans la grille fixe,
     les transformer en coordonnées physiques puis en indices continus de la grille mobile,
     et compter la fraction tombant à l'intérieur. Exact à la précision de Monte-Carlo,
     ~10 lignes, et applicable dans les deux sens ;
  2. intersection exacte de deux parallélépipèdes (algorithme de Sutherland–Hodgman en 3D) —
     plus juste, nettement plus long à écrire.
  Consigner la méthode utilisée dans le manifeste.
- **Tests nécessaires** :
  - `test_fov_overlap_of_two_identical_volumes_is_one` ;
  - `test_fov_overlap_is_not_overestimated_for_an_oblique_pair` — deux volumes inclinés de
    45° dont l'intersection réelle est connue analytiquement.

---

### L. Dette technique et risques futurs

---

#### L-01 — Version figée à `0.1.0` malgré des changements de comportement majeurs

- **Emplacement** : `pyproject.toml:7` ; `regix/__init__.py:21`.
- **Gravité** : **Mineur**
- **Constat** : la version est dupliquée dans deux fichiers, sans contrôle de cohérence, et
  n'a jamais bougé. L'historique montre pourtant des changements qui modifient les
  **résultats numériques** : abandon de la normalisation min-max (`7d6be2b`), introduction
  de `min_abs_final_metric`, refonte de la segmentation, types de pixels internes
  (`86b6f50`). Deux runs étiquetés `regix 0.1.0` dans leur manifeste peuvent produire des
  transformées différentes sur les mêmes données.
- **Conséquences** : la traçabilité du manifeste — argument central — est illusoire tant que
  la version ne discrimine rien. C'est le risque le plus structurant à moyen terme pour un
  logiciel dont l'objectif affiché est de pouvoir être relu six mois plus tard.
- **Correction recommandée** : source unique de vérité
  (`version = {attr = "regix.__version__"}` dans `pyproject.toml`), incrémentation à chaque
  changement de comportement numérique, `CHANGELOG.md` distinguant explicitement les
  changements « numériques » des autres, et — idéalement — inscription du **SHA de commit**
  dans le manifeste quand il est disponible.
- **Tests nécessaires** : `test_version_is_declared_once` ;
  `test_manifest_records_the_regix_version_and_commit`.

---

#### L-02 — Le doublon de format `.txt` est une bombe à retardement

- **Emplacement** : voir **D-05**, **B-03**, **A-14**.
- **Gravité** : **Important**
- **Le risque futur** : trois défauts distincts de cet audit ont la même racine — deux
  formats incompatibles partageant une extension, écrits côte à côte par le même outil.
  Chaque nouveau point d'entrée acceptant une transformée (une commande `regix invert`, un
  champ d'API, un chargement de chaîne partielle) reproduira le défaut, car rien dans le
  code ne canalise cette décision.
- **Correction recommandée** : au-delà de `load_any_transform` (D-05), **renommer les
  sorties** pour lever l'ambiguïté à la source :
  `transform/final_transform.itk.txt` (ou `.tfm` uniquement) contre
  `elastix/…/TransformParameters.0.txt`. Un changement de nom de fichier est une rupture
  mineure, à faire tant que le projet est en 0.x.
- **Tests nécessaires** : le test paramétré de D-05.

---

#### L-03 — `e2e_out/` : 40 Mo de sorties de run réel dans l'arbre de travail

- **Emplacement** : `e2e_out/` (non versionné, correctement couvert par `.gitignore:21`).
- **Gravité** : **Mineur**
- **Constat** : le répertoire contient `moving_registered.nii.gz` (34 Mo),
  `report.html` (3,9 Mo), les journaux elastix, et un `run_manifest.json` **contenant le
  chemin `C:\Users\thibault.escobar\Desktop\dataregix\Fixed`** (C-02). `git status` est
  propre : rien n'est versionné, et le `.gitignore` fait correctement son travail —
  l'en-tête « PATIENT DATA AND RUN OUTPUTS: NEVER COMMIT » et la largeur délibérée des
  règles sont un point fort du projet (§4).
- **Le risque résiduel** : ces fichiers proviennent de `Desktop\dataregix\`, c'est-à-dire de
  données réelles ou semi-réelles hors du dépôt. Ils sont susceptibles d'être joints à un
  rapport de bug, copiés dans une archive, ou capturés par une sauvegarde. Le
  `run_manifest.json` et les `elastix.log` contiennent des chemins.
- **Correction recommandée** : sans objet pour le dépôt (rien à corriger côté versionnement) ;
  côté pratique de travail, purger `e2e_out/` après usage, et — surtout — corriger C-02 pour
  que ces artefacts cessent de contenir des chemins.
- **Tests nécessaires** : `test_gitignore_covers_every_artifact_a_run_produces` — dériver la
  liste depuis `expected_outputs()` (D-06) et vérifier avec `git check-ignore`.

---

#### L-04 — La logique de repli silencieux est répandue et va à l'encontre du principe affiché

- **Emplacement** : transversal — `pipeline.py:523-529, 613-616, 763-764, 807-808, 869-870,
  924-925, 970-971, 1014-1015` ; `initialize.py:291-296, 363-365` ;
  `roi.py:57-59, 132-133, 143-149` ; `metrics.py:190-191, 257-259, 283-286` ;
  `segmenter.py:265-266` ; `warp.py:98-102`.
- **Gravité** : **Important**
- **Constat** : le principe de conception n°3 du README est sans ambiguïté :
  > « **A failure is labelled, never hidden.** … It is not replaced by a silent fallback,
  > because a degraded result with no signal is more dangerous than no result at all. »
  Ce principe est **rigoureusement appliqué aux portes QC** — qui sont excellentes (§4).
  Il l'est beaucoup moins ailleurs : on dénombre **une vingtaine** de `except Exception`
  suivis d'un repli. La qualité du signalement varie fortement :

  | Niveau de signalement | Occurrences | Atteint le rapport ? |
  |---|---|---|
  | `manifest.warn` | ~10 | **oui** — bon comportement |
  | `log.warning` seul | ~8 | **non** |
  | `log.debug` / `log.info` seul | ~4 | **non** |
  | `pass` silencieux | 2 (`segmenter.py:266`, `anatomix.py:170-173`) | non |

- **Les cas les plus problématiques** ont été détaillés individuellement : B-07
  (initialisation remplacée), F-10 (masque élargi au corps entier), B-12 (organe hors champ
  rétrogradé en WARN), E-06 (métriques élargies au volume entier sans trace).
  Leur point commun : le run se termine en PASS ou WARN, et **le rapport — le seul document
  que le relecteur ouvre — ne mentionne rien**.
- **Risque futur** : chaque nouveau repli ajouté suivra le motif local plutôt que le
  principe, parce que rien ne l'y contraint.
- **Correction recommandée** — une règle structurelle plutôt qu'une série de correctifs :
  1. **interdire `log.warning` seul dans le pipeline** : toute dégradation d'un
     comportement demandé passe par `manifest.warn`, qui atteint le manifeste **et** le
     rapport. C'est une règle vérifiable par un test réflexif ou par une règle ruff
     personnalisée ;
  2. distinguer dans le rapport les avertissements **informatifs** des
     **dégradations de ce qui a été demandé** (`manifest.degraded(...)`), et faire passer
     le statut à WARN au minimum dès qu'une dégradation est enregistrée ;
  3. ajouter au rapport une section « Ce qui a été demandé et n'a pas eu lieu », qui rendrait
     visibles d'un coup B-07, F-10, A-17 et les segmentations manquées.
- **Tests nécessaires** :
  - `test_every_degradation_reaches_the_manifest` : test paramétré simulant chaque cause de
    repli (segmenteur absent, organe manquant, landmarks illisibles, transformix en échec)
    et vérifiant la présence d'une entrée dans `manifest.warnings` **et** dans
    `report.html` ;
  - `test_a_degraded_run_is_never_reported_as_pass`.

---

#### L-05 — Le couplage `pipeline` ↔ QC empêche de recalculer un QC a posteriori

- **Emplacement** : `regix/pipeline.py:704-872` (`_quality_control`, méthode privée à
  14 paramètres).
- **Gravité** : **Mineur**
- **Constat** : tout le QC est enfermé dans une méthode privée du pipeline, qui prend en
  entrée des objets vivants (`Volume`, `AppliedTransform`, `RegistrationOutcome`).
  Il n'existe aucun moyen de :
  - recalculer un QC sur un run archivé (transformée + volumes sur disque) après avoir
    ajusté un seuil ;
  - régénérer un `report.html` depuis un `run_manifest.json` ;
  - comparer deux runs.
  Or c'est un besoin naturel : les portes d'acceptation sont *par nature* ce qu'un site
  ajuste après avoir vu ses premiers résultats, et le README encourage explicitement à
  « set the acceptance gates to the tolerance your application actually requires ».
  Aujourd'hui, ajuster un seuil impose de **relancer le recalage complet**.
- **Correction recommandée** : extraire une commande `regix qc OUT_DIR [--set qc.gates…]`
  qui relit `config_effective.yaml`, la transformée et les volumes cités par le manifeste,
  recalcule les métriques et régénère rapport et manifeste. Cela nécessite (a) que le
  manifeste enregistre les chemins d'entrée — ce qu'il fait déjà, quoique en clair (C-02) —
  et (b) que `_quality_control` devienne une fonction libre prenant une structure explicite
  (recoupe D-01 et H-08).
- **Tests nécessaires** :
  - `test_qc_can_be_recomputed_from_a_finished_run` ;
  - `test_recomputed_qc_matches_the_original_when_thresholds_are_unchanged`.

---

#### L-06 — Le projet n'a qu'un seul backend de segmentation par choix, mais l'abstraction n'est pas prête pour un second

- **Emplacement** : `regix/organs/segmenter.py:108-124` (`OrganSegmenter`),
  `:358-377` (`build_segmenter`).
- **Gravité** : **Cosmétique**
- **Constat** : le choix d'un seul backend automatique est **explicitement justifié**, et la
  justification est bonne (`segmenter.py:12-16` : les masques sont des a priori, pas des
  livrables ; un second segmenteur ajouterait des nomenclatures et des modes de défaillance
  sans améliorer la précision du recalage). Elle est reprise dans les limitations du README.
  **Ce n'est donc pas un défaut.**
- **La dette réside ailleurs** : la classe de base `OrganSegmenter` n'expose que
  `segment()` et `_cache_key()`, et `build_segmenter` est un `if/elif` sur une énumération
  fermée avec un `raise` final. Si le choix devait être révisé (segmentation MR, modèle
  local validé sur site), il faudrait toucher `OrganConfig`, `build_segmenter`, la
  résolution de nomenclature et le cache — et B-06 montre que les paramètres spécifiques
  au backend (`task`, `fast`) ne remontent déjà pas jusqu'à la configuration.
- **Correction recommandée** : rien d'urgent. Lors de la correction de B-06, en profiter
  pour faire porter les options spécifiques par un sous-modèle
  (`OrganConfig.backend_options: dict[str, Any]`) validé par le backend lui-même, plutôt que
  d'ajouter des champs `ts_*` au modèle générique.
- **Tests nécessaires** : sans objet à ce stade.

---

#### L-07 — La reproductibilité repose sur une propriété du build d'elastix, non maîtrisée

- **Emplacement** : `README.md:495-500` ; `regix/config.py:407` (`seed`).
- **Gravité** : **Mineur** *(risque assumé, honnêtement documenté)*
- **Constat** : le déterminisme des échantillonneurs aléatoires d'elastix est une propriété
  du build, pas une garantie de Regix — le README le dit explicitement, et **c'est le bon
  ton** (§4). Le risque est que cette propriété disparaisse à une montée de version
  d'`itk-elastix`, silencieusement : rien ne la surveille, et J-04 (absence de borne haute)
  rend la montée automatique.
- **Correction recommandée** : le test de déterminisme d'A-16 (deux processus, 400 points)
  exécuté en CI transforme cette propriété non maîtrisée en propriété **surveillée**.
  Combiné à la borne haute de J-04, cela suffit à contenir le risque.
- **Tests nécessaires** : celui d'A-16, exécuté en CI (`@pytest.mark.slow`, job dédié).

---

#### L-08 — Le passage à `keep_intermediate` fonctionnel entrera en conflit avec la rejouabilité

- **Emplacement** : interaction **E-11** × **A-03**.
- **Gravité** : **Cosmétique** *(risque de conception à anticiper)*
- **Constat** : E-11 recommande de faire fonctionner `keep_intermediate` (aujourd'hui
  inopérant), donc de supprimer des fichiers intermédiaires. A-03 recommande d'écrire
  **davantage** de fichiers (les images remises à elastix) pour tenir la promesse de
  rejouabilité. Les deux corrections tirent en sens opposés.
- **Résolution recommandée** : distinguer explicitement deux notions dans la configuration —
  `runtime.keep_intermediate` (journaux d'itération elastix, fichiers de travail : par
  défaut faux) et `output.write_elastix_inputs` (les images, nécessaires au rejeu : par
  défaut faux, mais documenté comme le prérequis de la rejouabilité). Et **ne jamais**
  supprimer `TransformParameters.0.txt` ni `parameters.txt`, qui sont des livrables copiés
  dans `transform/`.
- **Tests nécessaires** : `test_keep_intermediate_and_replayability_are_independent`.

---

#### L-09 — Aucune stratégie de gestion des volumes trop grands pour la mémoire

- **Emplacement** : transversal — voir G-01, G-02, G-05, G-09.
- **Gravité** : **Mineur**
- **Constat** : le projet charge systématiquement les volumes complets en mémoire et
  matérialise plusieurs copies simultanées. Le seul garde-fou existant est l'avertissement
  d'anatomix au-delà de 4 Go (`anatomix.py:157-163`) — bien conçu, mais isolé et limité à
  un chemin.
- **Le risque futur** : les volumes cliniques croissent (CT photon-counting, IRM haute
  résolution, TEP corps entier en 2 mm). Les cumuls décrits en section G — jusqu'à 7 Go pour
  un Jacobien inutile (G-09), 5 Go pour le QC d'intensité (G-01) — sont déjà à la limite
  d'un poste à 16 Go.
- **Correction recommandée** : au-delà des corrections ponctuelles de la section G, adopter
  **une** règle : toute opération dont le coût mémoire dépasse un multiple du volume doit
  soit sous-échantillonner, soit travailler par blocs, soit émettre l'avertissement
  standardisé d'anatomix. Ajouter à `regix doctor` l'affichage de la mémoire disponible, et
  au manifeste le pic mesuré (`tracemalloc` ou `resource.getrusage`), pour objectiver le
  sujet plutôt que de l'estimer.
- **Tests nécessaires** : `test_peak_memory_stays_within_a_multiple_of_the_input`
  (`tracemalloc`, marqué `slow`, sur un fantôme volontairement grand).

---

## 1bis. Récapitulatif quantitatif

**134 constats** répartis comme suit :

| Gravité | Nombre | Identifiants |
|---|---|---|
| **Bloquant** | 0 | — |
| **Critique** | 3 | B-01, C-01, J-01 |
| **Important** | 44 | A-01…A-06, A-08, A-14, A-17, B-02…B-09, B-11, B-12, B-14, B-15, C-02, C-03, C-05, C-06, D-01, D-02, D-05, E-05*, F-01, F-02, F-07, G-01, G-02, G-03, G-05, I-01…I-04, J-02, J-04, J-06, K-10, L-02, L-04 |
| **Mineur** | 55 | A-07, A-09, A-11…A-13, A-15, A-16, B-10*, B-13, B-16, C-04, C-08, D-03, D-06…D-08, E-01, E-03, E-04, E-05, E-06, E-09, E-11, F-03…F-06, F-08…F-13, F-16, G-04, G-06…G-09, H-01, H-02, H-04, H-06, H-07, H-09, I-05…I-08, I-10, J-03, J-05, J-08, K-01…K-04, K-06, L-01, L-03, L-05, L-07, L-09 |
| **Cosmétique** | 32 | A-10, A-18, C-07, D-04, E-02, E-07, E-08, E-10, E-12, F-14, F-15, F-17, H-3, H-05, H-08, I-09, J-07, K-05, K-07, K-08, K-09, L-06, L-08, et les entrées « Cosmétique » du tableau E-03 |

\* B-10 est classé Important ; E-05 apparaît aussi sous G-04.

**Par domaine** : A (cohérence doc/code) 18 · B (correction) 16 · C (sécurité) 8 ·
D (architecture) 8 · E (duplication/code mort) 12 · F (erreurs/cas limites) 17 ·
G (performances) 9 · H (types/contrats) 9 · I (tests) 10 · J (config/déploiement) 8 ·
K (doc/DX) 10 · L (dette) 9.

**Statut de vérification** : 31 constats **[CONFIRMÉ]** par exécution, 98 **[LECTURE]**,
5 **[À VÉRIFIER]** (A-15, F-06, K-03, E-04 point 3, et l'impact réel de A-04 sur données
cliniques).

---

## 2. Plan de correction ordonné par dépendances

Sept vagues. L'ordre est dicté par les **dépendances techniques** (une correction en
présuppose une autre) et par le **rapport risque/effort**, pas par la gravité seule.
Chaque vague est livrable indépendamment.

---

### Vague 0 — Filet de sécurité, avant toute modification *(≈ 0,5 jour)*

Rien ci-dessous ne change le comportement. L'objectif est de rendre les vagues suivantes
sûres et de figer l'état actuel.

| # | Action | Sert à |
|---|---|---|
| 0.1 | Écrire `tests/test_documentation.py` (**J-06**) — décomptes de tests, options CLI citées, tableau des presets, `#` dans `regix presets`, badge de couverture | rend rouges A-01, A-02, A-05, A-07, A-13 **avant** de les corriger |
| 0.2 | Ajouter un test de non-régression numérique de bout en bout : transformée finale d'un run fantôme comparée à une référence sérialisée, tolérance 1e-6 mm sur 400 points | protège toutes les vagues suivantes |
| 0.3 | Ajouter le test de contrat d'artefacts (**I-05**, **D-06**) : arborescence produite, JSON strict, absence de chemin | protège B-09, C-02, H-07 |
| 0.4 | Créer `tests/test_api.py` avec `TestClient` (**I-01**) | prérequis de C-03 et F-11 |
| 0.5 | Paramétrer `conftest.make_phantom` par direction et anisotropie (**I-04**) — sans encore paramétrer les tests | prérequis de la vague 3 |

**Critère de sortie** : la suite passe, sauf les tests de 0.1 qui documentent les
contradictions connues (à marquer `xfail(strict=True)` avec l'identifiant du constat).

---

### Vague 1 — Critique : ce qui produit un résultat faux ou expose des données *(≈ 2 jours)*

| # | Constat | Action | Dépend de |
|---|---|---|---|
| 1.1 | **B-01** | `label_of` → `labels_of` renvoyant une liste ; adapter `mask_for`, `organ_volumes_ml`, `present_organs`, et le filtre de nomenclature de `pipeline.py:742-746` | — |
| 1.2 | **J-01** | Porter la contrainte à `pydicom>=3.0` ; supprimer `is_little_endian` / `is_implicit_VR` ; afficher la version dans `doctor` | — |
| 1.3 | **C-01** | Supprimer le sel par défaut ; HMAC-SHA256 ; 32 caractères ; avertissement `manifest.warn` si le sel est faible ou absent ; corriger le docstring | — |
| 1.4 | **B-02** | `sitk.Extract` pour la 4D ; refuser explicitement 1D/2D/≥5D | — |

**Pourquoi en premier** : ce sont les trois seuls constats qui, aujourd'hui, peuvent
produire un **masque d'organe faux sans aucun signe** (1.1), une **installation dont toutes
les sorties DICOM plantent** (1.2), ou une **ré-identification de patient en quelques
secondes** (1.3). Aucun ne dépend d'un autre.

**Critère de sortie** : `test_a_directory_of_lung_lobes_yields_one_complete_lung_mask`
passe ; un job CI avec `pydicom==2.4.4` échoue à l'installation (contrainte honnête) ;
`test_a_short_numeric_patient_id_is_not_trivially_enumerable` passe.

---

### Vague 2 — Réparer les garde-fous qui ne se déclenchent pas *(≈ 2 jours)*

Ces mécanismes **existent, sont documentés, et ne fonctionnent pas**. Les réparer est
moins coûteux que d'en écrire de nouveaux, et c'est ce qui restaure la crédibilité du
dispositif de QC.

| # | Constat | Action | Dépend de |
|---|---|---|---|
| 2.1 | **B-04**, **H-06** | `dataclasses.replace(context, …)` dans `ElastixEngine.run` ; retirer les défauts des champs descriptifs de `ParamContext` | — |
| 2.2 | **B-05** | Rendre `min_abs_final_metric` consciente du sens de la métrique (`ZERO_IS_OPTIMAL`) | — |
| 2.3 | **B-12** | `organ_overlap_report` renvoie `dice=0.0` avec motif au lieu de sauter l'organe | — |
| 2.4 | **B-13** | Corriger l'expression `deformable=` passée à `evaluate_gates` | — |
| 2.5 | **F-07** | Refuser (ou convertir) un fichier de points elastix `index` ; contrôle de plausibilité du FOV | — |
| 2.6 | **H-07** | `_round` renvoie `None` ; `allow_nan=False` dans `RunManifest.save` | 0.3 |
| 2.7 | **E-11**, **L-08** | Implémenter `keep_intermediate` en préservant `TransformParameters.0.txt` et `parameters.txt` | — |

**Critère de sortie** : `test_the_quantisation_warning_fires_through_the_full_pipeline`
passe ; `test_an_organ_pushed_out_of_the_field_fails_the_dice_gate` passe ;
`test_manifest_is_strict_json_even_when_metrics_are_unavailable` passe.

---

### Vague 3 — Unifier les masques et la géométrie *(≈ 3 jours)*

C'est la vague la plus structurante. Elle **doit** venir après la vague 0.5
(fantômes obliques) car elle touche la géométrie.

| # | Constat | Action | Dépend de |
|---|---|---|---|
| 3.1 | **A-04**, **G-08** | Calculer le masque corporel **une fois**, sur le volume natif, à 4 mm, puis le rééchantillonner. Rendre le seuil explicite et paramétrable ; ne plus basculer sur Otsu implicitement | 0.5 |
| 3.2 | **D-02** | Introduire le type `Mask(image, role, dilated_mm)` ; implémenter réellement le masque d'initialisation non dilaté | 3.1 |
| 3.3 | **E-01**, **E-02** | Une seule implémentation de `_same_grid` et de `intensity_range` | — |
| 3.4 | **G-03** | Vectoriser `principal_axes` | 0.5 |
| 3.5 | **K-10** | Recouvrement de FOV par échantillonnage Monte-Carlo au lieu de l'intersection d'AABB | 0.5 |
| 3.6 | **I-04** | Paramétrer les tests clés sur identité / oblique / anisotrope | 0.5, 3.1 |
| 3.7 | **A-04** (doc) | Corriger la limitation du README avec le diagnostic exact, et **mesurer à nouveau** l'écart 26 365 / 19 114 mL après 3.1 | 3.1 |

**Critère de sortie** : `test_rigid_recovers_the_ground_truth[oblique]` passe ;
`test_qc_mask_and_criterion_mask_are_the_same_object_resampled` passe ; l'écart de volume
du masque corporel documenté dans le README est re-mesuré et le chiffre mis à jour.

---

### Vague 4 — Contrats d'interface et cas limites *(≈ 2,5 jours)*

| # | Constat | Action | Dépend de |
|---|---|---|---|
| 4.1 | **D-05**, **B-03**, **A-14**, **L-02** | `transforms.load_any_transform` reniflant le contenu ; l'utiliser dans `cli.apply` et `init.mode=file` ; renommer les sorties pour lever l'ambiguïté d'extension | — |
| 4.2 | **B-10** | Supprimer `if value is None: continue` de `_deep_update` ; auditer les appelants ; supprimer la ligne `merged["stages"]` redondante | 0.2 |
| 4.3 | **B-07**, **L-04** | Échec fatal d'un mode d'initialisation unique ; `manifest.warn` pour tout repli ; `init_report["requested"]` | 4.2 |
| 4.4 | **F-01**, **F-02** | `--set` : conversion de toutes les erreurs en `BadParameter` ; typer les options d'énumération | — |
| 4.5 | **B-11** | Passer systématiquement par `AffineTransform` dans `transform_to_elastix_initial` | 0.2 |
| 4.6 | **F-10** | Repli sur `body_mask` plutôt que sur l'union de toutes les structures ; remontée au manifeste | 3.2 |
| 4.7 | **H-01** | Contrat explicite de `target_registration_error` (`mapped_points` / `transform`) | — |
| 4.8 | **F-16** | Bornes pydantic sur tous les champs numériques | — |
| 4.9 | **B-09** | Nettoyage sur `--overwrite` par liste blanche ; `regix.log` en mode `"w"` | D-06 (0.3) |

**Critère de sortie** : `test_apply_accepts_every_transform_file_a_run_produces` passe ;
`test_a_child_preset_can_disable_a_parent_option_with_null` passe ;
`test_set_never_raises_anything_but_badparameter` passe.

---

### Vague 5 — Sécurité et sorties DICOM *(≈ 2,5 jours)*

| # | Constat | Action | Dépend de |
|---|---|---|---|
| 5.1 | **C-02** | `redact_path` appliquée au manifeste, à `config_effective.yaml` et aux messages d'exception entrant dans `manifest.warn` | 1.3 |
| 5.2 | **A-09** | Câbler `file_digest` dans `_load` ; remplacer le chemin par le digest au manifeste | 5.1 |
| 5.3 | **C-06** | SRO : référencer toutes les coupes, ajouter les séquences requises, garantir les type 2, conserver la date d'étude, supprimer le code mort | 1.2 |
| 5.4 | **C-05** | `output.dicom_uid_root` configurable + `REGIX_DICOM_UID_ROOT` ; avertissement sur la racine de test ; `Manufacturer` / `SoftwareVersions` | 1.2 |
| 5.5 | **C-03** | API : allowlist de racines, jeton optionnel, pas de `str(exc)` renvoyé, `overwrite` non forcé | 0.4 |
| 5.6 | **C-04** | `_safe_case_name` dans `batch` + unicité | — |
| 5.7 | **A-08**, **C-08** | Corriger le docstring de `io/dicom.py` ; liste blanche de tags pour la série dérivée ; `remove_private_tags` | 1.2 |
| 5.8 | **F-11** | Bornage de `_JOBS`, pagination, `lifespan` FastAPI | 0.4, 5.5 |
| 5.9 | **J-03** | Module `regix/env.py` ; validation du sel ; ligne « pseudonymisation » dans `doctor` | 1.3 |

**Critère de sortie** : `test_no_source_path_reaches_the_manifest_when_pseudonymize_is_on`
passe ; `test_sro_passes_dciodvfy` passe (ou est explicitement `skip` avec la raison) ;
`test_a_path_outside_the_allowlist_is_refused_with_400` passe.

---

### Vague 6 — Performances *(≈ 2 jours)*

À faire **après** la vague 3, qui supprime déjà une partie du coût (masque corporel calculé
une fois, à 4 mm).

| # | Constat | Action | Dépend de |
|---|---|---|---|
| 6.1 | **G-09** | Jacobien analytique pour une transformée linéaire ; champ en float32 | 0.2 |
| 6.2 | **G-01** | `paired_samples` en un passage ; float32 ; sous-échantillonnage au-delà d'un seuil, consigné au manifeste | 0.2 |
| 6.3 | **G-02** | `describe()` : un seul `np.percentile`, sous-échantillonnage | 0.2 |
| 6.4 | **G-05** | Estimation mémoire dans MIND ; calcul par blocs ; PCA en float32 par tranches | 0.2 |
| 6.5 | **E-05**, **G-04** | Distances de surface calculées une fois par organe | — |
| 6.6 | **G-07** | Format et dpi des figures configurables ; mode *sidecar* | — |
| 6.7 | **G-06**, **B-14** | `list_series` : fusion par UID à travers les répertoires, retri global, `max_depth` | 0.2 |
| 6.8 | **B-15** | Propager `series_uid` jusqu'au CLI et à l'API ; tri total déterministe ; remontée au manifeste | 6.7 |

**Critère de sortie** : le pic mémoire mesuré par `tracemalloc` sur un fantôme
volontairement grand reste sous un multiple documenté de la taille d'entrée ;
`test_a_series_split_across_two_directories_is_loaded_whole` passe.

---

### Vague 7 — Documentation, DX et dette *(≈ 2 jours)*

Les tests de la vague 0.1 sont déjà rouges pour la plupart de ces points : il s'agit de les
faire passer.

| # | Constats | Action |
|---|---|---|
| 7.1 | **A-01** | `regix presets NAME` lit le fichier source ; `--resolved` pour la config résolue |
| 7.2 | **A-02**, **A-07**, **A-12**, **A-13**, **A-17** | Corriger le tableau des presets, les décomptes de tests, l'inventaire de l'environnement, le docstring de `segment`, le nombre de formats de transformée |
| 7.3 | **A-03**, **L-08** | Écrire les images remises à elastix sous `output.write_elastix_inputs` ; générer la ligne de rejeu depuis les chemins réels |
| 7.4 | **A-05** | Corriger la description de `features.enabled` ; ajouter `--allow-cpu-features` |
| 7.5 | **A-06** | Câbler `mask_dilate_mm` / `roi_margin_mm` du profil ; **F-09** porte de déplacement dérivée de `typical_motion_mm` ; supprimer ou justifier `hu_window` |
| 7.6 | **B-08** | Repli en cascade anatomix → MIND → intensités ; corriger le README |
| 7.7 | **B-06** | `ts_task` / `ts_fast` configurables ; sélection automatique `total_mr` ; refus de `total` sur un MR |
| 7.8 | **A-16**, **L-07** | Test de déterminisme sur 400 points, exécuté en CI |
| 7.9 | **J-02**, **J-04**, **J-05**, **J-07** | Job d'installation par extra ; bornes hautes + fichier de verrouillage ; job de construction de paquet ; `timeout-minutes` |
| 7.10 | **K-01**, **L-01** | `CONTRIBUTING.md`, `CHANGELOG.md`, `docs/ARCHITECTURE.md` ; version en source unique |
| 7.11 | **A-11**, **K-04**, **K-05** | `doctor` : instanciation réelle du filtre, table homogène, versions, `except Exception` |
| 7.12 | **E-03** | Suppression du code mort listé « Cosmétique » ; ajout de `vulture` en CI |
| 7.13 | **H-08**, **H-09** | `TypedDict` pour les schémas de métriques ; `mypy` en CI en `continue-on-error` |
| 7.14 | **L-05** | Commande `regix qc OUT_DIR` recalculant le QC depuis un run terminé |

---

### Vue d'ensemble des dépendances

```
Vague 0  (filet)
   ├──> Vague 1  (critique)  ──> Vague 5  (sécurité / DICOM)
   ├──> Vague 2  (garde-fous)
   ├──> Vague 3  (masques / géométrie)  ──> Vague 6  (performances)
   │        └─ 0.5 (fantômes obliques) est un prérequis dur
   └──> Vague 4  (contrats / cas limites)  ──> Vague 7  (doc / DX)

Chemin critique : 0.5 -> 3.1 -> 3.2 -> 4.6
Parallélisables sans conflit : 1.2/1.3 · 2.x · 4.4/4.7/4.8 · 6.5/6.6 · 7.x
```

**Charge totale estimée** : ≈ 16,5 jours-personne, hors validation clinique.
Les vagues 0 à 2 (≈ 4,5 jours) suffisent à lever les trois constats Critiques et à
restaurer les garde-fous documentés.

---

## 3. Checklist de vérification

À dérouler après chaque vague. Les cases marquées **[auto]** sont vérifiables par un test à
écrire ; les autres demandent une inspection.

### 3.1 Correction numérique

- [ ] **[auto]** La transformée d'un run fantôme est identique à la référence à 1e-6 mm près sur 400 points *(0.2)*
- [ ] **[auto]** `mask_for(["lung_left"])` couvre **tous** les lobes gauches *(B-01)*
- [ ] **[auto]** `organ_volumes_ml` somme les labels homonymes *(B-01)*
- [ ] **[auto]** Un volume 4D produit un volume 3D, pixel à pixel identique à t=0 *(B-02)*
- [ ] **[auto]** Le recalage rigide est exact sur un fantôme **oblique** *(I-04)*
- [ ] **[auto]** Aller-retour ITK↔SimpleITK exact sur une direction non triviale *(I-04)*
- [ ] **[auto]** `transform_to_elastix_initial` → `parameter_map_to_transform` est l'identité, y compris pour une Euler ZYX *(B-11)*
- [ ] **[auto]** Le masque corporel est identique avant et après fenêtrage HU *(A-04)*
- [ ] **[auto]** `principal_axes` vectorisé donne le même résultat que la boucle *(G-03)*
- [ ] **[auto]** Jacobien analytique et dense concordent pour une affine *(G-09)*
- [ ] **[auto]** Le recouvrement de FOV n'est pas surestimé sur une paire oblique *(K-10)*

### 3.2 Garde-fous et QC

- [ ] **[auto]** L'avertissement de quantification se déclenche **à travers le pipeline** *(B-04)*
- [ ] **[auto]** Un stage MSE quasi parfait ne fait pas échouer la porte *(B-05)*
- [ ] **[auto]** Un stage MI dégénéré fait toujours échouer la porte *(B-05, non-régression)*
- [ ] **[auto]** Un organe expulsé du champ produit `dice=0.0` et un **FAIL** *(B-12)*
- [ ] **[auto]** Un recalage rigide sans Jacobien n'avertit pas au sujet du repliement *(B-13)*
- [ ] **[auto]** Un fichier de points `index` est refusé ou converti, jamais lu en mm *(F-07)*
- [ ] **[auto]** Une amplitude de déplacement implausible est signalée *(F-09)*
- [ ] **[auto]** Une initialisation demandée qui échoue est **fatale**, pas remplacée *(B-07)*
- [ ] **[auto]** Toute dégradation atteint `manifest.warnings` **et** `report.html` *(L-04)*
- [ ] **[auto]** Un run dégradé n'est jamais rapporté PASS *(L-04)*
- [ ] Le rapport comporte une section « demandé / effectivement réalisé » *(L-04)*

### 3.3 Sécurité et données patient

- [ ] **[auto]** Sans `REGIX_PSEUDONYM_SALT`, un avertissement explicite est émis (ou le run refuse) *(C-01)*
- [ ] **[auto]** Un identifiant patient numérique court n'est pas énumérable avec le sel par défaut *(C-01)*
- [ ] **[auto]** Le pseudonyme fait au moins 32 caractères hexadécimaux *(C-01)*
- [ ] **[auto]** Aucun chemin source n'apparaît dans le manifeste, `config_effective.yaml` ni le rapport *(C-02)*
- [ ] **[auto]** Un avertissement portant un chemin est rédigé dans le rapport *(C-02)*
- [ ] **[auto]** Un chemin hors allowlist est refusé par l'API en 400 *(C-03)*
- [ ] **[auto]** Un lien symbolique s'échappant de l'allowlist est refusé *(C-03)*
- [ ] **[auto]** Une erreur de job ne divulgue aucun chemin *(C-03)*
- [ ] **[auto]** `batch` refuse un nom de cas s'échappant du répertoire de sortie *(C-04)*
- [ ] **[auto]** La racine d'UID DICOM est configurable et utilisée par les deux écrivains *(C-05)*
- [ ] **[auto]** La racine de test déclenche un avertissement *(C-05)*
- [ ] **[auto]** La SRO référence **toutes** les coupes des deux séries *(C-06)*
- [ ] **[auto]** La SRO passe `dciodvfy` *(C-06)*
- [ ] La série DICOM dérivée conserve délibérément l'identité patient — **et le docstring le dit** *(A-08)*

### 3.4 Configuration et déploiement

- [ ] **[auto]** Un preset enfant peut annuler une option du parent avec `null` *(B-10)*
- [ ] **[auto]** `--set` et `with_overrides` se comportent identiquement sur `null` *(B-10)*
- [ ] **[auto]** Une porte QC peut être désactivée via l'API HTTP *(B-10)*
- [ ] **[auto]** `--set` ne lève jamais autre chose que `BadParameter` *(F-01)*
- [ ] **[auto]** Une valeur d'énumération invalide liste les choix acceptés *(F-02)*
- [ ] **[auto]** Toute valeur numérique hors bornes est refusée à la validation *(F-16)*
- [ ] **[auto]** `--overwrite` supprime les artefacts périmés et épargne les fichiers utilisateur *(B-09)*
- [ ] **[auto]** `regix.log` ne contient que le run courant *(B-09)*
- [ ] **[auto]** L'installation avec `pydicom==2.4.4` échoue à la résolution *(J-01)*
- [ ] **[auto]** Chaque extra documenté s'installe et `import regix` réussit *(J-02)*
- [ ] **[auto]** Une roue construite contient les presets et `regix presets` fonctionne hors du dépôt *(J-05)*
- [ ] Les dépendances numériquement significatives ont une borne haute *(J-04)*
- [ ] Un fichier de verrouillage existe et un job CI l'utilise *(J-04)*
- [ ] Tous les jobs CI ont un `timeout-minutes` *(J-07)*

### 3.5 Cohérence documentaire

- [ ] **[auto]** Les décomptes de tests du README correspondent à `--collect-only` *(A-07)*
- [ ] **[auto]** Toute option `--xxx` citée dans la doc existe dans l'app typer *(A-05, A-13)*
- [ ] **[auto]** Le tableau des presets du README concorde avec les YAML chargés *(A-02)*
- [ ] **[auto]** `regix presets NAME` contient les commentaires du fichier source *(A-01)*
- [ ] **[auto]** Les fichiers cités par la ligne `// Replay with:` existent *(A-03)*
- [ ] **[auto]** Chaque attribut d'`OrganProfile` est lu quelque part *(A-06)*
- [ ] **[auto]** Le manifeste enregistre la version de la liaison elastix *(A-12)*
- [ ] **[auto]** Le badge de couverture correspond à la mesure *(I-06)*
- [ ] Le diagnostic de la limitation « masque corporel » du README est re-mesuré et corrigé *(A-04)*
- [ ] Le chiffre « 0.000 mm sur 400 points » est reproductible par un test *(A-16)*

### 3.6 Performances

- [ ] **[auto]** Le pic mémoire reste sous un multiple documenté de la taille d'entrée *(L-09)*
- [ ] **[auto]** Une transformée linéaire ne matérialise aucun champ dense *(G-09)*
- [ ] **[auto]** Les tableaux masqués du QC sont calculés une seule fois *(G-01)*
- [ ] **[auto]** Les distances de surface sont calculées une fois par organe *(E-05)*
- [ ] **[auto]** MIND avertit avant de dépasser le budget mémoire *(G-05)*
- [ ] **[auto]** La taille du `report.html` reste sous un budget *(G-07)*
- [ ] **[auto]** Une série éclatée sur deux répertoires est chargée entière *(B-14)*

### 3.7 Qualité et non-régression

- [ ] **[auto]** `regix/api.py` est couvert par des tests *(I-01)*
- [ ] **[auto]** Le manifeste est du JSON strict, y compris sur un run dégradé *(H-07)*
- [ ] **[auto]** Le résumé des métriques du rapport est déterministe *(E-09)*
- [ ] **[auto]** `import regix.pipeline` ne charge ni torch ni convexadam *(D-04, E-04)*
- [ ] **[auto]** Deux runs identiques produisent la même transformée *(A-16)*
- [ ] **[auto]** Aucune fixture de test n'est ignorée par git *(I-09)*
- [ ] **[auto]** La version est déclarée une seule fois *(L-01)*
- [ ] `ruff check` et `ruff format --check` passent, et le second est bloquant *(J-07)*
- [ ] `mypy` s'exécute en CI *(H-09)*
- [ ] `vulture` ne signale aucun code mort hors allowlist *(E-03)*

---

## 4. Points positifs

Cet audit est volontairement sévère, comme demandé. Il serait malhonnête de ne pas dire que
ce dépôt est **nettement au-dessus de la moyenne** pour un projet de cette nature, et que
plusieurs de ses choix sont exemplaires.

### 4.1 La discipline de justification en commentaire

C'est la qualité la plus remarquable, et elle est systématique. Chaque choix non évident
est justifié **à l'endroit exact où il s'applique**, avec le raisonnement et souvent la
mesure. Exemples :

- `pyproject.toml:57-59` explique pourquoi `ruff` seul est borné haut — argument juste
  (une nouvelle version ajoute des règles et rougit la CI sans commit) ;
- `pyproject.toml:102-106` explique pourquoi `UP038` reste ignorée alors que la règle n'existe
  plus : « ignorer une règle disparue est l'orthographe portable ». Raisonnement fin ;
- `pyproject.toml:80-85` explique pourquoi `pythonpath = ["."]` est nécessaire — la différence
  entre `python -m pytest` et le script `pytest`, qui est exactement le piège que la CI a
  rencontré ;
- `.gitignore:32-36` documente précisément pourquoi `!tests/data/` n'affaiblit rien ;
- `mind.py:89-105` justifie l'abandon de scipy avec **trois mesures chiffrées** et deux
  alternatives explicitement évaluées et rejetées (`sitk.BoxMean` : 20 % d'écart ;
  cumsum numpy : 2,5× plus lent). C'est le niveau de rigueur qu'on attend d'une note
  technique, pas d'un commentaire de code ;
- `initialize.py:196-205` explique pourquoi l'ordre de composition de la rotation de sonde
  compte, avec le chiffre de l'erreur qu'aurait produite l'autre ordre (~50 mm) ;
- `intensity.py:100-118` (`resolve_prep`) explique pourquoi un sentinelle `"auto"` est
  nécessaire, pourquoi `model_fields_set` ne convient pas, et quel bug précis cela corrige.

**La consigne d'audit demandait de lire la justification avant de contester un choix.**
Dans la grande majorité des cas, la justification tenait. Les constats A-02, A-04, A-15 et
A-10 sont les rares cas où elle est fausse ou dépassée — sur plusieurs dizaines vérifiées.

### 4.2 Les invariants cliniques sont bien posés et bien tenus

- **Les intensités natives atteignent elastix.** L'analyse du bug (normalisation min-max
  cassant les fichiers de paramètres déclarant `short`), sa mesure (`6.7e-16` de MI,
  0,32 mm après correction), la généralisation qui en est tirée (« le préprocessing
  spécifique à un consommateur appartient à ce consommateur ») et le corollaire découvert
  au passage (le clipping d'anatomix était un no-op) constituent un raisonnement d'ingénierie
  de premier ordre. **Et il est verrouillé par cinq tests**
  (`test_no_bundled_preset_rescales_the_intensities`,
  `test_the_anatomix_preparation_stays_inside_the_feature_path`, etc.).
- **La sortie est reconstruite depuis le volume mobile d'origine sur la grille fixe
  d'origine.** L'invariant est énoncé en tête de `pipeline.py`, appliqué
  (`pipeline.py:391-396`, commentaire « native intensities, never the preprocessed volume »)
  et testé (`test_output_keeps_native_intensities`).
- **Le QC est indépendant du critère optimisé.** Le raisonnement (« marking your own
  homework ») est juste, la hiérarchie de fiabilité proposée (TRE > Dice/HD95 > Jacobien >
  NCC/NMI) est celle de la littérature, et le module la met effectivement en œuvre.
- **Aucune inverse de transformée dense n'est approchée.** `warp_landmarks_moving_to_fixed`
  renvoie `None` plutôt qu'une approximation, avec la justification explicite : « an
  invisible 3 mm error is more dangerous than no result at all ». C'est le bon arbitrage,
  et il est testé (`test_inverse_is_refused_for_a_dense_transform`).
- **La nomenclature n'est jamais devinée.** `ExternalSegmenter` nomme `label_N` et avertit
  plutôt que de supposer que 1 = foie. Le commentaire l'explique, un test le verrouille
  (`test_missing_nomenclature_guesses_nothing`).

### 4.3 Les presets encodent de vraies décisions cliniques

Le CBCT est rigide parce que la question clinique est le déplacement de la table ; le TEP
n'est pas déformé parce que cela redistribuerait l'activité et corromprait les SUV ; le
crâne ne change pas de taille en trois mois donc l'échelle est verrouillée à 2 %.
Ce ne sont pas des réglages arbitraires habillés en presets — c'est du savoir métier
encodé, et il est écrit noir sur blanc dans chaque fichier.

### 4.4 La protection des données est **pensée**, même là où elle est incomplète

Le `.gitignore` s'ouvre sur « A single DICOM slice left in a public repository is a health-data
breach » et applique des règles délibérément larges. Le disclaimer réglementaire a une
source unique (`regix.DISCLAIMER`) réutilisée par le manifeste, le rapport, l'API et le
CLI — et **un test vérifie qu'elle ne dérive pas**
(`test_the_regulatory_disclaimer_has_a_single_wording`). L'avertissement « Not a medical
device » figure en tête du rapport HTML. La pseudonymisation existe et est appliquée
systématiquement aux identifiants. Les constats C-01 et C-02 portent sur l'exécution, pas
sur l'intention.

### 4.5 Le dispositif de portes d'acceptation

`qc/gates.py` est le meilleur module du projet. Chaque contrôle rapporte **valeur mesurée,
seuil et verdict** ; les mesures indisponibles produisent WARN et non un silence ;
`min_abs_final_metric` est reporté **même quand il passe**, avec une justification explicite
(« this is the one number that betrays a silent failure »). La distinction PASS/WARN/FAIL
est cohérente et le statut agrégé prend le maximum de sévérité. C'est un modèle.

### 4.6 Honnêteté sur les limites

La section « Limitations and Future Improvements » est inhabituellement franche : le
déterminisme est présenté comme une propriété du build elastix et non comme une garantie
Regix, avec la consigne explicite de le revérifier ; l'écart du masque corporel est chiffré
et publié alors que rien n'y obligeait ; les chemins anatomix et TotalSegmentator sont
déclarés **non exécutés dans cet environnement**. Le README dit aussi que la couverture
manquante est documentée « rather than quietly excluded from the measurement » — et c'est
vrai, il n'y a aucun `# pragma: no cover` abusif (les 15 occurrences trouvées portent
toutes sur des branches réellement inatteignables).

### 4.7 Qualité de la suite de tests

122 tests, aucun besoin de GPU ni de données patient, fantômes numériques avec **vérité
terrain exacte** (la transformée est imposée, donc l'erreur mesurée est l'erreur réelle),
scénarios négatifs présents (deux volumes sans rapport doivent produire FAIL), et plusieurs
tests qui verrouillent des bugs passés (`test_the_real_zoo_fixture_reached_the_repository`,
`test_a_zero_similarity_gain_is_not_reported_as_a_clean_pass`,
`test_the_overlay_figure_moves_in_all_three_planes`). Transformer un incident en test est
la bonne réponse, et le projet le fait.

### 4.8 Autres points notables

- **Imports paresseux** : `regix/__init__.py` utilise `__getattr__` pour garder `import regix`
  instantané ; `cli.py` importe dans chaque commande. Mesuré, cohérent, documenté.
- **Fichiers de paramètres elastix bidirectionnels** : lire un fichier du zoo *et* en écrire
  un est rare, et le traitement des quatre clés ré-imposées — avec justification individuelle
  de chacune — est soigné.
- **`_from_parameter_file` refuse deux incohérences plutôt que de les avertir**, avec la
  bonne raison : « both produce a plausible wrong answer ». Distinction juste entre ce qui
  mérite un refus et ce qui mérite un avertissement.
- **`jacobian_figure` renvoie `None` pour une transformée linéaire** parce qu'une carte de
  couleur uniforme ressemblerait à un bug d'affichage. Attention au lecteur réel.
- **Les commentaires « établi par mesure »** (`engine.py:15-24`, `params.py:20-27`) portent
  sur des comportements elastix non documentés en amont, avec le message d'erreur exact que
  produirait l'oubli. C'est de la connaissance durement acquise, correctement consignée.
- **`_paired_arrays`** utilise `GetArrayFromImage` plutôt que `GetArrayViewFromImage`, avec
  un commentaire expliquant la violation d'accès qu'évite ce choix. Piège subtil, bien géré.

---

## 5. Zones non vérifiées, et pourquoi

Les points ci-dessous n'ont **pas** pu être vérifiés dans cet audit. Ils sont listés avec la
raison exacte et ce qu'il faudrait pour les couvrir.

| # | Zone | Raison exacte | Ce qu'il faudrait |
|---|---|---|---|
| 5.1 | **Inférence anatomix** (`features/anatomix.py:105-206`) — chargement HuggingFace, fenêtre glissante monai, `voxel_normalize` sur la variante de base (**A-15**) | `torch`, `monai` et le paquet `anatomix` ne sont pas installés dans cet environnement, et les poids ne sont pas redistribués | Un poste GPU avec `pip install -e ".[features]"` et les poids téléchargés ; comparer la TRE avec `voxel_normalize="l2"` et `"none"` sur une paire de référence |
| 5.2 | **Étape déformable GPU** (`registration/convexadam.py:46-155`) — la boucle Adam, la conversion `_normalized_to_world_field` | Aucun GPU, torch absent. **La conversion normalisé → monde n'est couverte par aucun test**, et son propre docstring signale trois sources d'erreur classiques | Un GPU ; et surtout un test CPU de `_normalized_to_world_field` seul, comparant à une transformée analytique connue sur une grille oblique — **faisable sans GPU** et à faire en priorité |
| 5.3 | **Appel TotalSegmentator** (`organs/segmenter.py:227-279`) | Paquet non installé ; le télécharger implique ses poids et plusieurs Go | Un environnement avec `pip install -e ".[totalsegmentator]"` ; ou, à moindre coût, un double de l'API Python pour tester le câblage (B-06) sans les poids |
| 5.4 | **Comportement sur données DICOM cliniques réelles** — B-14 (série éclatée), B-15 (ambiguïté de série), C-06 (acceptation de la SRO par un TPS), A-04 (impact réel du basculement Otsu) | Aucune donnée patient dans le périmètre d'audit ; les fantômes DICOM des tests sont synthétiques et mono-série | Un jeu de données de validation anonymisé, multi-séries, avec des acquisitions obliques ; et un import réel de la SRO dans un poste de fusion ou un TPS |
| 5.5 | **Conformité DICOM de la SRO et de la série dérivée** (**C-06**, **C-08**) | Aucun validateur DICOM (`dciodvfy`, `dcmvalidate`) disponible ici. L'analyse repose sur la lecture de l'IOD, pas sur une validation outillée | `dicom3tools` en CI, avec le test `test_sro_passes_dciodvfy` |
| 5.6 | **Compatibilité pydicom 2.x** (**J-01**) | pydicom 3.0.2 installé ; l'incompatibilité est établie par la **signature** de `save_as`, pas par une exécution en 2.x | Un job CI `pip install pydicom==2.4.4` — qui doit échouer aujourd'hui |
| 5.7 | **Comportement multi-plateforme** | Audit conduit sous Windows 11 / Python 3.13 uniquement. Les chemins, la sensibilité à la casse, les verrous de fichiers et `MAX_PATH` diffèrent sous Linux — or la CI ne teste que Linux, et l'auteur développe sous Windows : **les deux moitiés du domaine sont couvertes séparément, jamais ensemble** | Ajouter `windows-latest` et `macos-latest` à la matrice CI |
| 5.8 | **Python 3.10 et 3.11** | Environnement local en 3.13. La CI les couvre, mais **3.13 n'est dans aucune matrice** alors que le projet déclare `requires-python = ">=3.10"` sans borne haute | Ajouter 3.13 à la matrice CI |
| 5.9 | **numpy 1.x** (**J-04**) | numpy 2.3.5 installé ; `numpy>=1.24` autorise pourtant 1.x, dont les règles de promotion de types diffèrent (NEP 50) | Un job CI avec `numpy<2` |
| 5.10 | **Performances réelles** (section G entière) | Toutes les estimations mémoire et temps sont **calculées** à partir des dimensions et des types, pas **mesurées**. Le fantôme de test (64×80×80) est trop petit pour révéler quoi que ce soit | Profilage `tracemalloc` + `py-spy` sur un CT réel 512×512×800 |
| 5.11 | **Cas dégénéré du garde-fou de conversion** (**F-06**) | Je n'ai pas construit de transformée dont les trois sondes fixes échouent toutes. Le risque symétrique sur `flatten_linear` a été testé et **n'a pas** reproduit de faux positif | Un test avec une B-spline dont le support est volontairement éloigné de l'origine |
| 5.12 | **Corruption de sortie par `rich`** (**K-03**) | Les presets livrés ne contiennent pas de crochets ; le risque est déduit de l'API de `rich`, non reproduit | Un test avec `StageConfig.label = "[test]"` et une redirection de `--dry-run` |
| 5.13 | **Extras `organs` et `all`** (**E-04** point 3) | Auto-référence `regix[totalsegmentator]` non testée en installation éditable depuis un checkout | Le job d'installation par extra (J-02) |
| 5.14 | **Comportement sous charge de l'API** (**F-11**) | Aucun test d'API ; la croissance de `_JOBS` est déduite du code | Un test soumettant 10 000 jobs factices et mesurant la mémoire |
| 5.15 | **Qualité clinique du recalage** | Hors du périmètre d'un audit de code. Les tests fantômes prouvent que le pipeline **recouvre une transformée imposée**, pas qu'il recale correctement de l'anatomie réelle. Le README le dit lui-même : « The phantom has a ground truth; real patient data does not » | Une étude de validation sur données réelles avec landmarks posés par un radiologue |

---

## 6. Top 10 prioritaire

**Ce top 10 ne remplace pas la liste exhaustive ni le plan de correction.** Il répond à une
question différente : *si l'on ne pouvait corriger que dix choses avant de laisser
quelqu'un d'autre utiliser ce logiciel, lesquelles ?* Le critère est le produit
**(probabilité d'occurrence) × (gravité de la conséquence) × (invisibilité du défaut)**.

---

**1. B-01 — Un masque « poumon gauche » ne couvre qu'un lobe** · *Critique*
La collision d'alias fait que `mask_for(["lung_left"])` renvoie **la moitié de l'organe**
(confirmé : 200 voxels sur 400). Le masque tronqué alimente simultanément le critère
elastix, le recadrage ROI, l'initialisation et le Dice de QC — et le Dice en sort
**meilleur**, donc la porte `min_dice: 0.95` du preset `ct_ct_lung_4d` passe. Défaut
silencieux, sur un chemin nominal, qui produit une fausse assurance.
→ `labels_of` renvoyant une liste.

**2. C-01 — La pseudonymisation est réversible en quelques secondes** · *Critique*
Sel par défaut `"regix"` (constante publique), SHA-256 tronqué à 40 bits, espace
d'identifiants patients de 10⁶ à 10⁸. Le docstring affirme « Never reversible without the
salt » et le README ajoute « verified by test » — le test vérifie seulement que la sortie
diffère de l'entrée. Le rapport est explicitement conçu pour être envoyé par courriel.
→ Pas de sel par défaut, HMAC, 32 caractères, avertissement visible.

**3. J-01 — La contrainte `pydicom>=2.4` est fausse : le code exige 3.0** · *Critique*
`save_as(..., enforce_file_format=...)` n'existe pas en pydicom 2.x. Une installation
satisfaisant la contrainte déclarée voit **toutes ses sorties DICOM planter**, et ni
`regix doctor` ni la CI ne le détectent. Le code mélange par ailleurs les deux API
(`is_little_endian` est un vestige de la 2.x, inopérant en 3.x).
→ `pydicom>=3.0`, nettoyage des vestiges, version affichée par `doctor`.

**4. B-04 — Le garde-fou de quantification est mort** · *Important*
`ElastixEngine.run` reconstruit le `ParamContext` et **oublie `intensity_range`**
(confirmé), ce qui fait sortir `_warn_on_quantisation` immédiatement. Ce garde-fou occupe
une section entière du README et 33 lignes de commentaire. Il n'a jamais pu se déclencher.
Le test existant le valide en isolation, jamais à travers le pipeline — c'est le trou de
test qui l'a laissé passer.
→ `dataclasses.replace` au lieu d'une reconstruction champ par champ.

**5. C-02 — Les chemins patients circulent en clair dans les artefacts** · *Important*
Confirmé sur le manifeste réel : `inputs.fixed.source = C:\…\dataregix\Fixed`. Dans un
service, un chemin **est** un identifiant nominatif. Le rapport HTML — celui qu'on envoie
par courriel — hérite du problème par le canal `warnings`, donc **uniquement sur les runs
dégradés**, c'est-à-dire ceux qu'on transmet pour analyse. Le README affirme l'inverse.
→ `redact_path`, digest à la place du chemin.

**6. B-12 + B-07 + F-10 — Trois dégradations silencieuses d'un recalage ciblé** · *Important*
Prises ensemble, elles permettent qu'un preset centré organe se transforme intégralement en
recalage corps entier, avec un statut PASS : l'initialisation `organ_centroid` échoue et est
remplacée par la géométrie (B-07), le masque de critère devient le corps entier (F-10), et
un organe expulsé du champ produit un WARN au lieu d'un FAIL (B-12). Aucune de ces trois
dégradations n'atteint le rapport. C'est la violation la plus nette du principe affiché
« a failure is labelled, never hidden ».
→ Échec fatal en mode unique, remontée systématique au manifeste, `dice=0.0` explicite.

**7. B-02 — Le traitement 4D produit une image 2D** · *Important*
Confirmé : `image[..., 0]` renvoie une image de **dimension 2**, pas le premier point
temporel. Le message de log affirme le contraire. Le run échoue plus loin avec une erreur
sans rapport, après avoir dépensé le temps de chargement et de segmentation.
→ `sitk.Extract`, et refus explicite des dimensions non gérées.

**8. B-03 + A-14 + D-05 — Deux formats de transformée partagent l'extension `.txt`** · *Important*
Confirmé : `regix apply transform/final_transform.txt` — un fichier **que Regix vient
d'écrire** — plante avec une `RuntimeError` du parseur elastix. Le même défaut fait que
`init.mode=file` avec un fichier elastix échoue puis **se replie en silence** sur une
initialisation géométrique. Une seule cause : aucun chargeur unifié.
→ `load_any_transform` reniflant le contenu, et renommage des sorties.

**9. A-04 — Le diagnostic publié du défaut de masque corporel est faux** · *Important*
Le README explique l'écart 26 365 mL / 19 114 mL par la morphologie dépendante de la
résolution, « **pas** l'échelle d'intensité », en affirmant que les deux passes utilisent le
même seuil −300 HU. Confirmé : après une fenêtre `ct_liver` ou `ct_bone`, `body_mask`
**bascule sur Otsu** — un algorithme différent, pas un seuil différent. Cela concerne
**5 presets sur 8**. Un diagnostic faux envoie la correction future dans la mauvaise
direction, et fait diverger le masque de critère du masque de QC.
→ Un seul masque corporel, calculé sur le volume natif, seuil explicite.

**10. B-10 — Aucune option ne peut être annulée par héritage de preset ni par l'API** · *Important*
Confirmé : `_deep_update` ignore `None`, donc `percentile_clip: null`,
`working_spacing_mm: null`, `min_abs_final_metric: null` — toutes documentées comme
désactivables — sont **silencieusement ignorées**. `--set` y parvient, `with_overrides`
non : deux mécanismes documentés côte à côte, aux sémantiques divergentes. Aucun client
HTTP ne peut désactiver une porte QC. Le défaut est aujourd'hui latent par chance (les six
`null` des presets livrés aboutissent au même résultat par un autre chemin) ; il se
réveillera à la première déclaration qui compte.
→ Supprimer le filtrage de `None`, tester la parité des deux mécanismes.

---

### Mention hors classement

**J-06 — Rien ne vérifie la cohérence entre la doc et le code.**
Ce n'est pas un défaut du produit mais du processus, et c'est la cause d'une bonne partie de
la section A (onze affirmations fausses, toutes détectables mécaniquement). Écrire
`tests/test_documentation.py` coûte une centaine de lignes, se fait **avant** tout le reste
(vague 0.1), et empêche la réapparition de toute cette classe de défauts. Sur le rapport
valeur/effort, c'est la première chose à faire.

---

*Fin de l'audit. 134 constats, dont 31 confirmés par exécution. Aucun fichier du projet n'a
été modifié.*
