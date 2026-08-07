# Reprise du travail sur un autre poste

Ce fichier existe pour une raison précise : permettre de reprendre l'audit et le plan de
correction **à partir d'un simple `git clone`**, sans rien de ce qui restait sur la machine
d'origine. Il décrit l'état exact, ce qu'il faut refaire localement, et ce qui vient
ensuite.

Dernière mise à jour : **7 août 2026**, fin de la vague 0.

---

## 1. Où en est le travail

| Élément | Emplacement | État |
|---|---|---|
| Audit exhaustif — 134 constats | `AUDIT.md`, branche `main` | terminé |
| Vague 0 — filet de sécurité (tests seuls) | branche `fix/wave-0-safety-net` | terminée, vérifiée |
| Vagues 1 à 7 — corrections | — | **non commencées** |

```
main                   7e50115  docs: add the full audit (134 findings, 7 correction waves)
                       <ce commit>  docs: how to resume this work from another machine

fix/wave-0-safety-net  2ddccaf  test(wave-0): safety net before touching any behaviour
                       f975599  test: skip the golden comparison across platforms too
```

La branche part de `main` **avant** ce fichier ; `HANDOFF.md` n'y est donc pas visible.
C'est sans importance : on le lit depuis `main`, et la vague 1 partira de la branche.

**La vague 0 n'est pas fusionnée dans `main`, volontairement.** La convention retenue est
*une branche par vague*, pour que chaque vague reste relisible et réversible seule.
Fusionner ou non est une décision à prendre, pas un oubli — voir §6.

### Décisions déjà prises, à ne pas rejouer

1. **Ordre de travail** : vague 0 (filet) puis vague 1 (les 3 Critiques). Le plan complet
   des 7 vagues est en `AUDIT.md` §2, ordonné par dépendances techniques.
2. **Une branche par vague** : `fix/wave-N-<nom>`.
3. **Aucun fichier de production n'a été modifié à ce jour.** La vague 0 n'ajoute que des
   tests, un paramètre à `conftest.make_phantom` et deux jobs CI. Rien ne peut avoir
   régressé.

---

## 2. Remettre l'environnement en état

```bash
git clone https://github.com/Thibescobar/regix.git
cd regix
git checkout fix/wave-0-safety-net

python -m venv .venv
# Linux / macOS :  source .venv/bin/activate
# Windows        :  .venv\Scripts\activate

pip install -e ".[dev]"
pip install fastapi httpx      # facultatif : sans eux, tests/test_api.py se skippe proprement
```

Vérification :

```bash
regix doctor        # doit sortir en 0 et nommer itk-elastix
pytest -q           # ~3 min
```

### Ce à quoi il faut s'attendre au premier `pytest`

- **0 échec, 0 erreur.**
- **13 xfails** — normaux et voulus, voir §4.
- **1 skip** de `test_the_final_transform_matches_the_golden_reference` — normal sur une
  machine neuve, voir §3. **C'est le point le plus important de ce fichier.**
- `tests/test_api.py` entièrement skippé si `fastapi` n'est pas installé.

Si autre chose échoue, ce n'est pas un effet de la reprise : c'est une régression réelle,
ou une différence d'environnement à diagnostiquer avant d'aller plus loin.

---

## 3. Le point fragile : la référence numérique (« golden »)

`tests/test_contract.py` compare la transformée finale d'un recalage fantôme à une
référence enregistrée dans `tests/data/golden_transform.json`, sur 400 points, à **1e-6 mm**.
C'est le filet qui distingue un refactoring d'un changement de comportement silencieux, et
toutes les vagues suivantes s'appuient dessus.

**Cette référence est spécifique à la machine.** Elle a été capturée avec :

```
itk / elastix 5.4.6      plateforme win32      Python 3.13
```

Le déterminisme d'elastix est une propriété du build, pas une garantie que Regix
donne — le README le dit, et il a raison. Sur un autre poste, le test **skippe** au lieu
d'échouer, en nommant l'écart.

### Ce qu'il faut faire sur le nouveau poste, dans cet ordre

```bash
# 1. Sur un checkout STRICTEMENT NON MODIFIÉ de fix/wave-0-safety-net
git status                       # doit être propre

# 2. Capturer la référence de cette machine
REGIX_UPDATE_GOLDEN=1 pytest tests/test_contract.py -k golden
#   Windows PowerShell :  $env:REGIX_UPDATE_GOLDEN="1"; pytest tests/test_contract.py -k golden

# 3. Vérifier qu'elle mord
pytest tests/test_contract.py -k golden        # doit PASSER, pas skipper

# 4. Commiter la nouvelle référence AVANT de toucher au code
git add tests/data/golden_transform.json
git commit -m "test: recapture the golden reference on <machine/OS/elastix version>"
```

> **L'ordre n'est pas négociable.** Capturer la référence après avoir commencé la vague 1
> reviendrait à figer le comportement *corrigé* comme référence, et le filet ne servirait
> plus à rien. Il ne protégerait plus contre l'erreur qu'il est censé attraper.

Le test vérifie aussi que les points sondés eux-mêmes n'ont pas bougé : si `_probe_points`
ou la géométrie du fantôme change, il le dit plutôt que de comparer silencieusement deux
questions différentes.

---

## 4. Les 13 xfails stricts : comment les lire

Chaque `@pytest.mark.xfail(strict=True)` de `tests/test_documentation.py`,
`tests/test_contract.py` et `tests/test_api.py` documente **une affirmation du projet qui est
fausse aujourd'hui**, et nomme le constat d'audit correspondant dans sa `reason`.

| Constat | Test | Ce qu'il enregistre |
|---|---|---|
| A-01 | `test_presets_command_prints_the_source_comments` | `regix presets NAME` perd les commentaires |
| A-02 | `test_the_readme_preset_table_agrees_on_n4` | le README annonce N4, le preset le désactive |
| A-03 | `test_the_replay_line_points_at_files_that_exist` | la ligne `// Replay with:` cite des fichiers inexistants |
| A-06 | `test_every_organ_profile_field_is_consumed_somewhere` | 4 champs de profil d'organe ne sont jamais lus |
| A-07 | `test_readme_test_counts_match_the_collection` | badge 122 / prose 90 / réalité autre |
| A-12 | `test_the_environment_report_covers_...` | la version d'itk-elastix n'est pas enregistrée |
| A-13 | `test_every_documented_cli_option_exists` | `--backend` et `--allow-cpu-features` n'existent pas |
| C-02 | `test_no_input_path_reaches_the_artifacts` | le chemin source atterrit dans `run_manifest.json` |
| C-03 | `test_a_path_outside_the_allowlist_is_refused` | aucune allowlist de chemins n'existe |
| C-03 | `test_a_job_error_does_not_leak_a_filesystem_path` | l'erreur renvoyée au client nomme le fichier |
| E-12 | `test_declared_package_data_directories_exist` | `qc/templates/` déclaré, inexistant |
| F-11 | `test_the_job_history_is_bounded` | `_JOBS` croît sans borne, `/jobs` non paginé |
| H-07 | `test_the_manifest_refuses_to_serialise_a_non_finite_metric` | le manifeste écrit `NaN`, JSON invalide |
| I-06 | `test_the_coverage_badge_cannot_drift` | 3 points de jeu entre le badge et le seuil CI |

**La règle** : `strict=True` signifie qu'un xfail qui se met à **passer fait échouer la
CI**. C'est voulu. Quand une correction rend l'un d'eux vert, le marqueur doit disparaître
**dans le même commit que la correction**. Un xfail qui passe est un bug de ces fichiers,
jamais une bonne nouvelle à ignorer.

Trois de ces treize ne sont pas des reprises de lecture mais des **reproductions faites
pendant la vague 0** : C-02 (le canari a trouvé le chemin), H-07 (le manifeste a bien
écrit `NaN`) et A-03 (les fichiers de rejeu manquent bien).

---

## 5. La suite : vague 1

Objectif : les 3 constats **Critiques** plus B-02. Détail complet en `AUDIT.md` §2,
vague 1. Contrairement à la vague 0, **celle-ci change le comportement**.

```bash
git checkout fix/wave-0-safety-net
git checkout -b fix/wave-1-critical
```

| # | Constat | Fichiers | Nature |
|---|---|---|---|
| 1.1 | **B-01** — un masque « poumon gauche » ne couvre qu'un lobe | `regix/organs/segmenter.py`, `regix/organs/roi.py`, `regix/pipeline.py:742-746` | `label_of` → `labels_of` renvoyant une liste ; touche 5 sites d'appel |
| 1.2 | **J-01** — `pydicom>=2.4` déclaré, le code exige 3.0 | `pyproject.toml:40`, `regix/io/writers.py:207-209,326`, `regix/cli.py` | contrainte + retrait des vestiges 2.x + version dans `doctor` |
| 1.3 | **C-01** — pseudonymisation réversible en quelques secondes | `regix/logging_utils.py:72-78` | plus de sel par défaut, HMAC-SHA256, 32 caractères, avertissement visible |
| 1.4 | **B-02** — le traitement 4D produit une image 2D | `regix/io/volume.py:154-156` | `sitk.Extract` + refus explicite des dimensions non gérées |

### Tests à écrire avec ces corrections

- `test_a_directory_of_lung_lobes_yields_one_complete_lung_mask` — **le test qui aurait
  attrapé B-01** ; 5 fichiers de lobes, `mask_for(["lung_left"])` doit couvrir les deux
  lobes gauches ;
- `test_organ_volumes_sum_duplicate_names`, `test_present_organs_has_no_duplicates` ;
- `test_a_4d_volume_yields_the_first_3d_time_point`, `test_a_2d_image_is_refused` ;
- `test_the_default_salt_is_refused_or_warned_about`,
  `test_a_short_numeric_patient_id_is_not_trivially_enumerable` (doit **échouer** avant
  correction) ;
- job CI installant `pydicom==2.4.4`, qui doit échouer à la résolution.

### Vérification de sortie de vague

```bash
pytest tests/test_contract.py -k golden    # la transformée n'a pas bougé d'un micron
pytest -q                                   # 0 échec
ruff check regix tests && ruff format --check regix tests
```

**C-01 est une rupture volontaire** : quiconque s'appuyait sur le sel par défaut verra ses
pseudonymes changer. C'est le but — l'ancien était énumérable. À signaler dans le
`CHANGELOG.md` que la vague 7 doit créer.

---

## 6. Décisions en attente

1. **Fusionner `fix/wave-0-safety-net` dans `main` ?** Argument pour : un clone frais
   récupère le filet immédiatement, et la vague 1 doit de toute façon partir de là.
   Argument contre : la convention branche-par-vague voulait une relecture avant fusion.
2. **Visibilité du dépôt — le dépôt est public** (vérifié : `git ls-remote` aboutit sans
   authentification). `AUDIT.md` y décrit en détail trois faiblesses de sécurité **non
   corrigées** : pseudonymisation réversible en quelques secondes (C-01), API sans
   authentification acceptant des chemins arbitraires en lecture et en écriture (C-03),
   traversée de répertoire dans `batch` (C-04). C'est donc, publiquement, une carte des
   faiblesses du logiciel.
   Pour un projet de recherche audité par son propre auteur et que personne ne déploie
   en clinique, c'est une pratique normale et plutôt saine. Mais c'est un choix : les
   options sont (a) laisser tel quel et corriger vite — la vague 1 traite C-01, la
   vague 5 traite C-03 et C-04 ; (b) passer le dépôt en privé le temps des vagues 1 et 5 ;
   (c) retirer temporairement la section C de `AUDIT.md` du dépôt public.
   L'option (a) est raisonnable ici ; elle mérite d'être retenue sciemment.
3. **Resserrer le seuil de couverture** et régénérer le badge (I-06), une fois que les
   nouveaux tests auront déplacé la mesure.

---

## 7. Ce qui reste sur la machine d'origine et **ne doit pas** être récupéré

| Élément | Pourquoi il n'est pas dans le dépôt |
|---|---|
| `e2e_out/` (~40 Mo) | Sorties d'un run réel, correctement ignorées par `.gitignore`. Contient `run_manifest.json` avec un chemin nominatif (`…\Desktop\dataregix\Fixed`) — c'est d'ailleurs la preuve du constat C-02. **Ne jamais committer.** |
| venv jetable avec `fastapi` | Créé hors du dépôt pour vérifier `tests/test_api.py` sans toucher à l'environnement principal. Se recrée en une commande (§2). |
| `.coverage`, `.pytest_cache`, `.ruff_cache` | Ignorés, régénérés à l'exécution. |

Rien d'autre n'était local. Tout le travail utile est dans les deux commits.

---

## 8. Points d'entrée dans l'audit

`AUDIT.md` fait 4 300 lignes ; il n'est pas fait pour être lu d'un bout à l'autre.

| Besoin | Section |
|---|---|
| Par quoi commencer | §6 — top 10, puis §2 — plan en 7 vagues |
| Un constat précis | §1, identifiants `A-01` … `L-09`, groupés par domaine |
| Vérifier une correction | §3 — checklist, les cases **[auto]** indiquent le test à écrire |
| Ce qui va bien, et pourquoi ne pas y toucher | §4 — points positifs |
| Ce que l'audit n'a **pas** pu vérifier | §5 — 15 zones, avec la raison exacte |
| Compte par gravité | §1bis |

Deux conventions de l'audit valent d'être retenues :

- **[CONFIRMÉ]** = reproduit par exécution pendant l'audit (31 constats).
  **[LECTURE]** = tracé par lecture du code (98). **[À VÉRIFIER]** = risque identifié, non
  reproduit (5). Ne pas traiter les trois de la même façon.
- Le projet a une excellente convention : **justifier tout choix non évident en commentaire
  à l'endroit concerné**. Elle n'est écrite nulle part (constat K-01) mais elle est réelle et
  suivie. La respecter dans les corrections.
