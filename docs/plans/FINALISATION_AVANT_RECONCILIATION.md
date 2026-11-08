# Finalisation du travail en cours avant réconciliation du fork

Statut : plan de travail, inventaire établi le 6 novembre 2026. Ce document ne
certifie pas que les tests ou validations décrits ci-dessous ont réussi.

## Mission et limites

Stabiliser le travail local actuel pour obtenir un point de départ vérifié pour
la future réconciliation avec l'officiel. Cette mission ne comprend ni fusion,
ni rebase, ni mise à niveau des dépendances ou des noms d'outils MCP. Garder le
comportement actuel sauf correction justifiée et couverte. Ne pas ajouter le
support de plusieurs sessions simultanées, de PCM natif ou de nouveaux moteurs.

Point de départ : branche `codex/vscode-voice-bridge`, commit
`1bbcd9295266a892b61dbe7af00c5a4b17f6e835`. Une branche de sauvegarde
`safety/pre-reconciliation-20261106` pointe sur ce commit : elle **n'inclut pas**
les modifications non commitées. L'archive privée
`/Users/admin/voicebox-backups/working-tree-20261106-184748` contient les 32
fichiers modifiés ou nouveaux existants lors de l'inventaire, un patch des
fichiers suivis, un patch de l'index, l'état Git et un manifeste SHA-256 ; les
32 contenus archivés ont été vérifiés. Elle ne contient ni données utilisateur
ignorées, ni modèles, ni l'historique Git complet. Ne pas la publier : elle peut
contenir de la configuration locale. Ce plan a été créé après l'archive.

Ne pas restaurer cette sauvegarde sur le travail actuel sans avoir d'abord
comparé les états. Pour inspecter son contenu, lire `manifest.json` et utiliser
`tar -tzf working-files.tar.gz`. Pour récupérer un fichier, l'extraire dans un
répertoire temporaire et comparer avant toute copie. Le patch est relatif au
commit ci-dessus ; ne pas l'appliquer sur un arbre déjà modifié.

## Inventaire de départ

- **Sessions et lecteur** : `backend/services/speech_sessions.py`,
  `backend/routes/speech_sessions.py`, `app/src/lib/audio/speechSessionPlayer.ts`,
  `app/src/lib/hooks/useSpeechSession.ts`, `CapturePill.tsx`,
  `DictateWindow.tsx`, traductions EN/FR et tests associés. Changements :
  démarrage récupérable par UUID, indication attente/génération/lecture,
  relances de lecture de `/speak`, accès à `audio_path` dans le statut.
- **Pont VS Code** : `backend/agent_bridge/narrator.py`, ses deux suites de
  tests et `docs/plans/VSCODE_AGENT_VOICE_BRIDGE.md`. Le pont utilise l'UUID du
  démarrage pour les reprises et l'annulation ; il reste un prototype local.
- **Voxtral** : `backend/backends/voxtral_backend.py`, nouveau
  `backend/services/voxtral_memory.py`, tests Voxtral et affinité MLX, nouveau
  `docs/plans/VOXTRAL_MEMORY_GUARD.md`. La garde prévoit notamment un arrêt du
  processus en dernier recours : vérifier ce comportement sans provoquer d'OOM.
- **Résolution et diagnostic** : `backend/mcp_server/resolve.py`,
  `backend/services/generation.py`, `backend/routes/generations.py`, nouveau
  `backend/utils/frozen.py`, tests associés, `scripts/build-server.sh`. Examiner
  ensemble les priorités moteur/profil, l'erreur audio et le diagnostic de
  binaire remplacé pendant son exécution.
- **Documentation** : modifications de `MCP_SPEECH_STREAMING.md` et du plan du
  pont ; nouveau `SECURITY_SCAN_PLAYBOOK.md`. Ce dernier est distinct de la
  finalisation audio : le relire et le conserver séparément, sans lancer de scan
  ou publier de rapport automatiquement.
- `.mcp.json` : seule la fin de ligne change dans le diff observé. La conserver
  dans la sauvegarde ; ne pas l'inclure dans un commit fonctionnel par défaut.

Les tests et chiffres historiques présents dans les documents décrivent leurs
propres exécutions antérieures. Ils ne remplacent pas une nouvelle validation.

## Règles pour l'agent d'exécution

1. Travailler dans l'arbre existant après avoir contrôlé `git status --short`,
   `git diff` et les fichiers non suivis. Ne pas écraser, nettoyer, stasher ou
   réinitialiser les modifications d'autrui. Si l'état a changé depuis cet
   inventaire, intégrer ces changements dans l'analyse avant toute édition.
2. Lire les fichiers complets et les tests existants avant de corriger. Ne pas
   supprimer un test pour rendre la suite verte ni modifier les contrats sans
   mettre à jour leurs consommateurs et leur documentation.
3. Un test historique peut échouer pour une cause environnementale. Noter la
   commande, l'erreur et la cause ; distinguer échec réel, test ignoré et test
   non exécuté. Ne pas affirmer qu'un scénario matériel a été validé par un mock.
4. Ne pas lancer d'inférence risquant d'épuiser la mémoire, ni construire ou
   remplacer un binaire de l'application pendant qu'un serveur empaqueté utilise
   ce binaire. Les tests d'arrêt d'urgence doivent rester en processus enfant et
   utiliser des mesures simulées. Utiliser des données de test isolées.
5. Ne pas installer, modifier ou restaurer les réglages réels de VS Code par
   automatisme. Utiliser des répertoires et agents synthétiques pour les tests ;
   la validation réelle est un jalon manuel séparé.
6. Ne pas lancer de `just format`, `bun run format`, `check:fix` ou commande
   équivalente sur tout le dépôt : elles modifient les fichiers. Vérifier de
   manière ciblée et conserver les changements existants.
7. Après chaque lot, examiner `git diff --check`, le diff et l'état Git. Si une
   correction dépasse le périmètre, la décrire et la signaler plutôt que de
   démarrer une refonte. Ne pas committer les sauvegardes, secrets, données
   personnelles, modèles ou fichiers de configuration locale accidentels.

## Lot A — Baseline et contrats partagés

1. Vérifier la branche, le SHA, les fichiers suivis et non suivis. Lire ce plan
   ainsi que `MCP_SPEECH_STREAMING.md`, `VSCODE_AGENT_VOICE_BRIDGE.md` et
   `VOXTRAL_MEMORY_GUARD.md`. Vérifier la présence de `backend/venv/bin/python`
   et de `bun`, sans installer de dépendances par défaut.
2. Lancer d'abord les tests ciblés ci-dessous tels quels ; garder les sorties et
   déterminer si les échecs précèdent les corrections. Si un outil manque,
   documenter le blocage au lieu d'inventer un résultat.
3. Lire la résolution moteur dans `resolve.py`, ses appelants `/speak` et
   sessions, et `test_speak_speech_options.py`. Vérifier : moteur explicite >
   moteur intrinsèque pour preset/RVC via le résolveur de génération > moteur
   du profil cloné/conçu > défaut de binding lorsque le profil n'en a pas.
   Confirmer par les tests le comportement des combinaisons invalides. Ne pas
   forcer un moteur sur un profil incompatible.
4. Lire `backend/utils/frozen.py` et ses points d'import pour confirmer que
   l'identité du binaire est capturée avant les imports différés concernés.
   Vérifier que le message n'est enrichi qu'en présence d'un remplacement et
   d'une erreur de chargement reconnue ; une erreur ordinaire reste intelligible.
5. Examiner le `audio_path` du statut de génération, l'absence de fichier et le
   comportement de lecture : aucun audio inexistant ne doit être considéré comme
   lu ou effacé après un échec.

**Tests initiaux recommandés depuis la racine :**

```sh
backend/venv/bin/python -m pytest backend/tests/test_speak_speech_options.py backend/tests/test_audio_failed_generation.py backend/tests/test_frozen_binary_hint.py -q
bun test app/tests/speechSessionPlayer.test.ts
```

**Sortie attendue :** tableau des commandes, résultats et échecs préexistants ;
corrections minimales et tests de comportement si un défaut est confirmé.

## Lot B — Sessions vocales et lecteur

Fichiers principaux : `backend/services/speech_sessions.py`,
`backend/routes/speech_sessions.py`, `backend/tests/test_speech_sessions.py`,
`backend/tests/test_speech_session_api.py`, `app/src/lib/audio/speechSessionPlayer.ts`,
`app/src/lib/hooks/useSpeechSession.ts`, `app/tests/speechSessionPlayer.test.ts`,
`app/src/components/CapturePill/CapturePill.tsx`, `DictateWindow.tsx` et locales.

1. Vérifier le contrat de démarrage UUID : même propriétaire et mêmes options
   récupèrent exactement la même session, y compris après une réponse perdue et
   pour un état terminal encore en cache ; propriétaire ou options différents
   renvoient 409. Un UUID invalide est rejeté à l'entrée REST. Sans UUID, le
   comportement historique reste disponible. Vérifier qu'une autre session ne
   peut pas contourner la réservation et le drainage du nettoyage.
2. Vérifier les séquences d'append : reprises identiques idempotentes, doublon
   différent et trou rejetés, ordre conservé, crédit borné et 429 sans perte de
   texte. Vérifier fin, échec, annulation et expiration, y compris pendant une
   génération en cours. Ne pas annuler brutalement le worker GPU.
3. Vérifier les acquittements : un seul renderer, ordre strict, acquittement
   seulement après `audio.ended`, réponse perdue sans double lecture ; pause avec
   heartbeat sans acquittement ni suppression ; arrêt ou erreur sans annoncer
   une écoute réussie. Tester `keep_audio` et le nettoyage temporaire séparément.
4. Vérifier l'état affiché : attente de texte, préparation audio, lecture réelle
   et pause. Les événements `playing`, `waiting`, `pause` et `ended` doivent
   produire un état cohérent, y compris en cas de réponse serveur périmée.
5. Examiner les nouvelles relances de `/speak` dans `DictateWindow.tsx` : nombre
   borné, annulation des timers à la fermeture ou au changement de génération,
   absence de double lecteur si `onerror` et le rejet de `play()` surviennent
   ensemble, conservation du fichier en cas d'échec. Ajouter un test ciblé si ce
   scénario n'est pas couvert et corriger uniquement si le défaut est reproduit.
6. Vérifier les textes EN/FR et l'absence de régression pour la dictée ordinaire.
   Mettre à jour le plan de sessions si le comportement réel diffère du contrat.

```sh
backend/venv/bin/python -m pytest backend/tests/test_speech_sessions.py backend/tests/test_speech_session_api.py backend/tests/test_mcp_speak.py backend/tests/test_speak_speech_options.py -q
bun test app/tests/speechSessionPlayer.test.ts
npm run typecheck --workspace=@voicebox/app
```

**Passage :** tests pertinents verts ; aucun doublon, acquittement prématuré ou
perte de fichier observé dans les scénarios automatisés ; limites non testées
listées. La lecture sur haut-parleur reste à vérifier au lot E.

## Lot C — Pont VS Code

Fichiers principaux : `backend/agent_bridge/narrator.py`, autres modules de
`backend/agent_bridge/`, `backend/tests/test_agent_bridge.py`,
`backend/tests/test_agent_bridge_wire.py`, `app/tests/agentBridgeInstaller.test.ts`
et `docs/plans/VSCODE_AGENT_VOICE_BRIDGE.md`.

1. Suivre un tour de bout en bout : création UUID, start, append par séquences,
   finish, statut terminal et annulation. Reproduire réponse start perdue,
   connexion interrompue, 429, arrêt pendant start et serveur indisponible.
   Vérifier qu'un start ayant réussi sans réponse ne laisse pas une session
   orpheline ni ne crée un second lecteur.
2. Contrôler la borne des files et délais de reprise ; les erreurs de narration
   ne doivent ni bloquer ni altérer les octets du protocole de l'IDE. Vérifier
   qu'aucun prompt, sortie privée ou contenu d'outil n'est écrit dans les logs.
3. Vérifier les limites documentées : une session globale, aucun renouvellement
   de lease producteur par simple polling, aucun transfert automatique de
   renderer, wrapper Codex expérimental. Ne pas prétendre qu'une API stable de
   l'extension Codex est garantie.
4. Exécuter les tests d'installation sur des fichiers temporaires et confirmer
   que la restauration refuse d'écraser des réglages modifiés depuis l'installation.

```sh
backend/venv/bin/python -m pytest backend/tests/test_agent_bridge.py backend/tests/test_agent_bridge_wire.py -q
bun test app/tests/agentBridgeInstaller.test.ts
```

**Passage :** tests synthétiques verts et documentation cohérente. Le test des
vrais panneaux reste explicitement séparé et n'est jamais déduit du handshake.

## Lot D — Garde mémoire Voxtral et binaire

1. Lire entièrement `backend/services/voxtral_memory.py`, le backend Voxtral et
   leurs tests. Contrôler budget valide et plafonné, empreinte du processus,
   restauration des limites MLX en cas de succès et d'échec, arrêt du thread de
   surveillance et fermeture de l'itérateur. Vérifier qu'aucun appel MLX/Metal
   n'est effectué par le thread de surveillance.
2. Vérifier qu'un dépassement coopératif libère le modèle et permet un
   rechargement ultérieur ; l'arrêt d'urgence est testé uniquement en enfant
   avec mesures simulées. Vérifier qu'aucun texte prononcé n'est journalisé.
3. Contrôler la limite de 320 caractères et le budget audio, la concaténation
   exacte des fenêtres, la fréquence et les erreurs de génération vide ou
   tronquée. Confirmer que les appels directs et le découpage partagé respectent
   le contrat, y compris pour la voix de base d'un profil RVC.
4. Relire `scripts/build-server.sh` et `backend/utils/frozen.py` : l'avertissement
   ne remplace pas l'arrêt effectif du serveur et ne doit pas être présenté comme
   une protection automatique. Tester le diagnostic avec des doubles, pas en
   écrasant un binaire en service.

```sh
backend/venv/bin/python -m pytest backend/tests/test_voxtral_memory.py backend/tests/test_voxtral_backend.py backend/tests/test_mlx_thread_affinity.py backend/tests/test_frozen_binary_hint.py -q
```

**Passage :** tests simulés verts ; absence de fuite ou de régression logique
identifiée. Les mesures historiques du document Voxtral sont conservées comme
résultats datés, sans les présenter comme un nouveau test du binaire actuel.

## Lot E — Vérification de bout en bout et packaging

1. Exécuter les suites ciblées des lots précédents après les dernières corrections,
   puis, si l'environnement le permet :

```sh
backend/venv/bin/python -m pytest backend/tests -q
bun run typecheck
bun run build:web
bun run lint
git diff --check
```

   Documenter durée, échecs, tests ignorés et dépendances/manques matériels. Ne
   pas étendre ou relancer toute la campagne sans motif après un résultat stable.
2. Avant un build empaqueté, identifier l'application et le serveur en cours :
   demander une intervention si leur fermeture dérangerait l'utilisateur ; ne
   pas tuer de processus ni écraser un exécutable utilisé. Construire dans un
   environnement approprié et consigner commande, machine et résultat.
3. Sur l'application reconstruite, tester manuellement : un passage Voxtral
   d'au moins deux minutes et un passage RVC ; pause pendant une phrase, reprise,
   stop, réponse serveur perdue/déconnexion, audio temporaire et conservé ;
   niveau et jointures audibles, pas de double lecteur, fichiers conservés
   lisibles, nettoyage temporaire après drainage. Ne pas promettre du temps réel.
4. Dans chacun des vrais panneaux VS Code disponibles, vérifier une conversation,
   invite de permission, réponse longue, code Markdown, pause/reprise,
   interruption, message suivant, `keep audio` et serveur indisponible. Préserver
   les réglages et le mode de permission de l'utilisateur. Ne pas installer le
   wrapper réel silencieusement. Si impossible, marquer cette validation comme
   non exécutée et garder le pont au statut prototype.
5. Pour Voxtral réel, enregistrer machine, versions, modèle, durée, empreinte et
   qualité à l'écoute ; ne pas essayer de provoquer un OOM ou un arrêt d'urgence.
   La vérification automatisée de l'arrêt d'urgence est celle du lot D.

**Passage :** aucun échec non expliqué sur les contrôles applicables ; les
vérifications manuelles réellement exécutées et les limites sont consignées.
Une validation empaquetée ou matérielle indisponible reste un blocage pour une
promesse de release correspondante, pas nécessairement pour préparer la branche.

## Lot F — Revue et points de reprise

1. Examiner tous les fichiers suivis et non suivis, y compris ce plan et
   `SECURITY_SCAN_PLAYBOOK.md`. Conserver ou exclure `.mcp.json` après examen ;
   son changement observé ne contient qu'un saut de ligne.
2. Répartir les changements en commits cohérents **après** validation :
   contrats communs/diagnostic, sessions et lecteur, pont VS Code, garde Voxtral,
   documentation indépendante. Certains fichiers sont partagés entre lots :
   utiliser une sélection de hunks revue, ou regrouper les lots interdépendants
   plutôt que fabriquer des commits intermédiaires cassés. Tester chaque point
   de reprise si les commits sont destinés à être utilisés séparément.
3. Vérifier `git status`, l'absence de secrets et données locales, et conserver
   la sauvegarde d'origine. Ne pas pousser ni fusionner avec l'officiel dans ce lot.
4. Produire un compte rendu : commits créés, décisions, commandes/résultats,
   tests manuels effectivement faits, lacunes, risques résiduels et SHA final.
   Mettre à jour les documents existants sans transformer leurs validations
   historiques en validations actuelles.

**Définition de terminé :** état propre hormis les exclusions documentées,
fonctionnalités vérifiées au niveau annoncé, changements préservés dans des
commits relisibles, et point de départ identifié pour la future intégration.
