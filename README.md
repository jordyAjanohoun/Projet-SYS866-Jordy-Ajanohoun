# Projet-SYS866-Jordy-Ajanohoun
Codes pour l'expérimentation du projet du cours SYS866

J'ai moi même implémenté les scripts :
- [GraphCMR/eval.py] en me basant sur le script d'évaluation original : https://github.com/nkolot/GraphCMR/blob/master/eval.py
- hmr/eval_lsp_dataset.py et hmr/eval_up-3d.py à partir du script original de demonstration des auteurs (hmr/demo.py) et de mon script d'évaluation de la méthode CMR (GraphCMR/eval.py)
- eval.py en ne me basant sur rien

Pour chacune des méthodes, voici le répertoire GitHub original et offciel où j'ai récupéré le code des auteurs :
- CMR : https://github.com/nkolot/GraphCMR.git
- NBF : https://github.com/mohomran/neural_body_fitting.git
- HMR : https://github.com/akanazawa/hmr.git


Pour SMPLify il faut se rendre sur la page web du projet et s'enregistrer pour télécharger les modèles et le code : http://smplify.is.tue.mpg.de/

J'ai également eu besoin de OpenPose : https://github.com/CMU-Perceptual-Computing-Lab/openpose.git

Pour NBF, je n'ai pas implémenté de script de test pour les raisons énoncées dans mon rapport final. J'ai simplement suivit les instructions du README.md de la page d'acceuil du répertoire GitHub pour exécuter le code de démonstration comme je l'ai mentionné dans mon rapport final.

Dans le répertoire eval-output se trouve les fichiers .npz où je stock les valeurs d'erreurs selon les différentes métriques pour chaque image des bases de données avec les méthodes CMR et HMR. Je les ai obtenus en exécutant les scripts GraphCMR/eval.py, hmr/eval_lsp_dataset.py et hmr/eval_up-3d.py.
Le script eval.py sert à traiter ces données et à obtenir les résultats qui apparaissent dans mon rapport final.

Les scripts d'évaluation GraphCMR/eval.py, hmr/eval_lsp_dataset.py et hmr/eval_up-3d.py nécessitent de télécharger les modèles pré-entraînés des auteurs ainsi que les bases de données UP-3D et LSP pour être exécutés. Ce n'est pas le cas du script eval.py qui n'a acune dépendance si ce n'est numpy et matplotlib. 

