_aujourd'hui, je voudrais être en mesure de construire un code qui permet de faire l'analyse pca des souris par zone_
cela va me permettre de:
1. enlever les zones qui sont aberrante
2. avoir plus de points... moins précis soit-il
3. obtenir de l'info qui aurait pus disparaître quand je fais mon moyennage dans ma fonction python

bon les fichier du jour 4 sont vraiment pêlemêle donc je sais pas trop comment faire pour les trier... au moins j'ai 3 jours 😢

urggg avec mes propres mesure de jour 4 c'est pas très beau:

Variance expliquée par chaque composante :
  PC1 : 31.8%
  PC2 : 17.5%
  PC3 : 9.1%
  Total : 58.4%
  ![[PCA_globale_zone_j4_fixe.png]]
bon, ce que je vais faire c'est trouver plus de composante, on  va essayer avec 5 composante
aussi, grâce a P. J. Caspers et al., j'ai constaté que le spectre se limitait à 1800 cm^-1^ typiquement

il faut noter aussi que je n'ai pas les données du jour 0 ni du jour 4... (genre 1 petri sur 5)

_à venir cette semaine:_
1. comprendre le fonctionnement du laser
2. essayer de prendre des mesures d'échantillon
3. lire, lire, lire
4. faire jour 4 et jour 0

bon puisque que je fais du sens je commence par l'étape 4 qui devrait en fait etre l'étape 1 bc j'en ai besoin

OMG, J'AVAIS OUBLIÉ, MAIS DANIEL A FAIT LE RAMAN LA LUMIERE OUVERTE!! THATS WHY

bon entk, je commence avec un dark a 100 ms d'intégration LUMIERE OFF
je me ne souvenais pas trop comment faire, mais bon j'ai réussi. j'ai mis la plateforme à une hauteur de 8120, merdouille je devrais essayer de voir c combien de hauteur p/r au laser, mais bon assez difficile à calculer
- [x] petri 2
	- [x] s4
	- [x] s5
- [x] petri 3
	- [x] s1
	- [x] s2
	- [x] s3 EST UN MÂLE WHAT!!! a vérifier avec julie
- [x] petri 4
	- [x] s1
	- [x] s2
	- [x] s3
